# SPDX-License-Identifier: Apache-2.0
"""
Recurrent MLP design for the AMD XDNA 2 NPU.

This module generates MLIR code for a recurrent neural network mapped to
the physical tile array of the NPU. Each compute tile holds one copy of a
shared weight matrix W in its 64 KB SRAM and applies ``ReLU(x @ W)`` in a
tight hardware loop, never touching DDR until the final result is copied out.

**Architecture overview**::

    DDR input ──► Shim DMA ──► MemTile split ──► Compute tiles (rows 2-4)
                                                       │
                                                       │ hardware loop:
                                                       │   y = ReLU(x @ W)
                                                       │   x = ReLU(y @ W)
                                                       │   (repeat num_iters)
                                                       ▼
    DDR output ◄── Shim DMA ◄── MemTile join ◄── Compute tiles

**Key constraints**:

- Each compute tile has ~64 KB SRAM. With H=128 the weight matrix alone
  is 32 KB, leaving room for two activation buffers (4 KB each) plus stack.
- MemTiles (row 1) have ~6 northward master ports, limiting us to 3 compute
  rows (24 tiles). A 4th row exceeds the routing budget.
- Shim DMA buffer-descriptor sizes are 10-bit (max 1024 elements), so
  tensor access patterns must be factored into dimensions ≤ 1024.
- ObjectFIFO acquire/release *inside* ``range_()`` causes deadlock. We
  acquire all FIFOs *before* the loop and release *after*.

**Depth**: Each invocation computes ``2 × num_iters`` matmul+ReLU steps
(two per loop body, ping-ponging between buffers A and B).

**SRAM budget** (H=128, B=16)::

    Weight   (depth=1): 128×128×2 = 32 KB
    Input    (depth=1):  16×128×2 =  4 KB
    Output   (depth=1):  16×128×2 =  4 KB
    Stack:                          ~1 KB
    ──────────────────────────────────────
    Total:                         ~41 KB  (fits 64 KB)
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern

# ── Hardware constants ───────────────────────────────────────────────────
NUM_COLUMNS = 8   # columns in the XDNA 2 tile array
NUM_ROWS = 4      # compute-tile rows (rows 2-5)
MAX_TILES = NUM_COLUMNS * (NUM_ROWS - 1)  # 24 (MemTile routing limit)


# ── Validation ───────────────────────────────────────────────────────────

def _validate_config(num_tiles, num_iters, B, H):
    """Check that configuration fits the hardware constraints."""
    assert 1 <= num_tiles <= MAX_TILES, (
        f"num_tiles={num_tiles} exceeds max {MAX_TILES} "
        f"(MemTile routing limit is 3 compute rows)"
    )
    assert num_tiles <= NUM_COLUMNS or num_tiles % NUM_COLUMNS == 0, (
        f"For >8 tiles, num_tiles must be a multiple of {NUM_COLUMNS} "
        f"(got {num_tiles})"
    )
    assert B % 16 == 0 and H % 16 == 0, (
        f"B={B} and H={H} must be multiples of 16 "
        f"(AIE matmul tile alignment)"
    )
    assert num_iters >= 1, "Need at least 1 loop iteration"


# ── Buffer type helpers ──────────────────────────────────────────────────

def _activation_type(B, H):
    """IRON buffer type for one tile's activation vector (B×H bf16)."""
    return np.ndarray[(B * H,), np.dtype[bfloat16]]


def _weight_type(H):
    """IRON buffer type for one tile's weight matrix (H×H bf16)."""
    return np.ndarray[(H * H,), np.dtype[bfloat16]]


def _column_activation_type(tiles_per_col, B, H):
    """IRON buffer type for all activations in one column (tiles_per_col × B × H)."""
    return np.ndarray[(tiles_per_col * B * H,), np.dtype[bfloat16]]


def _host_type(num_tiles, B, H):
    """IRON buffer type for the full host-side tensor (num_tiles × B × H)."""
    return np.ndarray[(num_tiles * B * H,), np.dtype[bfloat16]]


# ── Kernel definitions ───────────────────────────────────────────────────

def _define_kernels(act_ty, weight_ty):
    """Create the four tile-level kernel objects.

    All kernels are linked from ``mlp_kernels.a`` which bundles:
    - ``mm.cc``  (from IRON): C += A × B  (bf16 matmul, accumulates)
    - ``mlp_kernels.cc`` (ours): zero, ReLU, copy

    Returns:
        Tuple of (zero, matmul, relu, copy) Kernel objects.
    """
    zero = Kernel("zero_bf16", "mlp_kernels.a", [act_ty])
    matmul = Kernel("matmul_bf16_bf16", "mlp_kernels.a",
                     [act_ty, weight_ty, act_ty])
    relu = Kernel("relu_inplace_bf16", "mlp_kernels.a",
                   [act_ty, np.int32])
    copy = Kernel("copy_bf16", "mlp_kernels.a",
                   [act_ty, act_ty, np.int32])
    return zero, matmul, relu, copy


# ── FIFO topology ────────────────────────────────────────────────────────

def _create_direct_fifos(col, act_ty, weight_ty):
    """Create direct DDR-to-tile FIFOs for a single-row column.

    When only one compute row is used, data flows straight from the shim
    DMA to the compute tile with no MemTile intermediary.

    Returns:
        (tile_fifos, ddr_fifos) where:
        - tile_fifos: dict {(col, 0): fifo} for weight/input/output
        - ddr_fifos: (ddr_weight, ddr_input, ddr_output) FIFOs
    """
    weight_fifo = ObjectFifo(weight_ty, name=f"w_{col}", depth=1)
    input_fifo = ObjectFifo(act_ty, name=f"in_{col}", depth=1)
    output_fifo = ObjectFifo(act_ty, name=f"out_{col}", depth=1)

    tile_fifos = {
        "weight": {(col, 0): weight_fifo},
        "input":  {(col, 0): input_fifo},
        "output": {(col, 0): output_fifo},
    }
    ddr_fifos = (weight_fifo, input_fifo, output_fifo)
    return tile_fifos, ddr_fifos


def _create_memtile_fifos(col, tiles_per_col, act_ty, weight_ty, B, H):
    """Create MemTile-routed FIFOs for a multi-row column.

    When multiple compute rows are used, the MemTile (row 1) acts as
    an intermediary:
    - Weights are **broadcast** via ``forward()`` to all rows.
    - Inputs are **split** so each row gets its own slice.
    - Outputs are **joined** back into one contiguous buffer.

    Returns:
        (tile_fifos, ddr_fifos) — same structure as _create_direct_fifos.
    """
    col_act_ty = _column_activation_type(tiles_per_col, B, H)
    mem = Tile(col=col, row=1)
    activation_size = B * H

    # Weight: broadcast to all rows in this column
    wt_ddr = ObjectFifo(weight_ty, name=f"w_l3_{col}", depth=1)
    wt_broadcast = wt_ddr.cons().forward(name=f"w_fwd_{col}", placement=mem)

    # Input: split into per-row slices
    in_ddr = ObjectFifo(col_act_ty, name=f"in_l3_{col}", depth=1)
    in_splits = in_ddr.cons().split(
        offsets=[activation_size * r for r in range(tiles_per_col)],
        obj_types=[act_ty] * tiles_per_col,
        names=[f"in_r{r}_{col}" for r in range(tiles_per_col)],
        depths=[1] * tiles_per_col,
        placement=mem,
    )

    # Output: join from per-row slices
    out_ddr = ObjectFifo(col_act_ty, name=f"out_l3_{col}", depth=1)
    out_joins = out_ddr.prod().join(
        offsets=[activation_size * r for r in range(tiles_per_col)],
        obj_types=[act_ty] * tiles_per_col,
        names=[f"out_r{r}_{col}" for r in range(tiles_per_col)],
        depths=[1] * tiles_per_col,
        placement=mem,
    )

    tile_fifos = {"weight": {}, "input": {}, "output": {}}
    for r in range(tiles_per_col):
        tile_fifos["weight"][(col, r)] = wt_broadcast
        tile_fifos["input"][(col, r)] = in_splits[r]
        tile_fifos["output"][(col, r)] = out_joins[r]

    ddr_fifos = (wt_ddr, in_ddr, out_ddr)
    return tile_fifos, ddr_fifos


# ── Worker definition ────────────────────────────────────────────────────

def _make_worker_body(num_iters, activation_size):
    """Return a worker function that runs the recurrent MLP loop.

    Each worker:
    1. Acquires its input (x), output (y), and weight (W) buffers.
    2. Loops ``num_iters`` times, ping-ponging between x and y:
       - y = ReLU(x @ W)
       - x = ReLU(y @ W)
    3. Copies the final result to the output buffer and releases all FIFOs.

    CRITICAL: All FIFO acquire/release must happen *outside* the loop.
    Putting them inside ``range_()`` causes a deadlock in the DMA engine.
    """
    def worker_body(of_in, of_out, of_w, zero_fn, mm_fn, relu_fn, copy_fn):
        x = of_in.acquire(1)
        y = of_out.acquire(1)
        w = of_w.acquire(1)

        for _ in range_(num_iters):
            zero_fn(y)
            mm_fn(x, w, y)       # y = x @ W  (mm.cc accumulates: y += x @ W)
            relu_fn(y, activation_size)

            zero_fn(x)
            mm_fn(y, w, x)       # x = y @ W
            relu_fn(x, activation_size)

        copy_fn(x, y, activation_size)  # copy final result to output buffer

        of_w.release(1)
        of_in.release(1)
        of_out.release(1)

    return worker_body


# ── Tensor access patterns ───────────────────────────────────────────────

def _column_activation_tap(col, num_tiles, tiles_per_col, B, H):
    """Build a TensorAccessPattern that selects one column's activations.

    The host buffer is laid out as [tile0_col0, tile1_col1, ..., tile7_col7,
    tile8_col0, tile9_col1, ...]. To select all slices belonging to a single
    column, we need a strided gather pattern.

    DMA buffer-descriptor size fields are 10-bit (max 1024), so we factor
    B×H into two dimensions [B, H] to keep every size ≤ 1024.
    """
    activation_size = B * H
    total_size = num_tiles * activation_size

    if tiles_per_col == 1:
        # Single-row: column col is a contiguous slice
        return TensorAccessPattern(
            (1, total_size),
            col * activation_size,
            [1, 1, B, H], [0, 0, H, 1],
        )
    else:
        # Multi-row: gather tiles_per_col non-contiguous slices
        return TensorAccessPattern(
            (1, total_size),
            col * activation_size,
            [1, tiles_per_col, B, H],
            [0, NUM_COLUMNS * activation_size, H, 1],
        )


def _weight_tap(H):
    """Build a TensorAccessPattern for the weight matrix.

    Weights are the same for every column, so the pattern always starts
    at offset 0 and reads the full H×H matrix.
    """
    return TensorAccessPattern(
        (1, H * H), 0,
        [1, 1, H, H], [0, 0, H, 1],
    )


# ── Runtime sequence (DMA fill/drain) ────────────────────────────────────

def _create_runtime_sequence(rt, workers, ddr_in, ddr_wt, ddr_out,
                             num_tiles, tiles_per_col, B, H):
    """Define the host-side DMA transfers that feed and drain the tile array.

    The sequence:
    1. Start all workers (they block on FIFO acquire).
    2. Fill input buffers — one DMA per column (parallel via task_group).
    3. Fill weight buffers — broadcast same W to all columns (parallel).
    4. Drain output buffers — one DMA per column (parallel, with wait).
    """
    n_cols = len(ddr_in)
    input_ty = _host_type(num_tiles, B, H)
    weight_ty = _weight_type(H)
    output_ty = _host_type(num_tiles, B, H)

    with rt.sequence(input_ty, weight_ty, output_ty) as (inp, wts, out):
        rt.start(*workers)

        # Fill inputs (one DMA per column, all in parallel)
        tg_in = rt.task_group()
        for col in range(n_cols):
            tap = _column_activation_tap(col, num_tiles, tiles_per_col, B, H)
            rt.fill(ddr_in[col].prod(), inp, tap, task_group=tg_in)
        rt.finish_task_group(tg_in)

        # Broadcast weights to all columns
        tg_w = rt.task_group()
        w_tap = _weight_tap(H)
        for col in range(n_cols):
            rt.fill(ddr_wt[col].prod(), wts, w_tap, task_group=tg_w)
        rt.finish_task_group(tg_w)

        # Drain outputs (one DMA per column, all in parallel)
        tg_out = rt.task_group()
        for col in range(n_cols):
            tap = _column_activation_tap(col, num_tiles, tiles_per_col, B, H)
            rt.drain(ddr_out[col].cons(), out, tap,
                     wait=True, task_group=tg_out)
        rt.finish_task_group(tg_out)


# ── Top-level design function ────────────────────────────────────────────

def recurrent_mlp(
    H: int = 128,
    B: int = 16,
    num_tiles: int = 24,
    num_iters: int = 1000,
):
    """Generate an MLIR module for the recurrent MLP on the XDNA 2 NPU.

    This is the entry point called by IRON's artifact system. It wires up
    kernels, FIFOs, workers, and DMA transfers, then compiles to an MLIR
    module ready for ``aiecc`` to lower to a bitstream.

    Args:
        H: Hidden dimension (weight matrix is H×H).
        B: Batch size per tile (total batch = num_tiles × B).
        num_tiles: Number of parallel compute tiles (1–24).
        num_iters: Hardware loop count (effective depth = 2 × num_iters).

    Returns:
        MLIR module (``ir.Module``).
    """
    _validate_config(num_tiles, num_iters, B, H)

    n_cols = min(num_tiles, NUM_COLUMNS)
    tiles_per_col = num_tiles // n_cols if num_tiles > NUM_COLUMNS else 1

    # Buffer types
    act_ty = _activation_type(B, H)
    weight_ty = _weight_type(H)

    # Kernels
    zero, matmul, relu, copy = _define_kernels(act_ty, weight_ty)

    # Build per-column FIFO topology
    all_tile_w, all_tile_in, all_tile_out = {}, {}, {}
    ddr_in, ddr_wt, ddr_out = [], [], []

    for col in range(n_cols):
        if tiles_per_col == 1:
            tile_fifos, (ddr_w, ddr_i, ddr_o) = \
                _create_direct_fifos(col, act_ty, weight_ty)
        else:
            tile_fifos, (ddr_w, ddr_i, ddr_o) = \
                _create_memtile_fifos(col, tiles_per_col, act_ty, weight_ty, B, H)

        all_tile_w.update(tile_fifos["weight"])
        all_tile_in.update(tile_fifos["input"])
        all_tile_out.update(tile_fifos["output"])
        ddr_wt.append(ddr_w)
        ddr_in.append(ddr_i)
        ddr_out.append(ddr_o)

    # Create workers — one per tile, all running the same recurrent body
    worker_fn = _make_worker_body(num_iters, B * H)
    workers = []
    for col in range(n_cols):
        for row in range(tiles_per_col):
            workers.append(Worker(
                worker_fn,
                fn_args=[
                    all_tile_in[(col, row)].cons(),
                    all_tile_out[(col, row)].prod(),
                    all_tile_w[(col, row)].cons(),
                    zero, matmul, relu, copy,
                ],
                placement=Tile(col=col, row=2 + row),
            ))

    # Runtime DMA sequence
    rt = Runtime()
    _create_runtime_sequence(rt, workers, ddr_in, ddr_wt, ddr_out,
                             num_tiles, tiles_per_col, B, H)

    # Compile to MLIR
    program = Program(NPU2(), rt)
    return program.resolve_program(SequentialPlacer())


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate MLIR for the recurrent MLP design.")
    parser.add_argument("--H", type=int, default=128,
                        help="Hidden dimension (default: 128)")
    parser.add_argument("--B", type=int, default=16,
                        help="Batch size per tile (default: 16)")
    parser.add_argument("--tiles", type=int, default=24,
                        help="Number of compute tiles, 1-24 (default: 24)")
    parser.add_argument("--iters", type=int, default=1000,
                        help="Hardware loop iterations (default: 1000)")
    parser.add_argument("-o", "--output", default="build/recurrent_mlp.mlir",
                        help="Output MLIR path (default: build/recurrent_mlp.mlir)")
    args = parser.parse_args()

    module = recurrent_mlp(
        H=args.H, B=args.B,
        num_tiles=args.tiles, num_iters=args.iters,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(str(module))
    print(f"Written to {out_path}")
