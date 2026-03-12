# SPDX-License-Identifier: Apache-2.0
"""
Snake pipeline design for the AMD XDNA 2 NPU.

Routes activations through ALL 32 compute tiles in a single serpentine path.
Each tile holds a unique weight matrix (static) and applies
``ReLU(RMSNorm(x, scale) @ W_i)`` exactly once as the activation flows through.

**Architecture**::

    DDR → Shim[0] → Mem[0]
            ↓
      (0,2)→(0,3)→(0,4)→(0,5)   layers 1-4     (W₁ W₂ W₃ W₄)
                           ↓ east
      (1,2)←(1,3)←(1,4)←(1,5)   layers 5-8     (W₅ W₆ W₇ W₈)
        ↓ east
      (2,2)→(2,3)→(2,4)→(2,5)   layers 9-12
                           ↓ east
      ...
      (7,2)←(7,3)←(7,4)←(7,5)   layers 29-32
        ↓
      Mem[7] → Shim[7] → DDR

**Key design principle**: Weights are static residents in each tile's 64KB SRAM.
Only the tiny activation vector (B×H bf16) flows tile-to-tile.

**SRAM budget** (B=8, H=160)::

    Weight+scale (one per tile): (160×160+160)×2 = 50.3 KB
    Input   (one buffer):         8×160×2        = 2.5 KB
    Output  (one buffer):         8×160×2        = 2.5 KB
    Stack + code:                                 ~1.5 KB
    ─────────────────────────────────────────────────────────
    Total:                                        ~56.8 KB  (fits 64 KB)
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib.tap import TensorAccessPattern

# ── Hardware constants ───────────────────────────────────────────────────
NUM_COLUMNS = 8
ROWS_PER_COL = 4       # compute-tile rows (rows 2-5)
TOTAL_TILES = NUM_COLUMNS * ROWS_PER_COL  # 32


# ── Buffer type helpers ──────────────────────────────────────────────────

def _activation_type(B, H):
    """IRON buffer type for one tile's activation (B×H bf16)."""
    return np.ndarray[(B * H,), np.dtype[bfloat16]]


def _weight_type(H):
    """IRON buffer type for one tile's weight + scale (H×H + H bf16)."""
    return np.ndarray[(H * H + H,), np.dtype[bfloat16]]


def _host_activation_type(B, H):
    """Host buffer type for activation (just B×H — single stream, no column split)."""
    return np.ndarray[(B * H,), np.dtype[bfloat16]]


def _host_weights_type(num_tiles, H):
    """Host buffer type for all weights (num_tiles × (H×H + H))."""
    return np.ndarray[(num_tiles * (H * H + H),), np.dtype[bfloat16]]


# ── Snake tile ordering ──────────────────────────────────────────────────

def snake_tile_order(num_cols, rows_per_col=ROWS_PER_COL):
    """Return the list of (col, row) tuples in snake traversal order.

    Even columns: rows go 2→3→4→5 (downward)
    Odd columns:  rows go 5→4→3→2 (upward)

    Returns:
        List of (col, row) tuples, length = num_cols × rows_per_col.
    """
    tiles = []
    for col in range(num_cols):
        if col % 2 == 0:
            rows = range(2, 2 + rows_per_col)   # 2,3,4,5
        else:
            rows = range(2 + rows_per_col - 1, 1, -1)  # 5,4,3,2
        for row in rows:
            tiles.append((col, row))
    return tiles


# ── Kernel definition ────────────────────────────────────────────────────

def _define_kernel(act_ty, weight_ty):
    """Create the fused norm+matmul+ReLU kernel: C = ReLU(RMSNorm(A) × W)."""
    return Kernel("norm_matmul_relu_bf16_bf16", "snake_kernels.a",
                  [act_ty, weight_ty, act_ty])


# ── Worker body ──────────────────────────────────────────────────────────

def _make_worker_body():
    """Worker function for one snake stage: acquire → compute → release."""
    def worker_body(of_in, of_out, of_w, kern):
        x = of_in.acquire(1)
        y = of_out.acquire(1)
        w = of_w.acquire(1)
        kern(x, w, y)
        of_w.release(1)
        of_in.release(1)
        of_out.release(1)
    return worker_body


# ── Top-level design ─────────────────────────────────────────────────────

def snake_pipeline(
    H: int = 160,
    B: int = 8,
    num_cols: int = 8,
):
    """Generate MLIR for a snake pipeline across all compute tiles.

    Args:
        H: Hidden dimension (weight matrix is H×H). Must be divisible by 8.
        B: Batch size (must be divisible by 8, typically 8).
        num_cols: Number of columns (1-8).

    Returns:
        MLIR module.
    """
    assert 1 <= num_cols <= NUM_COLUMNS
    assert B % 8 == 0 and H % 8 == 0
    num_tiles = num_cols * ROWS_PER_COL

    act_ty = _activation_type(B, H)
    wt_ty = _weight_type(H)
    kernel = _define_kernel(act_ty, wt_ty)
    worker_fn = _make_worker_body()

    # Get snake traversal order
    tile_order = snake_tile_order(num_cols)

    # ── Create per-column weight FIFOs (split through MemTile) ──────
    # Each column gets one DDR weight FIFO that splits into 4 tiles
    wt_ddrs = []  # one per column
    wt_endpoints = []  # flat list of per-tile endpoints, in snake order
    col_wt_ty = np.ndarray[(ROWS_PER_COL * (H * H + H),), np.dtype[bfloat16]]

    for col in range(num_cols):
        mem = Tile(col=col, row=1)
        wt_ddr = ObjectFifo(col_wt_ty, name=f"wt_col_{col}", depth=1)
        stage_size = H * H + H
        wt_splits = wt_ddr.cons().split(
            offsets=[stage_size * r for r in range(ROWS_PER_COL)],
            obj_types=[wt_ty] * ROWS_PER_COL,
            names=[f"wt_{col}_{r}" for r in range(ROWS_PER_COL)],
            depths=[1] * ROWS_PER_COL,
            placement=mem,
        )
        wt_ddrs.append(wt_ddr)
        # Map split endpoints to snake order within this column.
        # Split[i] carries the i-th weight from the column's data block.
        # Snake layers within the column are always ordered 0,1,2,3
        # regardless of physical row direction (even=down, odd=up).
        for r in range(ROWS_PER_COL):
            wt_endpoints.append(wt_splits[r])

    # ── Create activation chain ──────────────────────────────────────
    # Input: DDR → MemTile[first_col] → first tile
    first_col = tile_order[0][0]
    act_in_ddr = ObjectFifo(act_ty, name="act_in", depth=1)
    act_in_fwd = act_in_ddr.cons().forward(
        name="act_in_fwd",
        placement=Tile(col=first_col, row=1),
    )

    # Inter-tile FIFOs along the snake
    inter_fifos = []
    for idx in range(num_tiles - 1):
        fifo = ObjectFifo(act_ty, name=f"snake_{idx}", depth=1)
        inter_fifos.append(fifo)

    # Output: last tile → MemTile[last_col] → DDR
    last_col = tile_order[-1][0]
    act_out_ddr = ObjectFifo(act_ty, name="act_out", depth=1)
    act_out_join = act_out_ddr.prod().join(
        offsets=[0], obj_types=[act_ty],
        names=["act_out_j"], depths=[1],
        placement=Tile(col=last_col, row=1),
    )

    # Assemble per-tile input/output endpoints
    tile_inputs = [act_in_fwd]  # first tile reads from DDR
    for fifo in inter_fifos:
        tile_inputs.append(fifo)

    tile_outputs = list(inter_fifos)  # tiles 0..N-2 write to next tile
    tile_outputs.append(act_out_join[0])  # last tile writes to DDR

    # ── Create workers ───────────────────────────────────────────────
    workers = []
    for idx, (col, row) in enumerate(tile_order):
        in_ep = tile_inputs[idx]
        out_ep = tile_outputs[idx]
        wt_ep = wt_endpoints[idx]

        in_cons = in_ep.cons() if hasattr(in_ep, 'cons') else in_ep
        out_prod = out_ep.prod() if hasattr(out_ep, 'prod') else out_ep
        wt_cons = wt_ep.cons() if hasattr(wt_ep, 'cons') else wt_ep

        workers.append(Worker(
            worker_fn,
            fn_args=[in_cons, out_prod, wt_cons, kernel],
            placement=Tile(col=col, row=row),
        ))

    # ── Runtime sequence ─────────────────────────────────────────────
    host_act_ty = _host_activation_type(B, H)
    host_wt_ty = _host_weights_type(num_tiles, H)
    host_out_ty = _host_activation_type(B, H)

    col_wt_size = ROWS_PER_COL * (H * H + H)  # elements per column's weight block

    rt = Runtime()
    with rt.sequence(host_act_ty, host_wt_ty, host_out_ty) as (inp, wts, out):
        rt.start(*workers)

        # Fill input activation
        tg_in = rt.task_group()
        rt.fill(act_in_ddr.prod(), inp, task_group=tg_in)
        rt.finish_task_group(tg_in)

        # Fill weights (one split FIFO per column, all in parallel)
        tg_w = rt.task_group()
        for col in range(num_cols):
            tap = TensorAccessPattern(
                (1, num_tiles * (H * H + H)),
                col * col_wt_size,
                [1, ROWS_PER_COL, H + 1, H], [0, H * H + H, H, 1],
            )
            rt.fill(wt_ddrs[col].prod(), wts, tap, task_group=tg_w)
        rt.finish_task_group(tg_w)

        # Drain output
        tg_out = rt.task_group()
        rt.drain(act_out_ddr.cons(), out, wait=True, task_group=tg_out)
        rt.finish_task_group(tg_out)

    # Compile
    program = Program(NPU2(), rt)
    return program.resolve_program(SequentialPlacer())


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate MLIR for the snake pipeline design.")
    parser.add_argument("--H", type=int, default=160,
                        help="Hidden dimension (default: 160)")
    parser.add_argument("--B", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--cols", type=int, default=8,
                        help="Number of columns, 1-8 (default: 8)")
    parser.add_argument("-o", "--output", default="build/snake_mlp.mlir",
                        help="Output MLIR path")
    args = parser.parse_args()

    module = snake_pipeline(H=args.H, B=args.B, num_cols=args.cols)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(str(module))
    print(f"Written to {out_path}")
