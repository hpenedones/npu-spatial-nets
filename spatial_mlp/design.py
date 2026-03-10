# SPDX-License-Identifier: Apache-2.0
"""
Recurrent MLP: On-chip looping computation on the XDNA 2 NPU.

Architecture:
  - Up to 32 compute tiles (rows 2-5, 8 columns), each running independently
  - Single weight matrix W loaded once per tile, held in tile SRAM
  - Hardware loop (scf.for) applies ReLU(x @ W) repeatedly on-chip
  - Activation ping-pongs between two SRAM buffers (A→B, B→A)
  - DDR I/O only at start (input) and end (output)

Multi-row support (>8 tiles):
  - Input: DDR → MemTile (row 1) → split() to compute tiles in same column
  - Weight: DDR → MemTile → forward() (broadcast to all tiles in column)
  - Output: compute tiles → join() at MemTile → DDR
  - This keeps shim DMA channels to 3 per column regardless of row count

Effective depth per invocation = 2 * num_iters (two matmul+relu per loop body).

Tile SRAM budget per tile (~64 KB):
  - Weight FIFO (depth=1): H*H*2 bytes     = 32 KB for H=128
  - Input FIFO  (depth=1): B*H*2 bytes     =  4 KB for B=16, H=128
  - Output FIFO (depth=1): B*H*2 bytes     =  4 KB for B=16, H=128
  - Stack:                                   =  1 KB
  - Total:                                   = 41 KB (fits 64 KB)
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern

from spatial_mlp import to_tiled, from_tiled

N_COLS = 8
N_ROWS = 4  # compute rows 2-5


def recurrent_mlp(
    H: int = 128,
    B: int = 16,
    num_tiles: int = 24,
    num_iters: int = 1000,
):
    """
    Generate MLIR for a recurrent MLP on up to 24 compute tiles.

    For <= 8 tiles: direct FIFOs (one per tile).
    For > 8 tiles: MemTile split/forward/join per column.

    Note: 32 tiles (4 rows) exceeds MemTile stream switch capacity
    (6 master ports northward). 24 tiles (3 rows) is the practical max.

    Args:
        H: Hidden dimension (weight matrix is H×H)
        B: Batch size per tile
        num_tiles: Number of parallel tiles (1-24, multiples of 8 for >8)
        num_iters: Hardware loop count (depth = 2 * num_iters)
    """
    assert 1 <= num_tiles <= N_COLS * (N_ROWS - 1), \
        f"Max {N_COLS * (N_ROWS - 1)} tiles (MemTile routing limit)"
    assert num_tiles <= N_COLS or num_tiles % N_COLS == 0, \
        f"For >8 tiles, must be multiple of 8 (got {num_tiles})"
    assert B % 16 == 0 and H % 16 == 0
    assert num_iters >= 1

    n_cols = min(num_tiles, N_COLS)
    tiles_per_col = num_tiles // n_cols if num_tiles > N_COLS else 1

    dtype = bfloat16

    # Buffer types
    act_ty = np.ndarray[(B * H,), np.dtype[dtype]]
    weight_ty = np.ndarray[(H * H,), np.dtype[dtype]]
    col_act_ty = np.ndarray[(tiles_per_col * B * H,), np.dtype[dtype]]

    # Host-side tensor types
    input_ty = np.ndarray[(num_tiles * B * H,), np.dtype[dtype]]
    output_ty = np.ndarray[(num_tiles * B * H,), np.dtype[dtype]]

    # -- Kernels ----------------------------------------------------------
    zero_kernel = Kernel("zero_bf16", "mlp_kernels.a", [act_ty])
    matmul_kernel = Kernel(
        "matmul_bf16_bf16", "mlp_kernels.a",
        [act_ty, weight_ty, act_ty],
    )
    relu_kernel = Kernel(
        "relu_inplace_bf16", "mlp_kernels.a",
        [act_ty, np.int32],
    )
    copy_kernel = Kernel(
        "copy_bf16", "mlp_kernels.a",
        [act_ty, act_ty, np.int32],
    )

    # -- Per-tile FIFO handles (populated by direct or MemTile path) ------
    tile_w = {}    # (col, row_off) → ObjectFifo for weight
    tile_in = {}   # (col, row_off) → ObjectFifo for input
    tile_out = {}  # (col, row_off) → ObjectFifo for output

    # DDR-level FIFOs (for runtime fill/drain)
    ddr_in = []
    ddr_wt = []
    ddr_out = []

    for col in range(n_cols):
        if tiles_per_col == 1:
            # --- Direct FIFOs (single row, no MemTile) ---
            w_fifo = ObjectFifo(weight_ty, name=f"w_{col}", depth=1)
            in_fifo = ObjectFifo(act_ty, name=f"in_{col}", depth=1)
            out_fifo = ObjectFifo(act_ty, name=f"out_{col}", depth=1)

            tile_w[(col, 0)] = w_fifo
            tile_in[(col, 0)] = in_fifo
            tile_out[(col, 0)] = out_fifo

            ddr_wt.append(w_fifo)
            ddr_in.append(in_fifo)
            ddr_out.append(out_fifo)
        else:
            # --- MemTile intermediary (multi-row) ---
            mem = Tile(col=col, row=1)

            # Weight: broadcast via forward()
            wt_l3 = ObjectFifo(weight_ty, name=f"w_l3_{col}", depth=1)
            wt_fwd = wt_l3.cons().forward(
                name=f"w_fwd_{col}",
                placement=mem,
            )
            for r in range(tiles_per_col):
                tile_w[(col, r)] = wt_fwd  # all rows share broadcast
            ddr_wt.append(wt_l3)

            # Input: split to rows
            in_l3 = ObjectFifo(col_act_ty, name=f"in_l3_{col}", depth=1)
            in_splits = in_l3.cons().split(
                offsets=[B * H * r for r in range(tiles_per_col)],
                obj_types=[act_ty] * tiles_per_col,
                names=[f"in_r{r}_{col}" for r in range(tiles_per_col)],
                depths=[1] * tiles_per_col,
                placement=mem,
            )
            for r in range(tiles_per_col):
                tile_in[(col, r)] = in_splits[r]
            ddr_in.append(in_l3)

            # Output: join from rows
            out_l3 = ObjectFifo(col_act_ty, name=f"out_l3_{col}", depth=1)
            out_joins = out_l3.prod().join(
                offsets=[B * H * r for r in range(tiles_per_col)],
                obj_types=[act_ty] * tiles_per_col,
                names=[f"out_r{r}_{col}" for r in range(tiles_per_col)],
                depths=[1] * tiles_per_col,
                placement=mem,
            )
            for r in range(tiles_per_col):
                tile_out[(col, r)] = out_joins[r]
            ddr_out.append(out_l3)

    # -- Worker body (identical for all tiles) ----------------------------
    def worker_body(of_in, of_out, of_w, zero_fn, mm_fn, relu_fn, cp_fn):
        x = of_in.acquire(1)
        y = of_out.acquire(1)
        w = of_w.acquire(1)

        for _ in range_(num_iters):
            zero_fn(y)
            mm_fn(x, w, y)
            relu_fn(y, B * H)
            zero_fn(x)
            mm_fn(y, w, x)
            relu_fn(x, B * H)

        cp_fn(x, y, B * H)

        of_w.release(1)
        of_in.release(1)
        of_out.release(1)

    # -- Create workers on all tiles --------------------------------------
    workers = []
    for col in range(n_cols):
        for r in range(tiles_per_col):
            workers.append(Worker(
                worker_body,
                fn_args=[
                    tile_in[(col, r)].cons(),
                    tile_out[(col, r)].prod(),
                    tile_w[(col, r)].cons(),
                    zero_kernel, matmul_kernel, relu_kernel, copy_kernel,
                ],
                placement=Tile(col=col, row=2 + r),
            ))

    # -- Runtime sequence -------------------------------------------------
    chunk_act = B * H
    chunk_wt = H * H

    rt = Runtime()
    with rt.sequence(input_ty, weight_ty, output_ty) as (inp, wts, out):
        rt.start(*workers)

        # Fill inputs (per column)
        # Note: DMA BD size fields are 10-bit (max 1024), so we factorize
        # chunk_act = B*H into [B, H] to keep all sizes ≤ 1024.
        tg_in = rt.task_group()
        for col in range(n_cols):
            if tiles_per_col == 1:
                tap = TensorAccessPattern(
                    (1, num_tiles * chunk_act),
                    col * chunk_act,
                    [1, 1, B, H], [0, 0, H, 1],
                )
            else:
                # Gather tiles_per_col non-contiguous slices from host buffer
                # Host layout: tile0(col0,row0), tile1(col1,row0), ...
                # Column col's data at offsets: col, col+8, col+16, col+24
                tap = TensorAccessPattern(
                    (1, num_tiles * chunk_act),
                    col * chunk_act,
                    [1, tiles_per_col, B, H],
                    [0, N_COLS * chunk_act, H, 1],
                )
            rt.fill(ddr_in[col].prod(), inp, tap, task_group=tg_in)
        rt.finish_task_group(tg_in)

        # Fill weights (same W to every column)
        tg_w = rt.task_group()
        weight_tap = TensorAccessPattern(
            (1, chunk_wt), 0,
            [1, 1, H, H], [0, 0, H, 1],
        )
        for col in range(n_cols):
            rt.fill(ddr_wt[col].prod(), wts, weight_tap, task_group=tg_w)
        rt.finish_task_group(tg_w)

        # Drain outputs (per column)
        tg_out = rt.task_group()
        for col in range(n_cols):
            if tiles_per_col == 1:
                tap = TensorAccessPattern(
                    (1, num_tiles * chunk_act),
                    col * chunk_act,
                    [1, 1, B, H], [0, 0, H, 1],
                )
            else:
                tap = TensorAccessPattern(
                    (1, num_tiles * chunk_act),
                    col * chunk_act,
                    [1, tiles_per_col, B, H],
                    [0, N_COLS * chunk_act, H, 1],
                )
            rt.drain(
                ddr_out[col].cons(), out, tap,
                wait=True, task_group=tg_out,
            )
        rt.finish_task_group(tg_out)

    dev = NPU2()
    program = Program(dev, rt)
    module = program.resolve_program(SequentialPlacer())
    return module


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--tiles", type=int, default=24)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("-o", "--output", type=str, default="build/recurrent_mlp.mlir")
    args = p.parse_args()

    module = recurrent_mlp(
        H=args.H, B=args.B,
        num_tiles=args.tiles, num_iters=args.iters,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(str(module))
    print(f"Written to {out_path}")
