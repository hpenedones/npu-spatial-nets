"""
IRON design for the residual MLP snake pipeline.

Generates the MLIR that programs all 32 tiles into a serpentine chain.
Each tile receives activations from its predecessor, applies
``y = relu(x @ W) + x``, and passes the result to its successor.

Weights are loaded once per NPU call via per-column split FIFOs through
the MemTile.  Activations are tiny (B×H bf16 ≈ 2.5 KB) and flow
tile-to-tile through direct ObjectFIFOs.
"""

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib.tap import TensorAccessPattern

# ── Constants ────────────────────────────────────────────────────────────
ROWS_PER_COL = 4  # compute rows per column (rows 2–5)


def snake_tile_order(num_cols):
    """Serpentine traversal of the compute tile array.

    Even columns go downward (rows 2→5), odd columns go upward (rows 5→2).
    Returns list of (col, row) tuples in activation-flow order.
    """
    tiles = []
    for col in range(num_cols):
        rows = range(2, 6) if col % 2 == 0 else range(5, 1, -1)
        for row in rows:
            tiles.append((col, row))
    return tiles


def snake_pipeline(H=160, B=8, num_cols=8, archive_name="resmlp_kernel.a"):
    """Generate an IRON program for the residual MLP snake pipeline.

    Args:
        H: Hidden dimension. Weight matrix per tile is H×H.
        B: Batch size (samples processed in parallel).
        num_cols: Number of NPU columns to use (1–8).

    Returns:
        Resolved MLIR module ready for compilation.
    """
    assert 1 <= num_cols <= 8
    assert B % 8 == 0 and H % 8 == 0
    num_tiles = num_cols * ROWS_PER_COL
    tile_order = snake_tile_order(num_cols)

    # ── Buffer types ─────────────────────────────────────────────────
    act_ty = np.ndarray[(B * H,), np.dtype[bfloat16]]    # activation
    wt_ty = np.ndarray[(H * H,), np.dtype[bfloat16]]     # one tile's weight
    col_wt_ty = np.ndarray[(ROWS_PER_COL * H * H,), np.dtype[bfloat16]]

    # ── Kernel ───────────────────────────────────────────────────────
    kernel = Kernel(
        "matmul_relu_skip_infer_bf16", archive_name,
        [act_ty, wt_ty, act_ty],
    )

    # ── Weight FIFOs: DDR → MemTile split → 4 tiles per column ──────
    wt_ddrs = []       # one DDR FIFO per column
    wt_endpoints = []  # flat list, one per tile in snake order

    for col in range(num_cols):
        wt_ddr = ObjectFifo(col_wt_ty, name=f"wt_col{col}", depth=1)
        splits = wt_ddr.cons().split(
            offsets=[H * H * r for r in range(ROWS_PER_COL)],
            obj_types=[wt_ty] * ROWS_PER_COL,
            names=[f"wt_{col}_{r}" for r in range(ROWS_PER_COL)],
            depths=[1] * ROWS_PER_COL,
            placement=Tile(col=col, row=1),
        )
        wt_ddrs.append(wt_ddr)
        for r in range(ROWS_PER_COL):
            wt_endpoints.append(splits[r])

    # ── Activation FIFOs: DDR → tiles → DDR ──────────────────────────
    act_in = ObjectFifo(act_ty, name="act_in", depth=1)
    act_out = ObjectFifo(act_ty, name="act_out", depth=1)
    # One intermediate activation FIFO between each consecutive pair of snake tiles.
    act_inter = [ObjectFifo(act_ty, name=f"act_{i}", depth=1)
                 for i in range(num_tiles - 1)]

    # ── Worker function (same for all 32 tiles) ─────────────────────
    def worker_fn(of_in, of_out, of_w, kern):
        x = of_in.acquire(1)
        y = of_out.acquire(1)
        w = of_w.acquire(1)
        kern(x, w, y)
        of_w.release(1)
        of_in.release(1)
        of_out.release(1)

    # ── Create workers on each tile ──────────────────────────────────
    workers = []
    for idx, (col, row) in enumerate(tile_order):
        # Input: DDR for first tile, previous tile's output otherwise
        if idx == 0:
            in_ep = act_in.cons()
        else:
            in_ep = act_inter[idx - 1].cons()

        # Output: DDR for last tile, next tile's input otherwise
        if idx == num_tiles - 1:
            out_ep = act_out.prod()
        else:
            out_ep = act_inter[idx].prod()

        # Weight: from the per-column split
        wt_ep = wt_endpoints[idx]
        wt_cons = wt_ep.cons() if hasattr(wt_ep, 'cons') else wt_ep

        workers.append(Worker(
            worker_fn,
            fn_args=[in_ep, out_ep, wt_cons, kernel],
            placement=Tile(col=col, row=row),
        ))

    # ── Runtime sequence ─────────────────────────────────────────────
    host_act_ty = np.ndarray[(B * H,), np.dtype[bfloat16]]
    host_wt_ty = np.ndarray[(num_tiles * H * H,), np.dtype[bfloat16]]
    host_out_ty = np.ndarray[(B * H,), np.dtype[bfloat16]]

    # Number of packed weight elements sent to one column before the MemTile split.
    col_wt_elems = ROWS_PER_COL * H * H

    rt = Runtime()
    with rt.sequence(host_act_ty, host_wt_ty, host_out_ty) as (inp, wts, out):
        rt.start(*workers)

        tg = rt.task_group()

        # Send input activations to first tile
        rt.fill(act_in.prod(), inp, task_group=tg)

        # Send weights to each column (split in MemTile → 4 tiles)
        for col in range(num_cols):
            tap = TensorAccessPattern(
                (1, num_tiles * H * H),
                col * col_wt_elems,
                [1, ROWS_PER_COL, H, H],
                [0, H * H, H, 1],
            )
            rt.fill(wt_ddrs[col].prod(), wts, tap, task_group=tg)

        # Collect output from last tile
        rt.drain(act_out.cons(), out, wait=True, task_group=tg)

        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="Generate snake pipeline MLIR")
    p.add_argument("--H", type=int, default=160)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("-o", "--output", default="build/resmlp.mlir")
    args = p.parse_args()

    module = snake_pipeline(H=args.H, B=args.B, num_cols=args.cols)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(str(module))
    print(f"Written {args.output}")
