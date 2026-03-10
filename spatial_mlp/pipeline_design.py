# SPDX-License-Identifier: Apache-2.0
"""
Streaming pipeline design for the AMD XDNA 2 NPU.

This module generates MLIR code for a 32-stage pipeline where each compute
tile holds its *own* weight matrix and applies ``ReLU(x @ W_i)`` exactly
once.  Data flows vertically through 4 tiles per column, then the host
orchestrates column-to-column chaining (4 layers per NPU call, 8 calls
for a full 32-layer pass).

**Architecture overview**::

    Per-column pipeline (one NPU invocation covers all 8 columns in parallel):

    DDR input ──► Shim DMA ──► MemTile ──► Tile(col, row=2)  W0
                                                  │
                                           Tile(col, row=3)  W1
                                                  │
                                           Tile(col, row=4)  W2
                                                  │
                                           Tile(col, row=5)  W3
                                                  │
    DDR output ◄── Shim DMA ◄── MemTile ◄─────────┘

    All 8 columns run the same 4-stage pipeline in parallel on different
    batch slices.  The host calls the NPU 8 times with different weight
    sets to achieve 32 total layers.

**Key differences from the recurrent design (design.py)**:

- Each tile holds a **different** weight matrix (not broadcast).
- No hardware loop — each tile applies matmul+ReLU exactly once.
- Tiles within a column are chained via ObjectFIFOs (pipeline).
- Weights are split per row via MemTile, not broadcast.

**SRAM budget** (H=128, B=48)::

    Weight  (one per tile): 128×128×2 = 32 KB
    Input   (one buffer):    48×128×2 = 12 KB
    Output  (one buffer):    48×128×2 = 12 KB
    Stack:                              ~1 KB
    ──────────────────────────────────────────
    Total:                             ~57 KB  (fits 64 KB)
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
NUM_ROWS = 4       # compute-tile rows (rows 2-5)
STAGES_PER_COL = NUM_ROWS


# ── Buffer type helpers ──────────────────────────────────────────────────

def _activation_type(B, H):
    """IRON buffer type for one tile's activation (B×H bf16)."""
    return np.ndarray[(B * H,), np.dtype[bfloat16]]


def _weight_type(H):
    """IRON buffer type for one tile's weight matrix (H×H bf16)."""
    return np.ndarray[(H * H,), np.dtype[bfloat16]]


def _column_weights_type(H):
    """IRON buffer type for one column's weights (4×H×H bf16)."""
    return np.ndarray[(STAGES_PER_COL * H * H,), np.dtype[bfloat16]]


def _host_activation_type(num_cols, B, H):
    """IRON buffer type for all columns' activations (num_cols × B × H)."""
    return np.ndarray[(num_cols * B * H,), np.dtype[bfloat16]]


def _host_weights_type(num_cols, H):
    """IRON buffer type for all weights (num_cols × 4 × H × H)."""
    return np.ndarray[(num_cols * STAGES_PER_COL * H * H,), np.dtype[bfloat16]]


# ── Kernel definition ────────────────────────────────────────────────────

def _define_kernel(act_ty, weight_ty):
    """Create the matmul+ReLU kernel: C = ReLU(A × B)."""
    return Kernel("matmul_relu_bf16_bf16", "mlp_kernels.a",
                  [act_ty, weight_ty, act_ty])


# ── Worker body ──────────────────────────────────────────────────────────

def _make_pipeline_stage_body():
    """Return a worker function for one pipeline stage.

    Each stage does exactly one matmul+ReLU: output = ReLU(input @ W).
    The pipeline behavior emerges from FIFO dependencies: stage i+1
    blocks on acquire until stage i releases its output.
    """
    def worker_body(of_in, of_out, of_w, mm_relu_fn):
        x = of_in.acquire(1)
        y = of_out.acquire(1)
        w = of_w.acquire(1)

        mm_relu_fn(x, w, y)   # y = ReLU(x @ W)

        of_w.release(1)
        of_in.release(1)
        of_out.release(1)

    return worker_body


# ── FIFO topology for one column ─────────────────────────────────────────

def _create_column_fifos(col, act_ty, weight_ty, H):
    """Create the FIFO chain for a 4-stage pipeline in one column.

    Returns:
        (stage_inputs, stage_outputs, stage_weights, ddr_act_in, ddr_act_out, ddr_wt)

    The activation chain is::

        DDR → ddr_act_in → [forward through MemTile] → tile(col,2)
        tile(col,2) → inter_01 → tile(col,3)
        tile(col,3) → inter_12 → tile(col,4)
        tile(col,4) → inter_23 → tile(col,5)
        tile(col,5) → ddr_act_out → DDR

    Weights are split from one 4×H×H buffer through MemTile to 4 tiles.
    """
    mem = Tile(col=col, row=1)  # MemTile

    # ── Weights: DDR → MemTile → split to 4 rows ────────────────────
    col_wt_ty = _column_weights_type(H)
    wt_ddr = ObjectFifo(col_wt_ty, name=f"wt_l3_{col}", depth=1)
    wt_splits = wt_ddr.cons().split(
        offsets=[H * H * r for r in range(STAGES_PER_COL)],
        obj_types=[weight_ty] * STAGES_PER_COL,
        names=[f"wt_r{r}_{col}" for r in range(STAGES_PER_COL)],
        depths=[1] * STAGES_PER_COL,
        placement=mem,
    )

    # ── Activation input: DDR → MemTile → tile(col, row=2) ──────────
    act_in_ddr = ObjectFifo(act_ty, name=f"in_l3_{col}", depth=1)
    act_in_fwd = act_in_ddr.cons().forward(
        name=f"in_fwd_{col}", placement=mem,
    )

    # ── Inter-tile activation FIFOs (tile-to-tile, no MemTile) ───────
    inter_01 = ObjectFifo(act_ty, name=f"inter_01_{col}", depth=1)
    inter_12 = ObjectFifo(act_ty, name=f"inter_12_{col}", depth=1)
    inter_23 = ObjectFifo(act_ty, name=f"inter_23_{col}", depth=1)

    # ── Activation output: tile(col, row=5) → MemTile → DDR ─────────
    act_out_ddr = ObjectFifo(act_ty, name=f"out_l3_{col}", depth=1)
    act_out_join = act_out_ddr.prod().join(
        offsets=[0],
        obj_types=[act_ty],
        names=[f"out_r3_{col}"],
        depths=[1],
        placement=mem,
    )

    # Assemble per-stage FIFO endpoints
    stage_inputs = [
        act_in_fwd,       # stage 0 (row 2): from MemTile
        inter_01,         # stage 1 (row 3): from row 2
        inter_12,         # stage 2 (row 4): from row 3
        inter_23,         # stage 3 (row 5): from row 4
    ]
    stage_outputs = [
        inter_01,         # stage 0 → stage 1
        inter_12,         # stage 1 → stage 2
        inter_23,         # stage 2 → stage 3
        act_out_join[0],  # stage 3 → DDR (through MemTile join)
    ]
    stage_weights = wt_splits

    return (stage_inputs, stage_outputs, stage_weights,
            act_in_ddr, act_out_ddr, wt_ddr)


# ── Tensor access patterns ───────────────────────────────────────────────

def _activation_tap(col, num_cols, B, H):
    """TAP to select column col's B×H activation slice from the host buffer.

    Host buffer layout: [col0_act, col1_act, ..., col7_act] contiguous.
    """
    total_size = num_cols * B * H
    return TensorAccessPattern(
        (1, total_size),
        col * B * H,
        [1, 1, B, H], [0, 0, H, 1],
    )


def _weights_tap(col, num_cols, H):
    """TAP to select column col's 4×H×H weight block from the host buffer.

    Host buffer layout: [col0_weights(4×H×H), col1_weights(4×H×H), ...].
    """
    total_size = num_cols * STAGES_PER_COL * H * H
    block_size = STAGES_PER_COL * H * H
    return TensorAccessPattern(
        (1, total_size),
        col * block_size,
        [1, STAGES_PER_COL, H, H], [0, H * H, H, 1],
    )


# ── Runtime sequence ─────────────────────────────────────────────────────

def _create_runtime_sequence(rt, workers, ddr_in_list, ddr_wt_list,
                             ddr_out_list, num_cols, B, H):
    """DMA transfers that feed activations + weights and drain results."""
    input_ty = _host_activation_type(num_cols, B, H)
    weight_ty = _host_weights_type(num_cols, H)
    output_ty = _host_activation_type(num_cols, B, H)

    with rt.sequence(input_ty, weight_ty, output_ty) as (inp, wts, out):
        rt.start(*workers)

        # Fill activations (one DMA per column, all in parallel)
        tg_in = rt.task_group()
        for col in range(num_cols):
            tap = _activation_tap(col, num_cols, B, H)
            rt.fill(ddr_in_list[col].prod(), inp, tap, task_group=tg_in)
        rt.finish_task_group(tg_in)

        # Fill weights (one DMA per column, all in parallel)
        tg_w = rt.task_group()
        for col in range(num_cols):
            tap = _weights_tap(col, num_cols, H)
            rt.fill(ddr_wt_list[col].prod(), wts, tap, task_group=tg_w)
        rt.finish_task_group(tg_w)

        # Drain outputs (one DMA per column, all in parallel, with wait)
        tg_out = rt.task_group()
        for col in range(num_cols):
            tap = _activation_tap(col, num_cols, B, H)
            rt.drain(ddr_out_list[col].cons(), out, tap,
                     wait=True, task_group=tg_out)
        rt.finish_task_group(tg_out)


# ── Top-level design function ────────────────────────────────────────────

def pipeline_mlp(
    H: int = 128,
    B: int = 48,
    num_cols: int = 8,
):
    """Generate MLIR for a 4-stage streaming pipeline on the XDNA 2 NPU.

    Each column has 4 compute tiles (rows 2-5), each holding a different
    weight matrix.  Data flows tile-to-tile within a column.  All columns
    run in parallel on different batch slices.

    Args:
        H: Hidden dimension (weight matrix is H×H).
        B: Batch size per column (total batch = num_cols × B).
        num_cols: Number of columns to use (1-8, default 8).

    Returns:
        MLIR module.
    """
    assert 1 <= num_cols <= NUM_COLUMNS
    assert B % 16 == 0 and H % 16 == 0

    act_ty = _activation_type(B, H)
    weight_ty = _weight_type(H)

    matmul_relu = _define_kernel(act_ty, weight_ty)
    worker_fn = _make_pipeline_stage_body()

    workers = []
    ddr_in_list = []
    ddr_out_list = []
    ddr_wt_list = []

    for col in range(num_cols):
        (stage_ins, stage_outs, stage_wts,
         ddr_in, ddr_out, ddr_wt) = _create_column_fifos(
            col, act_ty, weight_ty, H)

        ddr_in_list.append(ddr_in)
        ddr_out_list.append(ddr_out)
        ddr_wt_list.append(ddr_wt)

        for row in range(STAGES_PER_COL):
            # Determine FIFO endpoints for this stage
            in_endpoint = stage_ins[row]
            out_endpoint = stage_outs[row]
            wt_endpoint = stage_wts[row]

            # For inter-tile FIFOs, extract the correct endpoint
            # stage_ins[0] is a forwarded FIFO → use .cons()
            # stage_ins[1-3] are ObjectFifo objects → use .cons()
            if hasattr(in_endpoint, 'cons'):
                in_cons = in_endpoint.cons()
            else:
                in_cons = in_endpoint

            if hasattr(out_endpoint, 'prod'):
                out_prod = out_endpoint.prod()
            else:
                out_prod = out_endpoint

            if hasattr(wt_endpoint, 'cons'):
                wt_cons = wt_endpoint.cons()
            else:
                wt_cons = wt_endpoint

            workers.append(Worker(
                worker_fn,
                fn_args=[in_cons, out_prod, wt_cons, matmul_relu],
                placement=Tile(col=col, row=2 + row),
            ))

    # Runtime DMA sequence
    rt = Runtime()
    _create_runtime_sequence(rt, workers, ddr_in_list, ddr_wt_list,
                             ddr_out_list, num_cols, B, H)

    # Compile to MLIR
    program = Program(NPU2(), rt)
    return program.resolve_program(SequentialPlacer())


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate MLIR for the pipeline MLP design.")
    parser.add_argument("--H", type=int, default=128,
                        help="Hidden dimension (default: 128)")
    parser.add_argument("--B", type=int, default=48,
                        help="Batch size per column (default: 48)")
    parser.add_argument("--cols", type=int, default=8,
                        help="Number of columns, 1-8 (default: 8)")
    parser.add_argument("-o", "--output", default="build/pipeline_mlp.mlir",
                        help="Output MLIR path")
    args = parser.parse_args()

    module = pipeline_mlp(H=args.H, B=args.B, num_cols=args.cols)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(str(module))
    print(f"Written to {out_path}")
