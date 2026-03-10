# SPDX-License-Identifier: Apache-2.0
"""
Recurrent MLP: On-chip looping computation on the XDNA 2 NPU.

Architecture:
  - Up to 8 compute tiles (row 2), one per column = 8 independent sequences
  - Single weight matrix W loaded once, held in tile SRAM
  - Hardware loop (scf.for) applies ReLU(x @ W) repeatedly on-chip
  - Activation ping-pongs between two SRAM buffers (A→B, B→A)
  - DDR I/O only at start (input) and end (output)

This amortizes the ~120 µs XRT/DMA per-invocation overhead across many
iterations of on-chip compute, enabling 3+ TFLOPS and 10-14× speedup over CPU.

Effective depth per invocation = 2 × num_iters (two matmul+relu per loop body).

Tile SRAM budget per tile (~64 KB):
  - Weight FIFO (depth=1): H×H×2 bytes     = 32 KB for H=128
  - Input FIFO  (depth=1): B×H×2 bytes     =  4 KB for B=16, H=128
  - Output FIFO (depth=1): B×H×2 bytes     =  4 KB for B=16, H=128
  - Stack:                                   =  1 KB
  - Total:                                   = 41 KB ✓
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

from spatial_mlp.design import to_tiled, from_tiled  # reuse conversion utils


def autoreg_mlp(
    H: int = 128,
    B: int = 16,
    num_pipelines: int = 8,
    num_iters: int = 1000,
):
    """
    Generate MLIR for a recurrent MLP that loops on-chip.

    Each compute tile holds one H×H weight matrix and applies ReLU(x @ W)
    in a tight hardware loop. Activations ping-pong between two SRAM buffers.
    The effective network depth = 2 × num_iters.

    Args:
        H: Hidden dimension (weight matrix is H×H)
        B: Batch size per pipeline
        num_pipelines: Number of parallel pipelines (1-8, one per column)
        num_iters: Hardware loop count (depth = 2 × num_iters)
    """
    assert 1 <= num_pipelines <= 8, "NPU2 has 8 columns"
    assert B % 16 == 0, f"B={B} must be divisible by 16 (2*r with r=8)"
    assert H % 16 == 0, f"H={H} must be divisible by 16 (2*t with t=8)"
    assert num_iters >= 1

    dtype = bfloat16

    # Buffer types (tiled layout, 8×8 blocks)
    act_ty = np.ndarray[(B * H,), np.dtype[dtype]]
    weight_ty = np.ndarray[(H * H,), np.dtype[dtype]]

    # Host-side tensor types
    input_ty = np.ndarray[(num_pipelines * B * H,), np.dtype[dtype]]
    output_ty = np.ndarray[(num_pipelines * B * H,), np.dtype[dtype]]

    # ── Kernels ─────────────────────────────────────────────────────────
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

    # ── ObjectFIFOs ─────────────────────────────────────────────────────
    of_weights = [
        ObjectFifo(weight_ty, name=f"w_{i}", depth=1)
        for i in range(num_pipelines)
    ]
    of_inputs = [
        ObjectFifo(act_ty, name=f"in_{i}", depth=1)
        for i in range(num_pipelines)
    ]
    of_outputs = [
        ObjectFifo(act_ty, name=f"out_{i}", depth=1)
        for i in range(num_pipelines)
    ]

    # ── Worker ──────────────────────────────────────────────────────────
    #
    # All three FIFOs acquired ONCE before the loop, released after.
    # The range_() hardware loop (scf.for) contains NO FIFO ops — just
    # kernel calls. Each iteration: A→B then B→A (ping-pong).
    # After the loop, result is in A. Copy A→B for DMA drain.

    def autoreg_body(of_in, of_out, of_w, zero_fn, mm_fn, relu_fn, cp_fn):
        x = of_in.acquire(1)   # Buffer A (input, then scratch)
        y = of_out.acquire(1)  # Buffer B (output, drained to DDR)
        w = of_w.acquire(1)    # Weight (held for entire execution)

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

    # ── Create workers ──────────────────────────────────────────────────
    workers = []
    for i in range(num_pipelines):
        workers.append(
            Worker(
                autoreg_body,
                fn_args=[
                    of_inputs[i].cons(),
                    of_outputs[i].prod(),
                    of_weights[i].cons(),
                    zero_kernel,
                    matmul_kernel,
                    relu_kernel,
                    copy_kernel,
                ],
                placement=Tile(col=i, row=2),
            )
        )

    # ── Runtime sequence ────────────────────────────────────────────────
    chunk_act = B * H
    chunk_wt = H * H

    rt = Runtime()
    with rt.sequence(input_ty, weight_ty, output_ty) as (inp, wts, out):
        rt.start(*workers)

        # Fill inputs (once, all pipelines in parallel)
        tg_in = rt.task_group()
        for i in range(num_pipelines):
            input_tap = TensorAccessPattern(
                (1, num_pipelines * chunk_act),
                i * chunk_act,
                [1, 1, 1, chunk_act],
                [0, 0, 0, 1],
            )
            rt.fill(
                of_inputs[i].prod(), inp, input_tap,
                task_group=tg_in,
            )
        rt.finish_task_group(tg_in)

        # Fill weight (once, all pipelines in parallel)
        tg_w = rt.task_group()
        weight_tap = TensorAccessPattern(
            (1, chunk_wt),
            0,
            [1, 1, 1, chunk_wt],
            [0, 0, 0, 1],
        )
        for i in range(num_pipelines):
            rt.fill(
                of_weights[i].prod(), wts, weight_tap,
                task_group=tg_w,
            )
        rt.finish_task_group(tg_w)

        # Drain outputs (once, all pipelines)
        tg_out = rt.task_group()
        for i in range(num_pipelines):
            output_tap = TensorAccessPattern(
                (1, num_pipelines * chunk_act),
                i * chunk_act,
                [1, 1, 1, chunk_act],
                [0, 0, 0, 1],
            )
            rt.drain(
                of_outputs[i].cons(), out, output_tap,
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
    p.add_argument("--pipelines", type=int, default=8)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("-o", "--output", type=str, default="build/autoreg_mlp.mlir")
    args = p.parse_args()

    module = autoreg_mlp(
        H=args.H, B=args.B,
        num_pipelines=args.pipelines, num_iters=args.iters,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(str(module))
    print(f"Written to {out_path}")
