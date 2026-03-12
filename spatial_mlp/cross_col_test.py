#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Minimal test: can the IRON router connect tiles across columns?

Test 1 (direct): ObjectFIFO between Tile(0,5) and Tile(1,5) — same row, adjacent cols.
Test 2 (via MemTile): Route through MemTile if direct fails.

Usage::

    source /opt/xilinx/xrt/setup.sh
    cd ~/source/IRON
    python ~/source/npu-spatial-nets/spatial_mlp/cross_col_test.py [--run-on-hardware]
"""

import os
import sys
import argparse
from pathlib import Path

IRON_DIR = os.environ.get("IRON_DIR", str(Path.home() / "source" / "IRON"))
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, IRON_DIR)
os.chdir(IRON_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile

# Small dimensions for a quick test
B = 8
H = 32  # small H so compilation is fast
ACT_SIZE = B * H
act_ty = np.ndarray[(ACT_SIZE,), np.dtype[bfloat16]]
wt_ty = np.ndarray[(H * H + H,), np.dtype[bfloat16]]


def make_design_direct(src_col, src_row, dst_col, dst_row):
    """Two tiles in different columns, connected by a direct ObjectFIFO.

    Data flow:
        DDR → MemTile[src_col] → Tile(src_col, src_row) →
        cross_col_fifo → Tile(dst_col, dst_row) → MemTile[dst_col] → DDR
    """
    src_mem = Tile(col=src_col, row=1)
    dst_mem = Tile(col=dst_col, row=1)

    kernel = Kernel("norm_matmul_relu_bf16_bf16", "mlp_kernels.a",
                     [act_ty, wt_ty, act_ty])

    # Input: DDR → MemTile[src] → Tile(src)
    act_in_ddr = ObjectFifo(act_ty, name="act_in_ddr", depth=1)
    act_in_fwd = act_in_ddr.cons().forward(name="act_in_fwd", placement=src_mem)

    # Weight for tile 0
    wt0_ddr = ObjectFifo(wt_ty, name="wt0_ddr", depth=1)
    wt0_fwd = wt0_ddr.cons().forward(name="wt0_fwd", placement=src_mem)

    # Weight for tile 1
    wt1_ddr = ObjectFifo(wt_ty, name="wt1_ddr", depth=1)
    wt1_fwd = wt1_ddr.cons().forward(name="wt1_fwd", placement=dst_mem)

    # Cross-column FIFO (the test!) — no placement, let router decide
    cross_col = ObjectFifo(act_ty, name="cross_col", depth=1)

    # Output: Tile(dst) → MemTile[dst] → DDR
    act_out_ddr = ObjectFifo(act_ty, name="act_out_ddr", depth=1)
    act_out_join = act_out_ddr.prod().join(
        offsets=[0], obj_types=[act_ty],
        names=["act_out_j"], depths=[1], placement=dst_mem,
    )

    # Worker 0: src tile
    def worker0_fn(of_in, of_out, of_w, kern):
        x = of_in.acquire(1)
        y = of_out.acquire(1)
        w = of_w.acquire(1)
        kern(x, w, y)
        of_w.release(1)
        of_in.release(1)
        of_out.release(1)

    # Worker 1: dst tile
    def worker1_fn(of_in, of_out, of_w, kern):
        x = of_in.acquire(1)
        y = of_out.acquire(1)
        w = of_w.acquire(1)
        kern(x, w, y)
        of_w.release(1)
        of_in.release(1)
        of_out.release(1)

    w0 = Worker(worker0_fn,
                fn_args=[act_in_fwd.cons(), cross_col.prod(), wt0_fwd.cons(), kernel],
                placement=Tile(col=src_col, row=src_row))
    w1 = Worker(worker1_fn,
                fn_args=[cross_col.cons(), act_out_join[0].prod(), wt1_fwd.cons(), kernel],
                placement=Tile(col=dst_col, row=dst_row))

    # Runtime sequence
    host_act_ty = np.ndarray[(ACT_SIZE,), np.dtype[bfloat16]]
    host_wt_ty = np.ndarray[(2 * (H * H + H),), np.dtype[bfloat16]]
    host_out_ty = np.ndarray[(ACT_SIZE,), np.dtype[bfloat16]]

    rt = Runtime()
    with rt.sequence(host_act_ty, host_wt_ty, host_out_ty) as (inp, wts, out):
        rt.start(w0, w1)

        from aie.helpers.taplib.tap import TensorAccessPattern
        wt_size = H * H + H

        # Fill input activations
        rt.fill(act_in_ddr.prod(), inp)

        # Fill weights (w0 from first half, w1 from second half)
        tap0 = TensorAccessPattern((1, 2 * wt_size), 0,
                                    [1, 1, 1, wt_size], [0, 0, 0, 1])
        tap1 = TensorAccessPattern((1, 2 * wt_size), wt_size,
                                    [1, 1, 1, wt_size], [0, 0, 0, 1])
        rt.fill(wt0_ddr.prod(), wts, tap0)
        rt.fill(wt1_ddr.prod(), wts, tap1)

        # Drain output
        rt.drain(act_out_ddr.cons(), out, wait=True)

    program = Program(NPU2(), rt)
    return program.resolve_program(SequentialPlacer())


def try_compile(label, src_col, src_row, dst_col, dst_row):
    """Attempt to compile a cross-column design and report success/failure."""
    print(f"\n{'='*60}")
    print(f"Test: {label}")
    print(f"  Tile({src_col},{src_row}) → Tile({dst_col},{dst_row})")
    print(f"{'='*60}")
    try:
        module = make_design_direct(src_col, src_row, dst_col, dst_row)
        print(f"  ✓ COMPILATION SUCCEEDED — router found a path!")
        return module
    except Exception as e:
        print(f"  ✗ COMPILATION FAILED: {e}")
        return None


def run_on_hardware(module, H=H, B=B):
    """Actually run the compiled design on NPU hardware."""
    from iron.common.aie_context import AIEContext
    from spatial_mlp import to_tiled, from_tiled

    # Save MLIR to file and compile
    mlir_path = Path("build/cross_col_test.mlir")
    mlir_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mlir_path, "w") as f:
        f.write(str(module))
    print(f"\nMLIR written to {mlir_path}")

    # Generate test data
    rng = np.random.default_rng(42)
    W0 = (rng.standard_normal((H, H)) * 0.5 / np.sqrt(H)).astype(bfloat16)
    W1 = (rng.standard_normal((H, H)) * 0.5 / np.sqrt(H)).astype(bfloat16)
    scale = np.ones(H, dtype=bfloat16)
    X = rng.standard_normal((B, H)).astype(bfloat16)

    # CPU reference
    x = X.astype(np.float32)
    s = scale.astype(np.float32)
    # Stage 0: RMSNorm + matmul + ReLU
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
    x = x / rms * s
    x = np.maximum(x @ W0.astype(np.float32), 0)
    # Stage 1: RMSNorm + matmul + ReLU
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
    x = x / rms * s
    x = np.maximum(x @ W1.astype(np.float32), 0)
    Y_ref = x.astype(bfloat16)

    print(f"CPU reference computed. Output range: [{Y_ref.min():.3f}, {Y_ref.max():.3f}]")
    print(f"  Sample output: {Y_ref[0, :5]}")

    # TODO: compile and run via AIEContext once we've verified routing works
    print("\n[Hardware execution not yet integrated — routing validation is the goal]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test cross-column IRON routing")
    parser.add_argument("--run-on-hardware", action="store_true")
    args = parser.parse_args()

    # Test 1: Same row, adjacent columns (the snake turn)
    m1 = try_compile("Adjacent columns, same row (row 5)",
                     src_col=0, src_row=5, dst_col=1, dst_row=5)

    # Test 2: Same row, adjacent columns (row 2 — the other turn direction)
    m2 = try_compile("Adjacent columns, same row (row 2)",
                     src_col=1, src_row=2, dst_col=2, dst_row=2)

    # Test 3: Same row but non-adjacent columns
    m3 = try_compile("Non-adjacent columns, same row",
                     src_col=0, src_row=2, dst_col=3, dst_row=2)

    # Test 4: Within same column (known to work, as control)
    m4 = try_compile("Same column, different rows (control)",
                     src_col=0, src_row=2, dst_col=0, dst_row=3)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, result in [
        ("Cross-col row 5 (0→1)", m1),
        ("Cross-col row 2 (1→2)", m2),
        ("Cross-col non-adj (0→3)", m3),
        ("Same-col control (0,2→0,3)", m4),
    ]:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {label}")

    if args.run_on_hardware and m1:
        run_on_hardware(m1)
