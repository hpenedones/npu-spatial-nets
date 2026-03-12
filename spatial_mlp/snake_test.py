#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test and benchmark the snake pipeline MLP on the NPU.

32 tiles in a serpentine path, each with a unique weight matrix.
Activations flow through all tiles in a single NPU call:
  h₃₂ = (ReLU∘RMSNorm∘W₃₂) ∘ ... ∘ (ReLU∘RMSNorm∘W₁)(h₀)

Usage::

    source /opt/xilinx/xrt/setup.sh
    python -m spatial_mlp.snake_test [--cols 1] [--H 32] [--B 8]
"""

import os
import sys
import time
import argparse
from pathlib import Path

IRON_DIR = os.environ.get("IRON_DIR", str(Path.home() / "source" / "IRON"))
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)


def _setup_environment():
    sys.path.insert(0, IRON_DIR)
    os.chdir(IRON_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)


_setup_environment()

import numpy as np
from ml_dtypes import bfloat16
from spatial_mlp import to_tiled, from_tiled
from spatial_mlp.snake_op import AIESnakeMLP, ROWS_PER_COL


def reference_snake(X, weights, scale, num_tiles):
    """NumPy reference: apply num_tiles stages of ReLU(RMSNorm(x, scale) @ W_i).

    Uses bf16 intermediate values to match NPU numerical behavior.
    """
    x = X.copy()  # bf16
    scale_bf16 = scale.copy()
    for i in range(num_tiles):
        # RMSNorm in float32 (matches kernel which uses float32 for reduction)
        x_f32 = x.astype(np.float32)
        rms = np.sqrt(np.mean(x_f32 ** 2, axis=-1, keepdims=True) + 1e-6)
        x = (x_f32 / rms * scale_bf16.astype(np.float32)).astype(bfloat16)
        # Matmul + ReLU in bf16 (matches NPU)
        W = weights[i]
        x = np.maximum(x.astype(np.float32) @ W.astype(np.float32), 0).astype(bfloat16)
    return x


def generate_test_data(H, B, num_tiles, seed=42):
    """Create random weights, scale, and input for benchmarking."""
    rng = np.random.default_rng(seed)

    # Scale weights to keep activations stable through many layers
    # Use smaller initialization for deeper pipelines
    fan_factor = 0.5 / np.sqrt(H)
    weights = []
    for _ in range(num_tiles):
        W = (rng.standard_normal((H, H)) * fan_factor).astype(bfloat16)
        weights.append(W)

    scale = np.ones(H, dtype=bfloat16)
    X = rng.standard_normal((B, H)).astype(bfloat16)
    return weights, scale, X


def tile_weights_for_npu(weights, scale, H, num_tiles):
    """Tile and concatenate all weights+scale in host buffer layout.

    Layout: [tile0: W_tiled(H×H) + scale(H), tile1: ..., ...]
    """
    parts = []
    for i in range(num_tiles):
        W = weights[i]
        parts.append(np.concatenate([to_tiled(W), scale]))
    return np.concatenate(parts)


def benchmark(H=160, B=8, num_cols=8, warmup=3, timed_iters=10):
    """Run the snake pipeline MLP on NPU and compare to CPU reference."""
    from iron.common.aie_context import AIEContext

    num_tiles = num_cols * ROWS_PER_COL
    flops_per_tile = B * 2 * H * H
    total_flops = num_tiles * flops_per_tile

    print(f"\n{'='*70}")
    print(f"Snake Pipeline MLP Benchmark")
    print(f"  Tiles:       {num_tiles} ({num_cols} cols × {ROWS_PER_COL} rows)")
    print(f"  Layers:      {num_tiles} (one per tile, unique weights)")
    print(f"  Batch:       {B}")
    print(f"  Hidden:      {H}")
    print(f"  Weights:     {num_tiles} × {H}×{H} = "
          f"{num_tiles * H * H * 2 / 1024:.0f} KB total")
    print(f"  FLOPs:       {total_flops/1e6:.2f} MFLOP per invocation")
    print(f"{'='*70}")

    # Generate test data
    weights, scale, X = generate_test_data(H, B, num_tiles)

    # CPU reference
    print("\nComputing CPU reference...")
    Y_ref = reference_snake(X, weights, scale, num_tiles)
    print(f"  Output range: [{float(Y_ref.min()):.4f}, {float(Y_ref.max()):.4f}]")
    print(f"  Sample output: {Y_ref[0, :5]}")

    # Tile data for NPU
    X_tiled = to_tiled(X)
    W_tiled = tile_weights_for_npu(weights, scale, H, num_tiles)
    zero_output = np.zeros(B * H, dtype=bfloat16)

    # Compile and run on NPU
    ctx = AIEContext()
    op = AIESnakeMLP(H=H, B=B, num_cols=num_cols, context=ctx)
    print("Compiling for NPU (this may take a few minutes)...")
    ctx.compile_all()
    ctx.prepare_runtime()
    print("Compilation done.")

    # Warmup
    print(f"Warmup ({warmup} runs)...")
    for _ in range(warmup):
        op.write_buffer("input", X_tiled)
        op.write_buffer("weights", W_tiled)
        op.write_buffer("output", zero_output.copy())
        op.run_runlist()

    # Timed runs
    print(f"Timed ({timed_iters} runs)...")
    npu_times = []
    for _ in range(timed_iters):
        op.write_buffer("input", X_tiled)
        op.write_buffer("weights", W_tiled)
        op.write_buffer("output", zero_output.copy())
        elapsed = op.run_runlist()
        npu_times.append(elapsed)

    Y_flat = op.read_buffer("output", (B * H,), copy=True)
    Y_npu = from_tiled(Y_flat, B, H)

    # Correctness
    ref_f32 = Y_ref.astype(np.float32)
    npu_f32 = Y_npu.astype(np.float32)
    nonzero = np.abs(ref_f32) > 1e-6
    if nonzero.sum() > 0:
        median_rel_err = float(np.median(
            np.abs(ref_f32[nonzero] - npu_f32[nonzero])
            / np.abs(ref_f32[nonzero])))
    else:
        median_rel_err = float('nan')

    close = np.isclose(ref_f32, npu_f32, rtol=0.3, atol=0.01)
    pct_close = 100 * close.mean()

    # CPU benchmark
    import torch
    x_t = torch.from_numpy(X.astype(np.float32)).to(torch.bfloat16)
    ws_t = [torch.from_numpy(w.astype(np.float32)).to(torch.bfloat16)
            for w in weights]
    scale_t = torch.from_numpy(scale.astype(np.float32)).to(torch.bfloat16)

    # Warmup CPU
    for _ in range(5):
        h = x_t.clone()
        for w in ws_t:
            rms = torch.sqrt(torch.mean(h.float() ** 2, dim=-1, keepdim=True) + 1e-6)
            h = (h.float() / rms * scale_t.float()).to(torch.bfloat16)
            h = torch.relu(h @ w)

    cpu_runs = max(10, 100)
    t0 = time.perf_counter()
    for _ in range(cpu_runs):
        h = x_t.clone()
        for w in ws_t:
            rms = torch.sqrt(torch.mean(h.float() ** 2, dim=-1, keepdim=True) + 1e-6)
            h = (h.float() / rms * scale_t.float()).to(torch.bfloat16)
            h = torch.relu(h @ w)
    cpu_latency = (time.perf_counter() - t0) / cpu_runs

    npu_avg = np.mean(npu_times)
    npu_std = np.std(npu_times)

    npu_gflops = total_flops / npu_avg / 1e9
    cpu_gflops = total_flops / cpu_latency / 1e9

    print(f"\n--- Results ---")
    print(f"  Correctness:     {close.sum()}/{close.size} ({pct_close:.1f}%)")
    print(f"  Median rel err:  {median_rel_err:.4f}")
    print(f"  NPU latency:     {npu_avg*1e3:.3f} ± {npu_std*1e3:.3f} ms")
    print(f"  NPU throughput:  {npu_gflops:.3f} GFLOPS")
    print(f"  CPU latency:     {cpu_latency*1e3:.3f} ms")
    print(f"  CPU throughput:  {cpu_gflops:.3f} GFLOPS")
    print(f"  Speedup:         {cpu_latency / npu_avg:.1f}×")
    print(f"  Per-seq chars/s: {B / npu_avg:,.0f} (8 sequences × {1/npu_avg:,.0f} steps/s)")
    print(f"{'='*70}")

    # Print sample values for debugging
    print(f"\nSample values (first 5 elements of row 0):")
    print(f"  Reference: {Y_ref[0, :5]}")
    print(f"  NPU:       {Y_npu[0, :5]}")

    return pct_close > 50


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test snake pipeline MLP on NPU.")
    parser.add_argument("--H", type=int, default=160,
                        help="Hidden dimension (default: 160)")
    parser.add_argument("--B", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--cols", type=int, default=8,
                        help="Number of columns (default: 8)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed", type=int, default=10)
    args = parser.parse_args()

    success = benchmark(
        H=args.H, B=args.B, num_cols=args.cols,
        warmup=args.warmup, timed_iters=args.timed,
    )
    sys.exit(0 if success else 1)
