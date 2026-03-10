#!/usr/bin/env python3
"""
Benchmark: Recurrent MLP on AMD XDNA 2 NPU vs CPU.

Architecture: Single weight matrix W, applied 2×num_iters times via
hardware loop. Activation ping-pongs between two SRAM buffers on-chip.
DDR I/O only at start and end → amortizes ~120 µs invocation overhead.
"""

import os
import sys
import time
import argparse

IRON_DIR = "/home/hpenedones/source/IRON"
sys.path.insert(0, IRON_DIR)
os.chdir(IRON_DIR)

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, "/home/hpenedones/source/npu-spatial-nets")
from spatial_mlp.op_autoreg import AIEAutoregMLP
from spatial_mlp.design import to_tiled, from_tiled


def autoreg_reference(input_data, weight, depth):
    """NumPy reference: apply ReLU(x @ W) depth times."""
    x = input_data.astype(np.float32)
    W = weight.astype(np.float32)
    for _ in range(depth):
        x = np.maximum(x @ W, 0)
    return x.astype(bfloat16)


def benchmark(H=128, B=16, num_pipelines=8, num_iters=1000,
              warmup=3, timed_iters=10):
    from iron.common.aie_context import AIEContext

    total_samples = num_pipelines * B
    depth = 2 * num_iters  # 2 matmul+relu per loop iteration (ping-pong)
    flops_per_layer = total_samples * 2 * H * H
    total_flops = depth * flops_per_layer

    print(f"\n{'='*70}")
    print(f"Autoregressive MLP Benchmark")
    print(f"  Network:    Linear({H}→{H}) + ReLU, applied {depth}× "
          f"(hw loop, {num_iters} iters × 2)")
    print(f"  Tiles:      {num_pipelines} compute tiles (1 per column)")
    print(f"  Batch:      {B} samples/tile × {num_pipelines} tiles = "
          f"{total_samples} samples")
    print(f"  Params:     {H * H:,} ({H * H * 2 / 1024:.0f} KB)")
    print(f"  FLOPs:      {total_flops:,} per invocation")
    print(f"{'='*70}")

    # ── Generate test data ──────────────────────────────────────────────
    rng = np.random.default_rng(42)
    # Weight scaled to prevent vanishing/exploding activations with many iters
    W = (rng.standard_normal((H, H)) / np.sqrt(H)).astype(bfloat16)
    X = rng.standard_normal((total_samples, H)).astype(bfloat16)

    # CPU reference
    print("\nComputing CPU reference...")
    Y_ref = autoreg_reference(X, W, depth)

    # Convert to tiled layout
    X_tiled = np.concatenate([to_tiled(X[i*B:(i+1)*B])
                              for i in range(num_pipelines)])
    W_tiled = to_tiled(W)

    # ── NPU ─────────────────────────────────────────────────────────────
    ctx = AIEContext()
    op = AIEAutoregMLP(
        H=H, B=B,
        num_pipelines=num_pipelines, num_iters=num_iters,
        context=ctx,
    )
    print("Compiling for NPU...")
    ctx.compile_all()
    ctx.prepare_runtime()

    op.write_buffer("input", X_tiled)
    op.write_buffer("weights", W_tiled)
    op.write_buffer("output", np.zeros(total_samples * H, dtype=bfloat16))

    print(f"Warmup ({warmup} iters)...")
    for _ in range(warmup):
        op.write_buffer("input", X_tiled)  # reset input each time
        op.run_runlist()

    print(f"Timed ({timed_iters} iters)...")
    npu_times = []
    for _ in range(timed_iters):
        op.write_buffer("input", X_tiled)
        t = op.run_runlist()
        npu_times.append(t)

    npu_avg = np.mean(npu_times)
    npu_min = np.min(npu_times)
    npu_std = np.std(npu_times)

    # Read output for correctness check
    Y_npu_flat = op.read_buffer("output", (total_samples * H,), copy=True)
    Y_npu = np.concatenate([from_tiled(Y_npu_flat[i*B*H:(i+1)*B*H], B, H)
                            for i in range(num_pipelines)])

    # Correctness (relaxed for deep bf16 computation)
    ref_f32 = Y_ref.astype(np.float32)
    npu_f32 = Y_npu.astype(np.float32)
    # Use relative tolerance on non-zero elements
    nonzero = np.abs(ref_f32) > 1e-6
    if nonzero.sum() > 0:
        rel_err = np.abs(ref_f32[nonzero] - npu_f32[nonzero]) / np.abs(ref_f32[nonzero])
        median_rel_err = np.median(rel_err)
    else:
        median_rel_err = float('nan')
    close = np.isclose(ref_f32, npu_f32, rtol=0.3, atol=0.01)

    npu_gflops = total_flops / npu_avg / 1e9
    npu_sps = total_samples / npu_avg
    npu_ips = num_iters / npu_avg  # iterations per second per pipeline

    print(f"\n--- NPU Results ---")
    print(f"  Correctness:     {close.sum()}/{close.size} "
          f"({100*close.mean():.1f}%)")
    print(f"  Median rel err:  {median_rel_err:.4f}")
    print(f"  Latency:         {npu_avg*1e3:.2f} ± {npu_std*1e3:.2f} ms "
          f"(min {npu_min*1e3:.2f} ms)")
    print(f"  Throughput:      {npu_gflops:.1f} GFLOPS")
    print(f"  Samples/sec:     {npu_sps:,.0f}")
    print(f"  Depth/sec:       {depth / npu_avg:,.0f} "
          f"({depth} steps in {npu_avg*1e3:.2f} ms)")

    # ── CPU (PyTorch bf16) ──────────────────────────────────────────────
    import torch
    x_cpu_orig = torch.from_numpy(X.astype(np.float32)).to(torch.bfloat16)
    w_cpu = torch.from_numpy(W.astype(np.float32)).to(torch.bfloat16)

    # Warmup
    for _ in range(5):
        x = x_cpu_orig.clone()
        for _it in range(depth):
            x = torch.relu(x @ w_cpu)

    # Timed
    cpu_runs = max(3, 100 // max(depth, 1))
    cpu_start = time.perf_counter()
    for _ in range(cpu_runs):
        x = x_cpu_orig.clone()
        for _it in range(depth):
            x = torch.relu(x @ w_cpu)
    cpu_elapsed = (time.perf_counter() - cpu_start) / cpu_runs

    cpu_gflops = total_flops / cpu_elapsed / 1e9
    cpu_sps = total_samples / cpu_elapsed
    cpu_depth_per_sec = depth / cpu_elapsed

    print(f"\n--- CPU Results (PyTorch bf16, {os.cpu_count()} cores) ---")
    print(f"  Latency:         {cpu_elapsed*1e3:.2f} ms")
    print(f"  Throughput:      {cpu_gflops:.1f} GFLOPS")
    print(f"  Samples/sec:     {cpu_sps:,.0f}")
    print(f"  Depth/sec:       {cpu_depth_per_sec:,.0f}")

    # ── Summary ─────────────────────────────────────────────────────────
    speedup = npu_sps / cpu_sps
    print(f"\n{'='*70}")
    print(f"  NPU:     {npu_avg*1e3:>8.2f} ms  "
          f"{npu_gflops:>7.1f} GFLOPS  {npu_sps:>12,.0f} samp/s")
    print(f"  CPU:     {cpu_elapsed*1e3:>8.2f} ms  "
          f"{cpu_gflops:>7.1f} GFLOPS  {cpu_sps:>12,.0f} samp/s")
    print(f"  Speedup: {speedup:.1f}×")
    print(f"{'='*70}")

    # ── Analysis ────────────────────────────────────────────────────────
    theoretical_compute_us = total_flops / 25e12 * 1e6
    overhead_us = max(0, npu_avg * 1e6 - theoretical_compute_us)
    compute_pct = theoretical_compute_us / (npu_avg * 1e6) * 100
    print(f"\n  Analysis:")
    print(f"    Theoretical compute: {theoretical_compute_us:.1f} µs (at 25 TFLOPS peak)")
    print(f"    Measured latency:    {npu_avg*1e6:.1f} µs")
    print(f"    Compute fraction:    {compute_pct:.1f}%")
    if speedup < 1:
        print(f"    CPU still wins — try increasing num_iters or H")
    else:
        print(f"    NPU wins by {speedup:.1f}× — overhead successfully amortized!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--pipelines", type=int, default=8)
    p.add_argument("--iters", type=int, default=1000,
                   help="Hardware loop count (depth = 2 × iters)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timed", type=int, default=10)
    args = p.parse_args()

    benchmark(
        H=args.H, B=args.B,
        num_pipelines=args.pipelines, num_iters=args.iters,
        warmup=args.warmup, timed_iters=args.timed,
    )
