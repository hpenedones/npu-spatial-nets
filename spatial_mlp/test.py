#!/usr/bin/env python3
"""
Benchmark: Recurrent MLP on AMD XDNA 2 NPU vs CPU.

Single weight matrix W held in tile SRAM, applied 2*num_iters times
via hardware loop. Scales across 1-32 compute tiles.
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
from spatial_mlp.op import AIERecurrentMLP
from spatial_mlp import to_tiled, from_tiled


def reference(input_data, weight, depth):
    """NumPy reference: apply ReLU(x @ W) depth times."""
    x = input_data.astype(np.float32)
    W = weight.astype(np.float32)
    for _ in range(depth):
        x = np.maximum(x @ W, 0)
    return x.astype(bfloat16)


def benchmark(H=128, B=16, num_tiles=32, num_iters=1000,
              warmup=3, timed_iters=10):
    from iron.common.aie_context import AIEContext

    total_samples = num_tiles * B
    depth = 2 * num_iters
    flops_per_step = total_samples * 2 * H * H
    total_flops = depth * flops_per_step

    print(f"\n{'='*70}")
    print(f"Recurrent MLP Benchmark")
    print(f"  Network:    ReLU(x @ W) applied {depth}x "
          f"(hw loop, {num_iters} iters x 2)")
    print(f"  Tiles:      {num_tiles} compute tiles "
          f"({min(num_tiles, 8)} cols x {(num_tiles + 7) // 8} rows)")
    print(f"  Batch:      {B}/tile x {num_tiles} tiles = "
          f"{total_samples} samples")
    print(f"  Weight:     {H}x{H} = {H*H:,} params ({H*H*2/1024:.0f} KB)")
    print(f"  FLOPs:      {total_flops/1e9:.2f} GFLOP per invocation")
    print(f"{'='*70}")

    # -- Generate test data -----------------------------------------------
    rng = np.random.default_rng(42)
    W = (rng.standard_normal((H, H)) / np.sqrt(H)).astype(bfloat16)
    X = rng.standard_normal((total_samples, H)).astype(bfloat16)

    print("\nComputing CPU reference...")
    Y_ref = reference(X, W, depth)

    X_tiled = np.concatenate([to_tiled(X[i*B:(i+1)*B])
                              for i in range(num_tiles)])
    W_tiled = to_tiled(W)

    # -- NPU --------------------------------------------------------------
    ctx = AIEContext()
    op = AIERecurrentMLP(
        H=H, B=B, num_tiles=num_tiles, num_iters=num_iters,
        context=ctx,
    )
    print("Compiling for NPU...")
    ctx.compile_all()
    ctx.prepare_runtime()

    op.write_buffer("input", X_tiled)
    op.write_buffer("weights", W_tiled)
    op.write_buffer("output", np.zeros(total_samples * H, dtype=bfloat16))

    print(f"Warmup ({warmup} runs)...")
    for _ in range(warmup):
        op.write_buffer("input", X_tiled)
        op.run_runlist()

    print(f"Timed ({timed_iters} runs)...")
    npu_times = []
    for _ in range(timed_iters):
        op.write_buffer("input", X_tiled)
        t = op.run_runlist()
        npu_times.append(t)

    npu_avg = np.mean(npu_times)
    npu_min = np.min(npu_times)
    npu_std = np.std(npu_times)

    Y_npu_flat = op.read_buffer("output", (total_samples * H,), copy=True)
    Y_npu = np.concatenate([from_tiled(Y_npu_flat[i*B*H:(i+1)*B*H], B, H)
                            for i in range(num_tiles)])

    ref_f32 = Y_ref.astype(np.float32)
    npu_f32 = Y_npu.astype(np.float32)
    nonzero = np.abs(ref_f32) > 1e-6
    if nonzero.sum() > 0:
        median_rel_err = np.median(
            np.abs(ref_f32[nonzero] - npu_f32[nonzero]) / np.abs(ref_f32[nonzero]))
    else:
        median_rel_err = float('nan')
    close = np.isclose(ref_f32, npu_f32, rtol=0.3, atol=0.01)

    npu_gflops = total_flops / npu_avg / 1e9
    npu_tflops = npu_gflops / 1000
    npu_sps = total_samples / npu_avg
    gflops_per_tile = npu_gflops / num_tiles
    pct_peak = npu_tflops / 25.0 * 100

    print(f"\n--- NPU Results ---")
    print(f"  Correctness:     {close.sum()}/{close.size} "
          f"({100*close.mean():.1f}%)")
    print(f"  Latency:         {npu_avg*1e3:.2f} +/- {npu_std*1e3:.2f} ms "
          f"(min {npu_min*1e3:.2f} ms)")
    print(f"  Throughput:      {npu_gflops:.0f} GFLOPS "
          f"({npu_tflops:.2f} TFLOPS)")
    print(f"  Per-tile:        {gflops_per_tile:.0f} GFLOPS/tile")
    print(f"  Peak util:       {pct_peak:.1f}% of 25 TFLOPS")
    print(f"  Samples/sec:     {npu_sps:,.0f}")

    # -- CPU (PyTorch bf16) -----------------------------------------------
    import torch
    x_cpu_orig = torch.from_numpy(X.astype(np.float32)).to(torch.bfloat16)
    w_cpu = torch.from_numpy(W.astype(np.float32)).to(torch.bfloat16)

    for _ in range(5):
        x = x_cpu_orig.clone()
        for _ in range(depth):
            x = torch.relu(x @ w_cpu)

    cpu_runs = max(3, 100 // max(depth, 1))
    cpu_start = time.perf_counter()
    for _ in range(cpu_runs):
        x = x_cpu_orig.clone()
        for _ in range(depth):
            x = torch.relu(x @ w_cpu)
    cpu_elapsed = (time.perf_counter() - cpu_start) / cpu_runs

    cpu_gflops = total_flops / cpu_elapsed / 1e9
    cpu_sps = total_samples / cpu_elapsed

    print(f"\n--- CPU Results (PyTorch bf16, {os.cpu_count()} cores) ---")
    print(f"  Latency:         {cpu_elapsed*1e3:.2f} ms")
    print(f"  Throughput:      {cpu_gflops:.0f} GFLOPS")

    # -- Summary ----------------------------------------------------------
    speedup = cpu_elapsed / npu_avg
    print(f"\n{'='*70}")
    print(f"  NPU:  {npu_avg*1e3:>8.2f} ms   {npu_tflops:>5.2f} TFLOPS  "
          f"({pct_peak:.1f}% peak)   {npu_sps:>10,.0f} samp/s")
    print(f"  CPU:  {cpu_elapsed*1e3:>8.2f} ms   {cpu_gflops/1000:>5.2f} TFLOPS  "
          f"              {cpu_sps:>10,.0f} samp/s")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"{'='*70}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--tiles", type=int, default=24,
                   help="Number of compute tiles (1-24)")
    p.add_argument("--iters", type=int, default=1000,
                   help="Hardware loop count (depth = 2 * iters)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timed", type=int, default=10)
    args = p.parse_args()

    benchmark(
        H=args.H, B=args.B,
        num_tiles=args.tiles, num_iters=args.iters,
        warmup=args.warmup, timed_iters=args.timed,
    )
