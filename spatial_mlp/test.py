#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark: Recurrent MLP on AMD XDNA 2 NPU vs CPU.

Compiles the recurrent MLP for a given tile/iteration configuration,
runs it on both NPU and CPU (PyTorch bf16), and reports latency,
throughput, correctness, and speedup.

Usage::

    # From the project root, with the .venv activated and XRT sourced:
    source /opt/xilinx/xrt/setup.sh
    python -m spatial_mlp.test --tiles 24 --iters 1000

Environment variables:
    IRON_DIR  Path to the IRON installation (default: ~/source/IRON)
"""

import os
import sys
import time
import argparse
from pathlib import Path

# ── Environment setup ────────────────────────────────────────────────────
# IRON must be importable and the working directory must be IRON's root
# so that the compilation system can find kernel sources and build tools.

IRON_DIR = os.environ.get("IRON_DIR",
                          str(Path.home() / "source" / "IRON"))
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)


def _setup_environment():
    """Add IRON and this project to the Python path; chdir to IRON root."""
    sys.path.insert(0, IRON_DIR)
    os.chdir(IRON_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)


_setup_environment()

import numpy as np
from ml_dtypes import bfloat16

from spatial_mlp.op import AIERecurrentMLP
from spatial_mlp import to_tiled, from_tiled

# ── Hardware constants ───────────────────────────────────────────────────

NPU_PEAK_TFLOPS = 25.0  # XDNA 2 theoretical bf16 peak


# ── Reference implementation ─────────────────────────────────────────────

def reference(input_data, weight, depth):
    """NumPy reference: apply ReLU(x @ W) ``depth`` times in float32.

    Uses float32 internally for numerical stability, then casts back
    to bfloat16 for comparison with NPU output.
    """
    x = input_data.astype(np.float32)
    W = weight.astype(np.float32)
    for _ in range(depth):
        x = np.maximum(x @ W, 0)
    return x.astype(bfloat16)


# ── Test data generation ─────────────────────────────────────────────────

def generate_test_data(H, B, num_tiles, seed=42):
    """Create random weight and input matrices for benchmarking.

    Weight is scaled by 1/√H to keep activations in a stable range
    across deep recurrent application.

    Returns:
        (W, X) — weight (H, H) and input (num_tiles*B, H), both bfloat16.
    """
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((H, H)) / np.sqrt(H)).astype(bfloat16)
    X = rng.standard_normal((num_tiles * B, H)).astype(bfloat16)
    return W, X


# ── Tiled buffer conversion ──────────────────────────────────────────────

def tile_activations(X, B, num_tiles):
    """Convert row-major activation matrix to tiled layout for all tiles."""
    return np.concatenate([to_tiled(X[i * B:(i + 1) * B])
                           for i in range(num_tiles)])


def untile_activations(flat, B, H, num_tiles):
    """Convert tiled layout back to row-major (num_tiles*B, H) matrix."""
    return np.concatenate([from_tiled(flat[i * B * H:(i + 1) * B * H], B, H)
                           for i in range(num_tiles)])


# ── Correctness checking ─────────────────────────────────────────────────

def check_correctness(ref, result, rtol=0.3, atol=0.01):
    """Compare NPU output against reference, returning accuracy metrics.

    Uses relaxed tolerances because bfloat16 matmul accumulation across
    thousands of iterations amplifies rounding errors.

    Returns:
        (num_close, total, pct_close, median_relative_error)
    """
    ref_f32 = ref.astype(np.float32)
    npu_f32 = result.astype(np.float32)

    nonzero = np.abs(ref_f32) > 1e-6
    if nonzero.sum() > 0:
        median_rel_err = float(np.median(
            np.abs(ref_f32[nonzero] - npu_f32[nonzero])
            / np.abs(ref_f32[nonzero])))
    else:
        median_rel_err = float('nan')

    close = np.isclose(ref_f32, npu_f32, rtol=rtol, atol=atol)
    return close.sum(), close.size, 100 * close.mean(), median_rel_err


# ── Performance metrics ──────────────────────────────────────────────────

def compute_metrics(total_flops, latency_seconds, num_tiles, total_samples):
    """Compute throughput metrics from raw timing.

    Returns:
        dict with keys: gflops, tflops, samples_per_sec, gflops_per_tile,
                        pct_peak
    """
    gflops = total_flops / latency_seconds / 1e9
    tflops = gflops / 1000
    return {
        "gflops": gflops,
        "tflops": tflops,
        "samples_per_sec": total_samples / latency_seconds,
        "gflops_per_tile": gflops / num_tiles,
        "pct_peak": tflops / NPU_PEAK_TFLOPS * 100,
    }


# ── NPU execution ────────────────────────────────────────────────────────

def run_npu(op, X_tiled, W_tiled, total_samples, H, warmup, timed_iters):
    """Execute the recurrent MLP on the NPU and return timing + output.

    Returns:
        (npu_times, Y_npu_flat) — list of per-run durations and raw output.
    """
    zero_output = np.zeros(total_samples * H, dtype=bfloat16)

    op.write_buffer("input", X_tiled)
    op.write_buffer("weights", W_tiled)
    op.write_buffer("output", zero_output)

    print(f"Warmup ({warmup} runs)...")
    for _ in range(warmup):
        op.write_buffer("input", X_tiled)
        op.run_runlist()

    print(f"Timed ({timed_iters} runs)...")
    npu_times = []
    for _ in range(timed_iters):
        op.write_buffer("input", X_tiled)
        elapsed = op.run_runlist()
        npu_times.append(elapsed)

    Y_flat = op.read_buffer("output", (total_samples * H,), copy=True)
    return npu_times, Y_flat


# ── CPU execution ─────────────────────────────────────────────────────────

def run_cpu(X, W, depth, warmup_runs=5, min_timed_runs=3):
    """Run the recurrent MLP on CPU using PyTorch bf16 and return timing.

    Returns:
        Average latency in seconds.
    """
    import torch

    x_orig = torch.from_numpy(X.astype(np.float32)).to(torch.bfloat16)
    w = torch.from_numpy(W.astype(np.float32)).to(torch.bfloat16)

    for _ in range(warmup_runs):
        x = x_orig.clone()
        for _ in range(depth):
            x = torch.relu(x @ w)

    timed_runs = max(min_timed_runs, 100 // max(depth, 1))
    start = time.perf_counter()
    for _ in range(timed_runs):
        x = x_orig.clone()
        for _ in range(depth):
            x = torch.relu(x @ w)
    return (time.perf_counter() - start) / timed_runs


# ── Results display ───────────────────────────────────────────────────────

def print_results(npu_metrics, cpu_metrics, correctness, npu_times,
                  cpu_latency, depth):
    """Print a formatted benchmark summary."""
    npu_avg = np.mean(npu_times)
    npu_std = np.std(npu_times)
    npu_min = np.min(npu_times)
    num_close, total, pct_close, median_rel_err = correctness

    print(f"\n--- NPU Results ---")
    print(f"  Correctness:     {num_close}/{total} ({pct_close:.1f}%)")
    print(f"  Latency:         {npu_avg*1e3:.2f} ± {npu_std*1e3:.2f} ms "
          f"(min {npu_min*1e3:.2f} ms)")
    print(f"  Throughput:      {npu_metrics['gflops']:.0f} GFLOPS "
          f"({npu_metrics['tflops']:.2f} TFLOPS)")
    print(f"  Per-tile:        {npu_metrics['gflops_per_tile']:.0f} GFLOPS/tile")
    print(f"  Peak util:       {npu_metrics['pct_peak']:.1f}% "
          f"of {NPU_PEAK_TFLOPS:.0f} TFLOPS")
    print(f"  Samples/sec:     {npu_metrics['samples_per_sec']:,.0f}")

    print(f"\n--- CPU Results (PyTorch bf16, {os.cpu_count()} cores) ---")
    print(f"  Latency:         {cpu_latency*1e3:.2f} ms")
    print(f"  Throughput:      {cpu_metrics['gflops']:.0f} GFLOPS")

    speedup = cpu_latency / npu_avg
    print(f"\n{'='*70}")
    print(f"  NPU:  {npu_avg*1e3:>8.2f} ms   {npu_metrics['tflops']:>5.2f} TFLOPS"
          f"  ({npu_metrics['pct_peak']:.1f}% peak)"
          f"   {npu_metrics['samples_per_sec']:>10,.0f} samp/s")
    print(f"  CPU:  {cpu_latency*1e3:>8.2f} ms   {cpu_metrics['tflops']:>5.2f} TFLOPS"
          f"               {cpu_metrics['samples_per_sec']:>10,.0f} samp/s")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"{'='*70}")


# ── Main benchmark ────────────────────────────────────────────────────────

def benchmark(H=128, B=16, num_tiles=24, num_iters=1000,
              warmup=3, timed_iters=10):
    """Run the full NPU vs CPU benchmark and print results.

    Args:
        H: Hidden dimension.
        B: Batch size per tile.
        num_tiles: Number of parallel compute tiles.
        num_iters: Hardware loop iterations (depth = 2 × num_iters).
        warmup: Number of NPU warmup runs.
        timed_iters: Number of timed NPU runs.
    """
    from iron.common.aie_context import AIEContext

    total_samples = num_tiles * B
    depth = 2 * num_iters
    flops_per_step = total_samples * 2 * H * H
    total_flops = depth * flops_per_step

    print(f"\n{'='*70}")
    print(f"Recurrent MLP Benchmark")
    print(f"  Network:    ReLU(x @ W) × {depth} "
          f"(hw loop: {num_iters} iters × 2)")
    print(f"  Tiles:      {num_tiles} "
          f"({min(num_tiles, 8)} cols × {(num_tiles + 7) // 8} rows)")
    print(f"  Batch:      {B}/tile × {num_tiles} tiles = "
          f"{total_samples} samples")
    print(f"  Weight:     {H}×{H} = {H*H:,} params ({H*H*2/1024:.0f} KB)")
    print(f"  FLOPs:      {total_flops/1e9:.2f} GFLOP per invocation")
    print(f"{'='*70}")

    # Generate test data
    W, X = generate_test_data(H, B, num_tiles)

    print("\nComputing CPU reference...")
    Y_ref = reference(X, W, depth)

    X_tiled = tile_activations(X, B, num_tiles)
    W_tiled = to_tiled(W)

    # Compile and run on NPU
    ctx = AIEContext()
    op = AIERecurrentMLP(
        H=H, B=B, num_tiles=num_tiles, num_iters=num_iters, context=ctx,
    )
    print("Compiling for NPU...")
    ctx.compile_all()
    ctx.prepare_runtime()

    npu_times, Y_npu_flat = run_npu(
        op, X_tiled, W_tiled, total_samples, H, warmup, timed_iters)
    Y_npu = untile_activations(Y_npu_flat, B, H, num_tiles)

    # Correctness
    correctness = check_correctness(Y_ref, Y_npu)

    # NPU metrics
    npu_avg = np.mean(npu_times)
    npu_metrics = compute_metrics(total_flops, npu_avg, num_tiles, total_samples)

    # CPU benchmark
    cpu_latency = run_cpu(X, W, depth)
    cpu_metrics = compute_metrics(total_flops, cpu_latency, num_tiles,
                                  total_samples)

    print_results(npu_metrics, cpu_metrics, correctness, npu_times,
                  cpu_latency, depth)


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark recurrent MLP on NPU vs CPU.")
    parser.add_argument("--H", type=int, default=128,
                        help="Hidden dimension (default: 128)")
    parser.add_argument("--B", type=int, default=16,
                        help="Batch size per tile (default: 16)")
    parser.add_argument("--tiles", type=int, default=24,
                        help="Number of compute tiles, 1-24 (default: 24)")
    parser.add_argument("--iters", type=int, default=1000,
                        help="Hardware loop count (depth = 2 × iters)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup runs (default: 3)")
    parser.add_argument("--timed", type=int, default=10,
                        help="Timed runs (default: 10)")
    args = parser.parse_args()

    benchmark(
        H=args.H, B=args.B,
        num_tiles=args.tiles, num_iters=args.iters,
        warmup=args.warmup, timed_iters=args.timed,
    )
