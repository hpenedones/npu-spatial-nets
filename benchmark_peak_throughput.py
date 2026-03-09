#!/usr/bin/env python3
"""
TileFlow — Phase 1: NPU Peak Throughput Benchmark

Measures the maximum compute throughput achievable on the XDNA 2 NPU
by running GEMM (matrix multiplication) across all 32 AIE tiles (8 columns × 4 rows)
using the IRON/MLIR-AIE toolchain.

Theoretical peak: ~25 TFLOPS (bfloat16) on Strix Point.

Each GEMM configuration runs in a separate subprocess because IRON's runtime
is a singleton that can only be initialized once per process.

Usage:
    source /opt/xilinx/xrt/setup.sh
    source .venv/bin/activate
    python benchmark_peak_throughput.py
"""

import json
import os
import subprocess
import sys
import time

import torch

IRON_DIR = "/home/hpenedones/source/IRON"

# ── Benchmark configurations ────────────────────────────────────────────────
# Each: (M, K, N, tile_m, tile_k, tile_n, num_cols, b_col_maj, c_col_maj)
# Configs taken from IRON's own passing test suite.
CONFIGS = [
    # Square matrices — increasing compute-to-transfer ratio
    (2048, 2048, 2048, 64, 64, 64, 1, False, False),
    (2048, 2048, 2048, 64, 64, 64, 2, True, False),
    (2048, 2048, 2048, 64, 64, 64, 8, True, True),
    # Larger — more compute to amortize DMA
    (4096, 4096, 4096, 64, 64, 64, 8, True, True),
    # Rectangular
    (4096, 2048, 2048, 64, 64, 64, 8, True, True),
    (2048, 2048, 4096, 64, 64, 64, 8, True, True),
]


# This script is the subprocess worker when called with --worker
WORKER_SCRIPT = r"""
import json, os, sys, logging
logging.basicConfig(level=logging.WARNING)

os.chdir("{iron_dir}")
sys.path.insert(0, "{iron_dir}")

from iron.common.aie_context import AIEContext
from iron.common.test_utils import run_test
from iron.operators.gemm.op import AIEGEMM
from iron.operators.gemm.reference import generate_golden_reference

cfg = json.loads('{cfg_json}')
M, K, N = cfg["M"], cfg["K"], cfg["N"]
tm, tk, tn = cfg["tile_m"], cfg["tile_k"], cfg["tile_n"]
nc = cfg["num_cols"]
b_col = cfg["b_col_maj"]
c_col = cfg["c_col_maj"]
warmup = cfg["warmup"]
iters = cfg["iters"]

ctx = AIEContext()
golden = generate_golden_reference(M=M, K=K, N=N, b_col_maj=b_col, c_col_maj=c_col)
op = AIEGEMM(
    M=M, K=K, N=N,
    tile_m=tm, tile_k=tk, tile_n=tn,
    num_aie_columns=nc,
    b_col_maj=b_col, c_col_maj=c_col,
    prio_accuracy=True,
    context=ctx,
)
input_bufs = {{"A": golden["input"].flatten()}}
output_bufs = {{}}
for i in range(1):
    input_bufs[f"B_{{i}}"] = golden["input_b"][i].flatten()
    output_bufs[f"C_{{i}}"] = golden["output"][i].flatten()

errors, latency_us, bw_gbps = run_test(
    op, input_bufs, output_bufs,
    warmup_iters=warmup, timed_iters=iters,
    rel_tol=0.005, abs_tol=0.005,
)

flops = 2.0 * M * K * N
gflops = flops / (latency_us * 1e-6) / 1e9
n_errors = sum(len(v) for v in errors.values()) if errors else 0

result = {{
    "latency_us": latency_us,
    "gflops": gflops,
    "tflops": gflops / 1e3,
    "bw_gbps": bw_gbps,
    "errors": n_errors,
}}
print("RESULT:" + json.dumps(result))
"""


def run_npu_gemm_subprocess(M, K, N, tile_m, tile_k, tile_n, num_cols,
                             b_col_maj, c_col_maj, warmup=5, iters=10):
    """Run a single GEMM config in a subprocess (IRON needs fresh runtime)."""
    cfg = {
        "M": M, "K": K, "N": N,
        "tile_m": tile_m, "tile_k": tile_k, "tile_n": tile_n,
        "num_cols": num_cols,
        "b_col_maj": b_col_maj, "c_col_maj": c_col_maj,
        "warmup": warmup, "iters": iters,
    }
    cfg_json = json.dumps(cfg)
    script = WORKER_SCRIPT.format(iron_dir=IRON_DIR, cfg_json=cfg_json)

    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600, env=env,
    )

    # Parse the RESULT line from stdout
    for line in result.stdout.splitlines():
        if line.startswith("RESULT:"):
            data = json.loads(line[7:])
            data.update({"M": M, "K": K, "N": N, "cols": num_cols})
            return data

    # If no result line, report the error
    stderr_tail = result.stderr.strip().splitlines()[-5:] if result.stderr.strip() else []
    raise RuntimeError(
        f"Worker failed (rc={result.returncode}): "
        + " | ".join(stderr_tail)
    )


def benchmark_cpu_gemm(M, K, N, warmup=5, iters=20):
    """Run GEMM on CPU (PyTorch bfloat16) for comparison."""
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)

    for _ in range(warmup):
        torch.matmul(A, B)

    start = time.perf_counter()
    for _ in range(iters):
        torch.matmul(A, B)
    elapsed = (time.perf_counter() - start) / iters

    flops = 2.0 * M * K * N
    gflops = flops / elapsed / 1e9
    return {"latency_us": elapsed * 1e6, "gflops": gflops, "tflops": gflops / 1e3}


def main():
    print("=" * 80)
    print("TileFlow — Phase 1: NPU Peak Throughput Benchmark (bfloat16 GEMM)")
    print("=" * 80)
    print("Hardware: XDNA 2 Strix Point — 32 AIE2 tiles (8 cols × 4 rows)")
    print("Theoretical peak: ~25 TFLOPS (bfloat16)")
    print()

    hdr = f"{'Config':>28s} {'Latency':>12s} {'GFLOPS':>10s} {'TFLOPS':>8s} {'BW GB/s':>10s} {'OK':>12s}"
    print(hdr)
    print("-" * 80)

    results = []
    for M, K, N, tm, tk, tn, nc, bcol, ccol in CONFIGS:
        label = f"{M}×{K}×{N} ({nc}col)"
        try:
            r = run_npu_gemm_subprocess(M, K, N, tm, tk, tn, nc, bcol, ccol)
            results.append(r)
            ok = "✓" if r["errors"] == 0 else f"✗({r['errors']})"
            print(f"{label:>28s} {r['latency_us']:>10.1f}µs {r['gflops']:>10.1f} "
                  f"{r['tflops']:>7.2f}T {r['bw_gbps']:>10.2f} {ok:>12s}")
        except Exception as e:
            err_str = str(e)[:60]
            print(f"{label:>28s}  ERROR: {err_str}")

    if not results:
        print("\nNo NPU results — aborting.")
        return

    best = max(results, key=lambda r: r["tflops"])
    print("-" * 80)
    print(f"{'Peak NPU':>28s} {'':>12s} {best['gflops']:>10.1f} "
          f"{best['tflops']:>7.2f}T")
    print()

    # ── CPU comparison ───────────────────────────────────────────────────
    print("CPU comparison (PyTorch bfloat16 matmul):")
    cpu_results = []
    for M, K, N, *_ in CONFIGS:
        cpu = benchmark_cpu_gemm(M, K, N)
        cpu_results.append(cpu)
        print(f"  {M}×{K}×{N}: {cpu['latency_us']:>10.0f} µs, "
              f"{cpu['gflops']:.1f} GFLOPS ({cpu['tflops']:.3f} TFLOPS)")

    best_cpu = max(cpu_results, key=lambda r: r["tflops"])
    speedup = best["gflops"] / best_cpu["gflops"]
    efficiency = best["tflops"] / 25.0 * 100

    print()
    print("=" * 80)
    print(f"  NPU peak:       {best['tflops']:.2f} TFLOPS (bfloat16)")
    print(f"  CPU peak:       {best_cpu['tflops']:.3f} TFLOPS (bfloat16)")
    print(f"  Speedup:        {speedup:,.1f}×")
    print(f"  NPU efficiency: {efficiency:.1f}% of theoretical 25 TFLOPS")
    print("=" * 80)


if __name__ == "__main__":
    main()
