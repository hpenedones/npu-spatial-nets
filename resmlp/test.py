"""
Test and benchmark for the residual MLP snake pipeline.

Three test modes:
  1. Zero weights  → identity: output should equal input
  2. Tiny identity → small perturbation: output ≈ 1.1 × input
  3. Random weights → compare NPU vs CPU reference

Usage:
    python resmlp/test.py --cols 1        # quick 4-tile test
    python resmlp/test.py --cols 8        # full 32-tile test
    python resmlp/test.py --cols 1 --bench  # benchmark mode
"""

import argparse
import time
import sys

import numpy as np
from ml_dtypes import bfloat16

from resmlp import to_tiled, from_tiled
from resmlp.op import ResMLP, ROWS_PER_COL
from iron.common.aie_context import AIEContext


# ── CPU reference ────────────────────────────────────────────────────────

def reference_resmlp(X, weights):
    """CPU reference: apply y = relu(x @ W) + x for each layer.

    Uses bf16 intermediate casts to approximate NPU's BFP16 quantization.
    """
    x = X.astype(np.float32)
    for W in weights:
        w = W.astype(np.float32)
        matmul_out = (x @ w).astype(bfloat16).astype(np.float32)
        relu_out = np.maximum(matmul_out, 0)
        x = (relu_out + x).astype(bfloat16).astype(np.float32)
    return x


# ── Host buffer assembly ────────────────────────────────────────────────

def pack_weights(weights, H):
    """Assemble flat host weight buffer: [W_0_tiled, W_1_tiled, ...]."""
    parts = [to_tiled(W) for W in weights]
    return np.concatenate(parts)


# ── Tests ────────────────────────────────────────────────────────────────

def run_test(H, B, num_cols, weights_fn, test_name, rtol=0.05, atol=0.01):
    """Run one test: compile, execute on NPU, compare to CPU reference."""
    num_tiles = num_cols * ROWS_PER_COL

    # Generate data
    rng = np.random.default_rng(42)
    X = rng.standard_normal((B, H)).astype(np.float32).astype(bfloat16)
    weights = [weights_fn(H, rng).astype(bfloat16) for _ in range(num_tiles)]

    # CPU reference
    Y_ref = reference_resmlp(X, weights).astype(bfloat16)

    # NPU
    ctx = AIEContext()
    op = ResMLP(H=H, B=B, num_cols=num_cols, context=ctx)
    print(f"  Compiling ({num_tiles} tiles, {H}×{H} weights)...", flush=True)
    ctx.compile_all()
    ctx.prepare_runtime()

    X_tiled = to_tiled(X)
    W_packed = pack_weights(weights, H)
    zero_out = np.zeros(B * H, dtype=bfloat16)

    # Warmup
    for _ in range(2):
        op.write_buffer("input", X_tiled)
        op.write_buffer("weights", W_packed)
        op.write_buffer("output", zero_out.copy())
        op.run_runlist()

    # Actual run
    op.write_buffer("input", X_tiled)
    op.write_buffer("weights", W_packed)
    op.write_buffer("output", zero_out.copy())
    op.run_runlist()

    Y_flat = op.read_buffer("output", (B * H,), copy=True)
    Y_npu = from_tiled(Y_flat, B, H)

    # Compare
    ref = Y_ref.astype(np.float32)
    npu = Y_npu.astype(np.float32)
    close = np.isclose(ref, npu, rtol=rtol, atol=atol)
    pct = close.mean() * 100
    max_diff = np.max(np.abs(ref - npu))

    status = "PASS" if pct > 95 else "FAIL"
    print(f"  [{status}] {test_name}: {pct:.1f}% close, max diff = {max_diff:.4f}")
    print(f"    Reference [0,:5]: {ref[0,:5]}")
    print(f"    NPU       [0,:5]: {npu[0,:5]}")

    return pct > 95


def run_benchmark(H, B, num_cols, num_iters=50):
    """Measure NPU latency and throughput."""
    num_tiles = num_cols * ROWS_PER_COL
    rng = np.random.default_rng(0)

    weights = [rng.standard_normal((H, H)).astype(np.float32).astype(bfloat16)
               * bfloat16(0.01) for _ in range(num_tiles)]

    ctx = AIEContext()
    op = ResMLP(H=H, B=B, num_cols=num_cols, context=ctx)
    print(f"  Compiling for benchmark...", flush=True)
    ctx.compile_all()
    ctx.prepare_runtime()

    X = rng.standard_normal((B, H)).astype(np.float32).astype(bfloat16)
    X_tiled = to_tiled(X)
    W_packed = pack_weights(weights, H)
    zero_out = np.zeros(B * H, dtype=bfloat16)

    # Warmup
    for _ in range(5):
        op.write_buffer("input", X_tiled)
        op.write_buffer("weights", W_packed)
        op.write_buffer("output", zero_out.copy())
        op.run_runlist()

    # Timed runs
    times = []
    for _ in range(num_iters):
        op.write_buffer("input", X_tiled)
        op.write_buffer("weights", W_packed)
        op.write_buffer("output", zero_out.copy())
        t0 = time.perf_counter()
        op.run_runlist()
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    flops_per_call = num_tiles * (2 * B * H * H + B * H)  # matmul + add
    lat = np.median(times_ms)
    gflops = flops_per_call / (lat * 1e-3) / 1e9

    print(f"\n  ── Benchmark ({num_tiles} tiles, B={B}, H={H}) ──")
    print(f"  Median latency: {lat:.3f} ms")
    print(f"  Throughput:     {gflops:.2f} GFLOPS")
    print(f"  Per-tile:       {gflops/num_tiles:.2f} GFLOPS")

    # CPU comparison
    t0 = time.perf_counter()
    for _ in range(10):
        reference_resmlp(X, weights)
    cpu_ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"  CPU latency:    {cpu_ms:.3f} ms")
    print(f"  Speedup:        {cpu_ms/lat:.1f}×")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test residual MLP on NPU")
    parser.add_argument("--H", type=int, default=160)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--cols", type=int, default=1, help="Columns (1–8)")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    args = parser.parse_args()

    H, B = args.H, args.B
    print(f"═══ Residual MLP Test (B={B}, H={H}, cols={args.cols}) ═══\n")

    all_pass = True

    # Test 1: Zero weights → identity (output = input)
    print("Test 1: Zero weights (identity)")
    ok = run_test(H, B, args.cols,
                  weights_fn=lambda H, rng: np.zeros((H, H), dtype=np.float32),
                  test_name="zero weights → identity")
    all_pass &= ok

    # Test 2: Small scaled identity → slight growth
    print("\nTest 2: Tiny identity (W = 0.1 × I)")
    ok = run_test(H, B, args.cols,
                  weights_fn=lambda H, rng: np.eye(H, dtype=np.float32) * 0.1,
                  test_name="0.1×I → gradual growth",
                  rtol=0.3, atol=1.0)
    all_pass &= ok

    # Test 3: Small random weights
    print("\nTest 3: Small random weights")
    ok = run_test(H, B, args.cols,
                  weights_fn=lambda H, rng: rng.standard_normal((H, H)).astype(np.float32) * 0.01,
                  test_name="random (σ=0.01)",
                  rtol=0.3, atol=0.5)
    all_pass &= ok

    if args.bench:
        print()
        run_benchmark(H, B, args.cols)

    print(f"\n{'═' * 50}")
    print(f"{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
