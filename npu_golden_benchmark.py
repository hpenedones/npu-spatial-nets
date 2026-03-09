"""
TileFlow — Spatial Neural Network Benchmark (CPU Baseline)

This script defines the *logical* architecture of the spatial MLP that will be
mapped to the XDNA 2 NPU's 8×4 tile array via IRON/MLIR-AIE.

It serves two purposes:
  1. Document the network architecture in familiar PyTorch terms.
  2. Provide a CPU baseline measurement for comparison with the NPU.

The NPU implementation (Phase 2) will use IRON to explicitly place each layer
on a column of tiles and connect them with ObjectFIFO hardware streams.

Architecture:
  - 8 layers (one per NPU column), each is a small linear + ReLU
  - INT8 weights and activations (matching NPU's 512 INT8 MACs/cycle/tile)
  - 4-way data parallelism within each layer (one per NPU row)
  - Input/output: INT8 vectors

Hardware target:
  - 32 AIE2 tiles (8 cols × 4 rows)
  - ~48 KB SRAM per tile
  - Peak: 50 TOPS INT8

Weight budget per tile:
  Each tile has ~48 KB SRAM. A layer split across 4 tiles means each tile
  holds 1/4 of the layer's weights. For a layer with input_dim=N and
  output_dim=M in INT8: weight size = N*M bytes, split 4 ways = N*M/4 per tile.
  With 48 KB budget (leaving room for activations + code), ~32 KB for weights
  gives us layers up to roughly 128×256 or 256×128 per column.
"""

import time
import numpy as np

# ── Architecture parameters (must match the IRON NPU implementation) ────────

NUM_LAYERS = 8        # One per NPU column
HIDDEN_DIM = 128      # Activation vector width (INT8)
INPUT_DIM = 128
OUTPUT_DIM = 128
BATCH_SIZE = 1        # Single-sample latency benchmark

# The 4-way row parallelism is implicit in the NPU implementation.
# On CPU we just do the full matmul — the math is identical.

def build_weights(rng):
    """Generate random INT8 weight matrices for each layer."""
    weights = []
    biases = []
    for i in range(NUM_LAYERS):
        in_dim = INPUT_DIM if i == 0 else HIDDEN_DIM
        out_dim = OUTPUT_DIM if i == NUM_LAYERS - 1 else HIDDEN_DIM
        W = rng.integers(-128, 127, size=(out_dim, in_dim), dtype=np.int8)
        b = rng.integers(-128, 127, size=(out_dim,), dtype=np.int8)
        weights.append(W)
        biases.append(b)
    return weights, biases


def forward_int8(x, weights, biases):
    """
    Forward pass using INT8 arithmetic (simulated on CPU via int32 accumulation).

    On the NPU, each layer runs on one column of 4 tiles. The matmul is split
    across tiles (each tile handles out_dim/4 rows of W), and the results are
    concatenated via stream switches.

    On CPU, we simply do the full matmul in int32 (since NumPy doesn't have
    native int8 matmul) and clip back to int8 range — this is the same
    mathematical result, just slower.
    """
    for W, b in zip(weights, biases):
        # INT8 matmul: accumulate in int32, then quantize back
        x = W.astype(np.int32) @ x.astype(np.int32) + b.astype(np.int32)
        # ReLU + re-quantize to INT8 range
        x = np.clip(x, 0, 127).astype(np.int8)
    return x


def benchmark_cpu(weights, biases, n_warmup=100, n_iter=10000):
    """Measure CPU inference latency."""
    rng = np.random.default_rng(42)
    x = rng.integers(-128, 127, size=(INPUT_DIM,), dtype=np.int8)

    # Warmup
    for _ in range(n_warmup):
        forward_int8(x, weights, biases)

    # Timed run
    start = time.perf_counter()
    for _ in range(n_iter):
        forward_int8(x, weights, biases)
    elapsed = time.perf_counter() - start

    latency_us = (elapsed / n_iter) * 1e6
    throughput = n_iter / elapsed

    # Compute total operations per inference
    total_ops = 0
    for i, W in enumerate(weights):
        out_dim, in_dim = W.shape
        total_ops += 2 * out_dim * in_dim  # multiply + accumulate
    tops_achieved = (total_ops * throughput) / 1e12

    return latency_us, throughput, tops_achieved, total_ops


def main():
    print("=" * 65)
    print("TileFlow — Spatial MLP Benchmark (CPU Baseline)")
    print("=" * 65)
    print()
    print(f"Architecture: {NUM_LAYERS}-layer MLP, {INPUT_DIM}→{HIDDEN_DIM}→{OUTPUT_DIM}")
    print(f"Precision:    INT8 weights & activations (int32 accumulation)")
    print(f"Batch size:   {BATCH_SIZE}")
    print()

    rng = np.random.default_rng(0)
    weights, biases = build_weights(rng)

    # Print weight budget analysis
    total_params = sum(W.size + b.size for W, b in zip(weights, biases))
    print(f"Total parameters: {total_params:,} ({total_params / 1024:.1f} KB in INT8)")
    for i, (W, b) in enumerate(zip(weights, biases)):
        tile_bytes = W.size // 4  # 4 tiles per column
        print(f"  Layer {i}: {W.shape[1]}×{W.shape[0]} = {W.size:,} bytes "
              f"({tile_bytes:,} bytes/tile, "
              f"{'✓' if tile_bytes < 32768 else '✗'} fits in 32 KB)")
    print()

    # Run benchmark
    print("Running CPU benchmark (10,000 iterations)...")
    latency_us, throughput, tops, total_ops = benchmark_cpu(weights, biases)

    print(f"\nResults:")
    print(f"  Latency:    {latency_us:.1f} µs per inference")
    print(f"  Throughput: {throughput:,.0f} inferences/sec")
    print(f"  Ops/infer:  {total_ops:,}")
    print(f"  CPU TOPS:   {tops:.6f}")
    print()

    # NPU theoretical comparison
    npu_tops = 50.0  # Theoretical peak
    npu_realistic_tops = 38.0  # Achieved in GEMM benchmarks
    npu_latency_theoretical = (total_ops / (npu_tops * 1e12)) * 1e6
    npu_latency_realistic = (total_ops / (npu_realistic_tops * 1e12)) * 1e6

    print(f"NPU projection (theoretical {npu_tops} TOPS):")
    print(f"  Latency:    {npu_latency_theoretical:.3f} µs per inference")
    print(f"  Speedup:    {latency_us / npu_latency_theoretical:,.0f}×")
    print()
    print(f"NPU projection (realistic {npu_realistic_tops} TOPS):")
    print(f"  Latency:    {npu_latency_realistic:.3f} µs per inference")
    print(f"  Speedup:    {latency_us / npu_latency_realistic:,.0f}×")
    print()
    print("Next step: implement the IRON/MLIR-AIE version (Phase 2) to measure")
    print("actual NPU performance and compare against this CPU baseline.")


if __name__ == "__main__":
    main()
