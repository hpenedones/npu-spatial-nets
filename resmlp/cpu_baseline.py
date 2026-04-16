"""CPU throughput baseline for the residual MLP.

Measures pure-CPU forward-pass throughput for the same model configurations
reported in the paper (H=32/L=8 speed variant, H=64/L=32 accuracy variant).
This provides the missing baseline identified in the peer review (Issue #1):
the NPU throughput claim needs a CPU comparison point.

Usage:
    python -m resmlp.cpu_baseline
    python -m resmlp.cpu_baseline --hidden-dim 64 --num-layers 32 --batch-size 256
"""

import argparse
import sys
import time

import numpy as np
import torch

from resmlp.model import ResMLP

# Paper configurations
CONFIGS = {
    "speed": {"hidden_dim": 32, "num_layers": 8},
    "accuracy": {"hidden_dim": 64, "num_layers": 32},
}

INPUT_DIM = 28
NUM_CLASSES = 2


def benchmark_pytorch(model, batch_size, num_samples, warmup_calls=10):
    """Benchmark PyTorch forward pass on CPU. Returns samples/sec."""
    dummy = torch.randn(batch_size, INPUT_DIM)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_calls):
            model(dummy)

    calls = max(1, num_samples // batch_size)
    processed = calls * batch_size

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(calls):
            model(dummy)
        elapsed = time.perf_counter() - t0

    return {
        "num_samples_processed": processed,
        "wall_s": elapsed,
        "samples_per_s": processed / elapsed,
    }


def benchmark_numpy_decomposed(model, batch_size, num_samples, warmup_calls=10):
    """Benchmark the same computation decomposed into numpy ops (embed + residual + head).

    This mirrors the NPU inference path more closely: the embed and head run
    on CPU via numpy, and the residual body is a sequential chain of matmul+relu+skip.
    """
    embed_w = model.embed.weight.detach().float().numpy()  # [H, input_dim]
    embed_b = model.embed.bias.detach().float().numpy()  # [H]
    head_w = model.head.weight.detach().float().numpy()  # [C, H]
    head_b = model.head.bias.detach().float().numpy()  # [C]
    residual_ws = [
        layer.weight.detach().float().numpy() for layer in model.layers
    ]

    dummy = np.random.randn(batch_size, INPUT_DIM).astype(np.float32)

    def forward(x):
        h = x @ embed_w.T + embed_b
        for W in residual_ws:
            h = np.maximum(0, h @ W) + h
        return h @ head_w.T + head_b

    # Warmup
    for _ in range(warmup_calls):
        forward(dummy)

    calls = max(1, num_samples // batch_size)
    processed = calls * batch_size

    t0 = time.perf_counter()
    for _ in range(calls):
        forward(dummy)
    elapsed = time.perf_counter() - t0

    return {
        "num_samples_processed": processed,
        "wall_s": elapsed,
        "samples_per_s": processed / elapsed,
    }


def run_config(name, hidden_dim, num_layers, batch_sizes, num_samples):
    """Run benchmarks for one model configuration across batch sizes."""
    model = ResMLP(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Config: {name}  (H={hidden_dim}, L={num_layers}, params={n_params:,})")
    print(f"{'='*60}")

    for bs in batch_sizes:
        print(f"\n  batch_size={bs}, num_samples={num_samples:,}")

        pt = benchmark_pytorch(model, bs, num_samples)
        print(f"    PyTorch CPU:        {pt['samples_per_s']:>12,.0f} samples/s"
              f"  ({pt['wall_s']:.3f}s)")

        np_res = benchmark_numpy_decomposed(model, bs, num_samples)
        print(f"    NumPy decomposed:   {np_res['samples_per_s']:>12,.0f} samples/s"
              f"  ({np_res['wall_s']:.3f}s)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CPU throughput baseline for the residual MLP"
    )
    parser.add_argument(
        "--configs", nargs="*", default=["speed", "accuracy"],
        choices=list(CONFIGS.keys()),
        help="Which paper configurations to benchmark (default: both)",
    )
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Override hidden dim (runs a single custom config)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override num layers (runs a single custom config)")
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default=[8, 64, 256, 1024, 4096],
        help="Batch sizes to test",
    )
    parser.add_argument("--num-samples", type=int, default=1_000_000,
                        help="Total samples per benchmark run")
    args = parser.parse_args(argv)

    torch.set_num_threads(torch.get_num_threads())
    print(f"CPU baseline benchmark")
    print(f"  PyTorch {torch.__version__}")
    print(f"  Threads: {torch.get_num_threads()}")
    print(f"  NumPy {np.__version__}")

    if args.hidden_dim is not None or args.num_layers is not None:
        h = args.hidden_dim or 64
        l = args.num_layers or 32
        run_config("custom", h, l, args.batch_sizes, args.num_samples)
    else:
        for name in args.configs:
            cfg = CONFIGS[name]
            run_config(name, cfg["hidden_dim"], cfg["num_layers"],
                       args.batch_sizes, args.num_samples)

    print(f"\n{'='*60}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
