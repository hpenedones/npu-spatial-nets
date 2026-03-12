"""
Run MNIST inference on the NPU using trained ResMLP weights.

Usage:
    python resmlp/infer.py resmlp/checkpoints/resmlp_epoch009.pt
    python resmlp/infer.py resmlp/checkpoints/resmlp_epoch009.pt --bench

Flow:
    1. Load trained PyTorch model → extract 32 weight matrices as bf16
    2. Compile IRON snake pipeline for the NPU
    3. For each batch of 8 MNIST images:
       a. CPU: embed (784 → 160)
       b. NPU: 32 residual layers (160 → 160) in one call
       c. CPU: classify (160 → 10)
    4. Report accuracy and timing
"""

import argparse
import sys
import time

import numpy as np
import torch
from ml_dtypes import bfloat16
from torchvision import datasets, transforms

from resmlp import to_tiled, from_tiled
from resmlp.model import ResMLP
from resmlp.op import ResMLP as NPUResMLP, ROWS_PER_COL


def main():
    parser = argparse.ArgumentParser(description="MNIST inference on NPU")
    parser.add_argument("checkpoint", help="Path to trained .pt checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-cols", type=int, default=8)
    parser.add_argument("--bench", action="store_true", help="Show timing")
    args = parser.parse_args()

    B = 8   # NPU batch size
    H = args.hidden_dim
    num_cols = args.num_cols
    num_tiles = num_cols * ROWS_PER_COL
    assert args.num_layers == num_tiles, \
        f"Model has {args.num_layers} layers but NPU has {num_tiles} tiles"

    # ── Load model ───────────────────────────────────────────────────
    print(f"Loading {args.checkpoint}...")
    model = ResMLP(hidden_dim=H, num_layers=args.num_layers)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Checkpoint epoch {ckpt['epoch']}, test acc = {ckpt.get('test_acc', '?')}")

    # Extract CPU layers + NPU weights
    embed_weight = model.embed.weight.detach().float()
    embed_bias = model.embed.bias.detach().float()
    head_weight = model.head.weight.detach().float()
    head_bias = model.head.bias.detach().float()
    npu_weights = model.export_npu_weights()  # list of (H,H) bf16

    # ── Compile NPU operator ─────────────────────────────────────────
    print(f"Compiling NPU pipeline ({num_tiles} tiles, H={H})...")
    from iron.common.aie_context import AIEContext
    ctx = AIEContext()
    npu_op = NPUResMLP(H=H, B=B, num_cols=num_cols, context=ctx)
    ctx.compile_all()
    ctx.prepare_runtime()

    # Pack weights once
    W_packed = np.concatenate([to_tiled(W) for W in npu_weights])

    # ── Load MNIST ───────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST("data", train=False, download=True,
                             transform=transform)

    # ── Inference loop ───────────────────────────────────────────────
    correct = 0
    total = 0
    npu_time = 0
    cpu_time = 0

    zero_out = np.zeros(B * H, dtype=bfloat16)

    n_batches = (len(test_ds) + B - 1) // B
    print(f"Running inference on {len(test_ds)} images ({n_batches} batches)...")

    for batch_idx in range(n_batches):
        start = batch_idx * B
        end = min(start + B, len(test_ds))
        actual_B = end - start

        # Gather batch
        images = torch.stack([test_ds[i][0] for i in range(start, end)])
        labels = torch.tensor([test_ds[i][1] for i in range(start, end)])

        # Pad to B if needed
        if actual_B < B:
            pad = torch.zeros(B - actual_B, *images.shape[1:])
            images = torch.cat([images, pad])

        # CPU: embed
        t0 = time.perf_counter()
        x_flat = images.view(B, -1).float()
        x_hidden = (x_flat @ embed_weight.T + embed_bias).numpy()
        x_bf16 = x_hidden.astype(bfloat16)
        x_tiled = to_tiled(x_bf16)
        cpu_time += time.perf_counter() - t0

        # NPU: 32 residual layers
        t0 = time.perf_counter()
        npu_op.write_buffer("input", x_tiled)
        npu_op.write_buffer("weights", W_packed)
        npu_op.write_buffer("output", zero_out.copy())
        npu_op.run_runlist()
        npu_time += time.perf_counter() - t0

        # CPU: classify
        t0 = time.perf_counter()
        y_flat = npu_op.read_buffer("output", (B * H,), copy=True)
        y_np = from_tiled(y_flat, B, H).astype(np.float32)
        y_torch = torch.from_numpy(y_np)
        logits = y_torch @ head_weight.T + head_bias
        preds = logits[:actual_B].argmax(1)
        correct += (preds == labels).sum().item()
        total += actual_B
        cpu_time += time.perf_counter() - t0

    accuracy = correct / total
    print(f"\n{'═' * 50}")
    print(f"NPU accuracy: {accuracy:.4f} ({correct}/{total})")

    if args.bench:
        print(f"\nTiming:")
        print(f"  NPU total:  {npu_time*1000:.1f} ms ({npu_time/n_batches*1000:.3f} ms/batch)")
        print(f"  CPU total:  {cpu_time*1000:.1f} ms")
        print(f"  Throughput: {total/npu_time:.0f} images/sec (NPU only)")

    # Compare to pure CPU
    print(f"\nVerification (pure CPU):")
    with torch.no_grad():
        sample = torch.stack([test_ds[i][0] for i in range(100)])
        cpu_logits = model(sample)
        cpu_preds = cpu_logits.argmax(1)
        cpu_labels = torch.tensor([test_ds[i][1] for i in range(100)])
        cpu_acc = (cpu_preds == cpu_labels).float().mean().item()
    print(f"  CPU accuracy (100 samples): {cpu_acc:.4f}")

    return 0 if accuracy > 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
