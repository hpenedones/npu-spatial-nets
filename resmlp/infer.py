"""
Run MNIST inference on the NPU using trained ResMLP weights.

Examples:
    python -m resmlp.infer resmlp/checkpoints/resmlp_hybrid_epoch009.pt
    python -m resmlp.infer resmlp/checkpoints/resmlp_full_npu_epoch009.pt

Hybrid checkpoints contain 32 residual layers and map directly onto the
32-tile snake. Full-NPU checkpoints contain 30 residual layers because one
tile is used for the embed and one for the head during training; inference pads
those two missing residual slots with identity layers.
"""

import argparse
import sys
import time

import numpy as np
import torch
from ml_dtypes import bfloat16
from torchvision import datasets, transforms

from resmlp import from_tiled, to_tiled
from resmlp.model import ResMLP
from resmlp.op import ResMLP as NPUResMLP, ROWS_PER_COL


def main():
    parser = argparse.ArgumentParser(description="MNIST inference on NPU")
    parser.add_argument("checkpoint", help="Path to trained .pt checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-cols", type=int, default=8)
    parser.add_argument("--bench", action="store_true", help="Show timing")
    args = parser.parse_args()

    B = 8
    num_cols = args.num_cols
    num_tiles = num_cols * ROWS_PER_COL

    print(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    H = args.hidden_dim if args.hidden_dim is not None else ckpt.get("hidden_dim", 160)
    num_layers = args.num_layers if args.num_layers is not None else ckpt.get("num_layers", 32)
    pipeline = ckpt.get("pipeline", "hybrid")

    if num_layers not in {num_tiles, num_tiles - 2}:
        raise ValueError(
            f"Checkpoint uses {num_layers} residual layers, but the {num_tiles}-tile "
            "inference pipeline can only handle models with either all tiles "
            "used as residual layers or with 2 identity-padded endpoints."
        )

    model = ResMLP(hidden_dim=H, num_layers=num_layers)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  pipeline={pipeline}, epoch={ckpt['epoch']}, test acc={ckpt.get('test_acc', '?')}")

    embed_weight = model.embed.weight.detach().float()
    embed_bias = model.embed.bias.detach().float()
    head_weight = model.head.weight.detach().float()
    head_bias = model.head.bias.detach().float()
    residual_weights = model.export_residual_weights()

    if num_layers == num_tiles - 2:
        zero_w = np.zeros((H, H), dtype=bfloat16)
        residual_weights = [zero_w] + residual_weights + [zero_w]
        print("  padded 30-layer checkpoint with 2 identity residual tiles for inference")

    print(f"Compiling NPU pipeline ({num_tiles} tiles, H={H})...")
    from iron.common.aie_context import AIEContext
    ctx = AIEContext()
    npu_op = NPUResMLP(H=H, B=B, num_cols=num_cols, context=ctx)
    ctx.compile_all()
    ctx.prepare_runtime()

    W_packed = np.concatenate([to_tiled(W) for W in residual_weights])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST("data", train=False, download=True,
                             transform=transform)

    correct = 0
    total = 0
    npu_time = 0.0
    cpu_time = 0.0
    zero_out = np.zeros(B * H, dtype=bfloat16)

    n_batches = (len(test_ds) + B - 1) // B
    print(f"Running inference on {len(test_ds)} images ({n_batches} batches)...")

    for batch_idx in range(n_batches):
        start = batch_idx * B
        end = min(start + B, len(test_ds))
        actual_B = end - start

        images = torch.stack([test_ds[i][0] for i in range(start, end)])
        labels = torch.tensor([test_ds[i][1] for i in range(start, end)])

        if actual_B < B:
            pad = torch.zeros(B - actual_B, *images.shape[1:])
            images = torch.cat([images, pad])

        t0 = time.perf_counter()
        x_flat = images.view(B, -1).float()
        x_hidden = (x_flat @ embed_weight.T + embed_bias).numpy()
        x_tiled = to_tiled(x_hidden.astype(bfloat16))
        cpu_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        npu_op.write_buffer("input", x_tiled)
        npu_op.write_buffer("weights", W_packed)
        npu_op.write_buffer("output", zero_out.copy())
        npu_op.run_runlist()
        npu_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        y_flat = npu_op.read_buffer("output", (B * H,), copy=True)
        y_np = from_tiled(y_flat, B, H).astype(np.float32)
        logits = torch.from_numpy(y_np) @ head_weight.T + head_bias
        preds = logits[:actual_B].argmax(1)
        correct += (preds == labels).sum().item()
        total += actual_B
        cpu_time += time.perf_counter() - t0

    accuracy = correct / total
    print(f"\n{'═' * 50}")
    print(f"NPU accuracy: {accuracy:.4f} ({correct}/{total})")

    if args.bench:
        print("\nTiming:")
        print(f"  NPU total:  {npu_time * 1000:.1f} ms ({npu_time / n_batches * 1000:.3f} ms/batch)")
        print(f"  CPU total:  {cpu_time * 1000:.1f} ms")
        print(f"  Throughput: {total / npu_time:.0f} images/sec (NPU only)")

    print("\nVerification (pure CPU):")
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
