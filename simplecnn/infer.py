"""
Run fully-on-NPU MNIST inference with the one-column convnet.

Examples:
    python -m simplecnn.infer path/to/checkpoint.pt
    python -m simplecnn.infer path/to/checkpoint.pt --max-batches 32 --verify-batches 4
"""

import argparse
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from iron.common.aie_context import AIEContext
from simplecnn.config import BATCH_SIZE, IMG_H, IMG_W, N_CLASSES
from simplecnn.inference_op import SimpleCNNInferencePipeline
from simplecnn.model import TinyConvNet
from simplecnn.reference import forward_reference_logits, pack_images, quantized_model_copy


def get_test_loader(batch_size: int, data_dir: str = "data") -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def pad_images(images: torch.Tensor) -> tuple[torch.Tensor, int]:
    actual_batch = images.size(0)
    if actual_batch == BATCH_SIZE:
        return images, actual_batch
    pad = torch.zeros(BATCH_SIZE - actual_batch, 1, IMG_H, IMG_W, dtype=images.dtype)
    return torch.cat([images, pad], dim=0), actual_batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fully-on-NPU simple convnet")
    parser.add_argument("checkpoint", help="Path to a TinyConvNet checkpoint")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--verify-batches",
        type=int,
        default=0,
        help="Compare NPU logits against the quantized CPU reference on the first N batches",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    if args.batch_size != BATCH_SIZE:
        raise ValueError(f"Batch size must be {BATCH_SIZE}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TinyConvNet()
    model.load_state_dict(ckpt["model"])
    model.eval()
    quantized_model = quantized_model_copy(model)
    packed_weights = quantized_model.export_packed_weights()

    print(f"Loading {args.checkpoint}...")
    print(
        f"  epoch={ckpt.get('epoch', '?')}  "
        f"test_acc={ckpt.get('test_acc', '?')}  "
        f"architecture={ckpt.get('architecture', 'conv3-gap-linear')}"
    )

    print("Compiling NPU pipeline (1 column, 4 tiles)...", flush=True)
    t0 = time.time()
    ctx = AIEContext()
    npu_op = SimpleCNNInferencePipeline(context=ctx)
    ctx.compile_all()
    ctx.prepare_runtime()
    print(f"  Compiled in {time.time() - t0:.1f}s")

    npu_op.write_weights(packed_weights)
    loader = get_test_loader(args.batch_size, data_dir=args.data_dir)

    correct = 0
    total = 0
    npu_time = 0.0
    verify_close = []
    verify_max_abs = 0.0
    verify_pred_match = True

    for batch_idx, (images, labels) in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        padded_images, actual_batch = pad_images(images)
        t0 = time.perf_counter()
        logits_flat = npu_op.run_batch(pack_images(padded_images))
        npu_time += time.perf_counter() - t0

        logits = np.asarray(logits_flat, dtype=np.float32).reshape(BATCH_SIZE, N_CLASSES)
        preds = logits[:actual_batch].argmax(1)
        correct += int((preds == labels.numpy()).sum())
        total += actual_batch

        if batch_idx < args.verify_batches:
            ref_logits = (
                forward_reference_logits(quantized_model, padded_images)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            verify_close.append(
                np.isclose(ref_logits, logits, rtol=0.05, atol=0.05).mean() * 100.0
            )
            verify_max_abs = max(
                verify_max_abs,
                float(np.max(np.abs(ref_logits - logits))),
            )
            verify_pred_match &= np.array_equal(
                ref_logits[:actual_batch].argmax(1),
                preds,
            )

    accuracy = correct / total if total else 0.0
    print(f"\n{'═' * 50}")
    print(f"NPU accuracy: {accuracy:.4f} ({correct}/{total})")
    if total:
        print(f"Throughput:   {total / npu_time:.0f} images/sec (NPU loop)")

    if verify_close:
        mean_close = float(np.mean(verify_close))
        print("\nVerification:")
        print(f"  mean close: {mean_close:.1f}%")
        print(f"  max |Δ|:    {verify_max_abs:.4f}")
        print(f"  pred match: {'yes' if verify_pred_match else 'no'}")
        if mean_close < 95.0 or not verify_pred_match:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
