import argparse
import sys

import numpy as np
import torch

from iron.common.aie_context import AIEContext
from simplecnn.config import BATCH_SIZE, IMG_H, IMG_W, N_CLASSES
from simplecnn.inference_op import SimpleCNNInferencePipeline
from simplecnn.model import TinyConvNet
from simplecnn.reference import forward_reference_logits, pack_images, quantized_model_copy


def compare_logits(ref: np.ndarray, got: np.ndarray, *, rtol: float, atol: float) -> bool:
    close = np.isclose(ref, got, rtol=rtol, atol=atol)
    pct = close.mean() * 100.0
    max_diff = float(np.max(np.abs(ref - got)))
    status = "PASS" if pct > 95.0 else "FAIL"
    print(f"  [{status}] logits: {pct:.1f}% close, max diff = {max_diff:.4f}")
    return pct > 95.0


def compare_exact(name: str, ref: np.ndarray, got: np.ndarray) -> bool:
    ok = np.array_equal(ref, got)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        print(f"    ref: {ref}")
        print(f"    got: {got}")
    return ok


def run_test(conv_scale: float = 0.25, head_scale: float = 0.25) -> bool:
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    model = TinyConvNet()
    model.scale_initial_weights(conv_scale=conv_scale, head_scale=head_scale)
    quantized_model = quantized_model_copy(model)

    images = torch.from_numpy(
        (rng.standard_normal((BATCH_SIZE, 1, IMG_H, IMG_W)) * 0.25).astype(np.float32)
    )
    ref_logits = (
        forward_reference_logits(quantized_model, images)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    ctx = AIEContext()
    op = SimpleCNNInferencePipeline(context=ctx)
    print("Compiling simplecnn inference pipeline...", flush=True)
    ctx.compile_all()
    ctx.prepare_runtime()
    op.write_weights(quantized_model.export_packed_weights())

    got_runs = []
    for _ in range(2):
        logits_flat = op.run_batch(pack_images(images))
        got_runs.append(
            np.asarray(logits_flat, dtype=np.float32).reshape(BATCH_SIZE, N_CLASSES)
        )

    ok = True
    ok &= compare_logits(ref_logits, got_runs[0], rtol=0.05, atol=0.05)
    ok &= compare_exact("predictions", ref_logits.argmax(1), got_runs[0].argmax(1))
    ok &= compare_exact("repeat stability", got_runs[0], got_runs[1])
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate simplecnn fully-on-NPU inference")
    parser.add_argument("--conv-scale", type=float, default=0.25)
    parser.add_argument("--head-scale", type=float, default=0.25)
    args = parser.parse_args()

    print("═══ SimpleCNN NPU Inference Test ═══\n")
    ok = run_test(conv_scale=args.conv_scale, head_scale=args.head_scale)
    print(f"\n{'═' * 50}")
    print("ALL PASSED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
