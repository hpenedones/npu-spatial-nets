import argparse
import sys

import numpy as np
import torch

from iron.common.aie_context import AIEContext
from convsnake.config import BATCH_SIZE, IMG_H, IMG_W, N_CLASSES, num_blocks_for_cols
from convsnake.model import StreamingConvNet
from convsnake.reference import forward_reference_logits, pack_image_batches, quantized_model_copy
from convsnake.streaming_op import StreamingConvSnake


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


def compare_predictions(ref_logits: np.ndarray, got_logits: np.ndarray, *, margin_tol: float = 1e-3) -> bool:
    ref_pred = ref_logits.argmax(2)
    got_pred = got_logits.argmax(2)
    mismatches = ref_pred != got_pred
    if not mismatches.any():
        print("  [PASS] predictions")
        return True

    def top_margin(logits: np.ndarray) -> np.ndarray:
        top2 = np.partition(logits, -2, axis=2)[:, :, -2:]
        return top2[:, :, 1] - top2[:, :, 0]

    ref_margin = top_margin(ref_logits)
    got_margin = top_margin(got_logits)
    severe = mismatches & (ref_margin > margin_tol) & (got_margin > margin_tol)
    ok = not severe.any()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] predictions")
    if not ok:
        print(f"    ref: {ref_pred}")
        print(f"    got: {got_pred}")
    return ok


def run_test(cols: int = 1, stream_depth: int = 4, conv_scale: float = 0.25, head_scale: float = 0.25) -> bool:
    num_blocks = num_blocks_for_cols(cols)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    model = StreamingConvNet(num_same_blocks=num_blocks)
    model.scale_initial_weights(conv_scale=conv_scale, head_scale=head_scale)
    quantized_model = quantized_model_copy(model)

    images = torch.from_numpy(
        (rng.standard_normal((stream_depth, BATCH_SIZE, 1, IMG_H, IMG_W)) * 0.25).astype(np.float32)
    )
    ref_logits = np.stack(
        [
            forward_reference_logits(quantized_model, images[idx])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            for idx in range(stream_depth)
        ]
    )

    ctx = AIEContext(use_runlist=False)
    op = StreamingConvSnake(
        quantized_model.export_npu_weights(),
        num_cols=cols,
        stream_depth=stream_depth,
        context=ctx,
    )
    print(f"Compiling convsnake streaming pipeline ({cols * 4} tiles, stream_depth={stream_depth})...", flush=True)
    ctx.compile_all()
    ctx.prepare_runtime()

    packed_images = pack_image_batches(images)
    got_runs = []
    for _ in range(2):
        op.write_buffer("input", packed_images)
        op.run_stream()
        logits_flat = op.read_buffer(
            "output",
            (stream_depth * BATCH_SIZE * N_CLASSES,),
            copy=True,
        )
        got_runs.append(
            np.asarray(logits_flat, dtype=np.float32).reshape(stream_depth, BATCH_SIZE, N_CLASSES)
        )

    ok = True
    ok &= compare_logits(ref_logits, got_runs[0], rtol=0.05, atol=0.05)
    ok &= compare_predictions(ref_logits, got_runs[0])
    ok &= compare_exact("repeat stability", got_runs[0], got_runs[1])
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate streamed conv snake inference")
    parser.add_argument("--cols", type=int, default=1)
    parser.add_argument("--stream-depth", type=int, default=4)
    parser.add_argument("--conv-scale", type=float, default=0.25)
    parser.add_argument("--head-scale", type=float, default=0.25)
    args = parser.parse_args()

    print("═══ ConvSnake Streaming Inference Test ═══\n")
    ok = run_test(
        cols=args.cols,
        stream_depth=args.stream_depth,
        conv_scale=args.conv_scale,
        head_scale=args.head_scale,
    )
    print(f"\n{'═' * 50}")
    print("ALL PASSED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
