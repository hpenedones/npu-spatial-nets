"""Streaming inference and benchmark CLI for the convolutional snake."""

import argparse
import math
import sys
import time

import numpy as np
import torch

from iron.common.aie_context import AIEContext

from convsnake.config import BATCH_SIZE, SUPPORTED_DATASETS, build_config, num_blocks_for_cols, total_flops_per_image
from convsnake.model import StreamingConvNet
from convsnake.reference import forward_reference_logits, pack_image_batches, quantized_model_copy
from convsnake.streaming_op import StreamingConvSnake
from resmlp.data_utils import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_SIZE,
    get_eval_dataset,
    resolve_dataset_name,
)


class ConvSnakeStreamingInferenceService:
    def __init__(
        self,
        checkpoint=None,
        *,
        dataset=None,
        num_cols=8,
        stream_depth=32,
        conv_scale=0.25,
        head_scale=0.25,
    ):
        self.num_cols = num_cols
        self.stream_depth = stream_depth
        self.num_blocks = num_blocks_for_cols(num_cols)

        self.pipeline = "random-init"
        self.epoch = None
        self.test_acc = None
        self.eval_split = None
        self.val_size = DEFAULT_VAL_SIZE
        self.split_seed = DEFAULT_SPLIT_SEED
        self.residual_blocks = True

        ckpt = None
        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.pipeline = ckpt.get("pipeline", "convsnake")
            self.epoch = ckpt.get("epoch")
            self.test_acc = ckpt.get("test_acc") or ckpt.get("val_acc")
            self.eval_split = ckpt.get("eval_split")
            self.val_size = ckpt.get("val_size", DEFAULT_VAL_SIZE)
            self.split_seed = ckpt.get("split_seed", DEFAULT_SPLIT_SEED)
            self.residual_blocks = bool(ckpt.get("residual_blocks", True))

        self.dataset = resolve_dataset_name(dataset, ckpt.get("dataset") if ckpt else None)
        self.config = build_config(self.dataset)
        self.B = self.config.batch_size

        if ckpt is not None:
            checkpoint_blocks = ckpt.get("num_blocks", self.num_blocks)
            if checkpoint_blocks != self.num_blocks:
                raise ValueError(
                    f"Checkpoint expects {checkpoint_blocks} conv blocks, but --num-cols={num_cols} "
                    f"maps to {self.num_blocks}"
                )

        model = StreamingConvNet(num_same_blocks=self.num_blocks, config=self.config)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
        else:
            model.scale_initial_weights(conv_scale=conv_scale, head_scale=head_scale)

        model.eval()
        self.model = quantized_model_copy(model)
        self.zero_group = torch.zeros(
            self.stream_depth,
            self.B,
            self.config.img_c,
            self.config.img_h,
            self.config.img_w,
        )

        ctx = AIEContext(use_runlist=False)
        self.npu_op = StreamingConvSnake(
            self.model.export_npu_weights(),
            config=self.config,
            B=self.B,
            num_cols=self.num_cols,
            stream_depth=self.stream_depth,
            context=ctx,
        )
        ctx.compile_all()
        ctx.prepare_runtime()

    def _pad_and_pack_images(self, image_batches):
        valid = len(image_batches)
        group = torch.stack(image_batches)
        if valid < self.stream_depth:
            pad = torch.zeros(
                self.stream_depth - valid,
                self.B,
                self.config.img_c,
                self.config.img_h,
                self.config.img_w,
                dtype=group.dtype,
            )
            group = torch.cat([group, pad], dim=0)
        return pack_image_batches(group), valid

    def process_image_chunk(self, image_batches):
        if not image_batches:
            return [], 0.0

        packed_input, valid = self._pad_and_pack_images(image_batches)
        self.npu_op.write_buffer("input", packed_input)
        kernel_s = self.npu_op.run_stream()
        logits_flat = self.npu_op.read_buffer(
            "output",
            (self.stream_depth * self.config.logits_elems,),
            copy=True,
        )
        logits = np.asarray(logits_flat, dtype=np.float32).reshape(
            self.stream_depth,
            self.B,
            self.config.num_classes,
        )
        return [logits[idx] for idx in range(valid)], kernel_s

    def benchmark(self, num_images: int, warmup_calls: int = 4):
        packed_zero = pack_image_batches(self.zero_group)
        self.npu_op.write_buffer("input", packed_zero)
        for _ in range(warmup_calls):
            self.npu_op.run_stream()

        batches = math.ceil(num_images / self.B)
        calls = math.ceil(batches / self.stream_depth)
        kernel_s = 0.0
        t0 = time.perf_counter()
        for _ in range(calls):
            self.npu_op.write_buffer("input", packed_zero)
            kernel_s += self.npu_op.run_stream()
        wall_s = time.perf_counter() - t0

        per_image_flops = total_flops_per_image(self.num_cols, config=self.config)
        wall_img_s = num_images / wall_s
        kernel_img_s = num_images / kernel_s
        return {
            "dataset": self.dataset,
            "num_images": num_images,
            "wall_s": wall_s,
            "kernel_s": kernel_s,
            "wall_img_s": wall_img_s,
            "kernel_img_s": kernel_img_s,
            "wall_tflops": wall_img_s * per_image_flops / 1e12,
            "kernel_tflops": kernel_img_s * per_image_flops / 1e12,
            "per_image_flops": per_image_flops,
            "calls": calls,
        }


def dataset_batch_iterator(dataset, batch_size):
    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        actual_b = end - start
        images = torch.stack([dataset[i][0] for i in range(start, end)])
        labels = torch.tensor([dataset[i][1] for i in range(start, end)])
        if actual_b < batch_size:
            pad = torch.zeros(batch_size - actual_b, *images.shape[1:])
            images = torch.cat([images, pad])
        yield images, labels, actual_b


def prediction_match_with_margin(ref_logits: np.ndarray, got_logits: np.ndarray, *, actual_b: int, margin_tol: float = 1e-3) -> bool:
    ref = ref_logits[:actual_b]
    got = got_logits[:actual_b]
    ref_pred = ref.argmax(1)
    got_pred = got.argmax(1)
    mismatches = ref_pred != got_pred
    if not mismatches.any():
        return True

    def top_margin(logits: np.ndarray) -> np.ndarray:
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    ref_margin = top_margin(ref)
    got_margin = top_margin(got)
    severe = mismatches & (ref_margin > margin_tol) & (got_margin > margin_tol)
    return not severe.any()


def main():
    parser = argparse.ArgumentParser(description="Streaming conv-snake inference on NPU")
    parser.add_argument("checkpoint", nargs="?", default=None, help="Optional trained .pt checkpoint")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=None)
    parser.add_argument("--num-cols", type=int, default=8)
    parser.add_argument("--stream-depth", type=int, default=32)
    parser.add_argument("--bench-images", type=int, default=None, help="Run a synthetic throughput benchmark")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--verify-batches", type=int, default=0)
    parser.add_argument("--eval-split", choices=("val", "test"), default=None)
    parser.add_argument("--conv-scale", type=float, default=0.25)
    parser.add_argument("--head-scale", type=float, default=0.25)
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    if args.checkpoint is None and args.bench_images is None:
        raise ValueError("either a checkpoint or --bench-images is required")

    print("Building streaming conv snake...")
    t0 = time.time()
    service = ConvSnakeStreamingInferenceService(
        args.checkpoint,
        dataset=args.dataset,
        num_cols=args.num_cols,
        stream_depth=args.stream_depth,
        conv_scale=args.conv_scale,
        head_scale=args.head_scale,
    )
    print(
        f"  dataset={service.dataset}, pipeline={service.pipeline}, "
        f"epoch={service.epoch}, acc={service.test_acc}"
    )
    print(
        f"Compiled streaming NPU pipeline ({service.num_cols * 4} tiles, "
        f"blocks={service.num_blocks}, stream_depth={service.stream_depth}) in {time.time() - t0:.1f}s"
    )

    if args.bench_images is not None:
        stats = service.benchmark(args.bench_images)
        for key in (
            "dataset",
            "num_images",
            "wall_s",
            "kernel_s",
            "wall_img_s",
            "kernel_img_s",
            "wall_tflops",
            "kernel_tflops",
        ):
            print(f"{key}={stats[key]}")
        return 0

    eval_split = args.eval_split or service.eval_split or "test"
    eval_ds = get_eval_dataset(
        service.dataset,
        split=eval_split,
        data_dir=args.data_dir,
        val_size=service.val_size,
        split_seed=service.split_seed,
    )

    correct = 0
    total = 0
    kernel_s = 0.0
    calls = 0
    verify_close = []
    verify_max_abs = 0.0
    verify_pred_match = True

    batch_iter = dataset_batch_iterator(eval_ds, service.B)
    consumed_batches = 0
    while True:
        batch_group = []
        labels_meta = []
        try:
            while len(batch_group) < service.stream_depth:
                if args.max_batches is not None and consumed_batches >= args.max_batches:
                    break
                images, labels, actual_b = next(batch_iter)
                batch_group.append(images)
                labels_meta.append((labels, actual_b))
                consumed_batches += 1
        except StopIteration:
            pass

        if not batch_group:
            break

        outputs, elapsed = service.process_image_chunk(batch_group)
        kernel_s += elapsed
        calls += 1

        for logits, (labels, actual_b), images in zip(outputs, labels_meta, batch_group):
            preds = logits[:actual_b].argmax(1)
            correct += int((preds == labels.numpy()).sum())
            total += actual_b

            if len(verify_close) < args.verify_batches:
                ref_logits = (
                    forward_reference_logits(service.model, images).detach().cpu().numpy().astype(np.float32)
                )
                verify_close.append(np.isclose(ref_logits, logits, rtol=0.05, atol=0.05).mean() * 100.0)
                verify_max_abs = max(
                    verify_max_abs,
                    float(np.max(np.abs(ref_logits - logits))),
                )
                verify_pred_match &= prediction_match_with_margin(
                    ref_logits,
                    logits,
                    actual_b=actual_b,
                )

    accuracy = correct / total if total else 0.0
    print(f"\n{eval_split} accuracy: {accuracy:.4f} ({correct}/{total})")
    if calls:
        print(f"Kernel time: {kernel_s:.3f}s over {calls} streamed calls")
    if verify_close:
        avg_close = sum(verify_close) / len(verify_close)
        print(
            f"Verification: {avg_close:.1f}% close, "
            f"max abs diff={verify_max_abs:.4f}, pred match={verify_pred_match}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
