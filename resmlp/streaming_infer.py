"""Streaming inference service for the residual MLP NPU path."""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

from iron.common.aie_context import AIEContext

from resmlp import from_tiled, to_tiled
from resmlp.data_utils import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_SIZE,
    SUPPORTED_DATASETS,
    get_dataset_config,
    get_eval_dataset,
    resolve_dataset_name,
    save_prediction_preview,
)
from resmlp.model import ResMLP
from resmlp.design import ROWS_PER_COL
from resmlp.streaming_logits_design import N_CLS_PADDED, NUM_CLASSES
from resmlp.streaming_logits_op import StreamingResMLPLogits
from resmlp.streaming_op import StreamingResMLP


class ResidualStreamingInferenceService:
    """Host-facing infinite stream API over repeated multi-microbatch NPU chunks."""

    def __init__(
        self,
        checkpoint,
        *,
        hidden_dim=None,
        num_layers=None,
        dataset=None,
        batch_size=None,
        npu_head=True,
        num_cols=8,
        stream_depth=32,
    ):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.B = batch_size if batch_size is not None else ckpt.get("npu_batch_size", 8)
        self.num_cols = num_cols
        self.num_tiles = num_cols * ROWS_PER_COL
        self.stream_depth = stream_depth
        self.npu_head = npu_head
        self.dataset = resolve_dataset_name(dataset, ckpt.get("dataset"))
        dataset_cfg = get_dataset_config(self.dataset)
        self.hidden_dim = hidden_dim if hidden_dim is not None else ckpt.get("hidden_dim", 160)
        self.num_layers = num_layers if num_layers is not None else ckpt.get("num_layers", 32)
        self.input_dim = ckpt.get("input_dim", dataset_cfg["input_dim"])
        self.num_classes = ckpt.get("num_classes", dataset_cfg["num_classes"])
        self.residual_bias = bool(ckpt.get("residual_bias", False))
        self.pipeline = ckpt.get("pipeline", "hybrid")
        self.epoch = ckpt["epoch"]
        self.eval_split = ckpt.get("eval_split", "test")
        self.val_size = ckpt.get("val_size", DEFAULT_VAL_SIZE)
        self.split_seed = ckpt.get("split_seed", DEFAULT_SPLIT_SEED)
        self.saved_eval_acc = ckpt.get(
            f"{self.eval_split}_acc",
            ckpt.get("val_acc", ckpt.get("test_acc")),
        )

        if self.num_layers not in {self.num_tiles, self.num_tiles - 2}:
            raise ValueError(
                f"Checkpoint uses {self.num_layers} residual layers, but the {self.num_tiles}-tile "
                "streaming inference pipeline can only handle models with either all tiles "
                "used as residual layers or with 2 identity-padded endpoints."
            )
        if self.residual_bias:
            raise ValueError(
                "Streaming NPU residual inference does not support residual-bias checkpoints yet. "
                "Train/evaluate them on CPU/GPU first or add residual-bias support to the NPU kernels."
            )

        self.model = ResMLP(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            residual_bias=self.residual_bias,
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.embed_weight = self.model.embed.weight.detach().float()
        self.embed_bias = self.model.embed.bias.detach().float()
        residual_weights = self.model.export_residual_weights()

        if self.num_layers == self.num_tiles - 2:
            zero_w = np.zeros((self.hidden_dim, self.hidden_dim), dtype=bfloat16)
            residual_weights = [zero_w] + residual_weights + [zero_w]

        self.packed_weights_by_tile = np.stack(
            [np.asarray(to_tiled(w), dtype=bfloat16) for w in residual_weights]
        )
        self.cpu_head_weight = self.model.head.weight.detach().cpu().float().numpy()
        self.cpu_head_bias = self.model.head.bias.detach().cpu().float().numpy()
        self.zero_hidden = np.zeros((self.B, self.hidden_dim), dtype=bfloat16)

        ctx = AIEContext(use_runlist=False)
        if self.npu_head:
            if self.num_classes != NUM_CLASSES:
                raise ValueError(
                    f"NPU logits head currently supports {NUM_CLASSES} classes, "
                    f"but checkpoint uses {self.num_classes}; retry with --cpu-head"
                )
            packed_head_weight = np.asarray(
                to_tiled(self.model.export_head_weight(padded_classes=N_CLS_PADDED)),
                dtype=bfloat16,
            )
            packed_head_bias = np.zeros((N_CLS_PADDED,), dtype=bfloat16)
            packed_head_bias[:NUM_CLASSES] = self.cpu_head_bias.astype(bfloat16)
            self.npu_op = StreamingResMLPLogits(
                self.packed_weights_by_tile,
                packed_head_weight,
                packed_head_bias,
                H=self.hidden_dim,
                B=self.B,
                num_cols=self.num_cols,
                stream_depth=self.stream_depth,
                context=ctx,
            )
        else:
            self.npu_op = StreamingResMLP(
                self.packed_weights_by_tile,
                H=self.hidden_dim,
                B=self.B,
                num_cols=self.num_cols,
                stream_depth=self.stream_depth,
                context=ctx,
            )
        ctx.compile_all()
        ctx.prepare_runtime()

    def _embed_images(self, images):
        x_flat = images.view(self.B, -1).float()
        x_hidden = (x_flat @ self.embed_weight.T + self.embed_bias).numpy()
        return x_hidden.astype(bfloat16)

    def _pad_and_pack_hidden(self, hidden_batches):
        valid = len(hidden_batches)
        hidden_tiles = [to_tiled(batch.astype(bfloat16, copy=False)) for batch in hidden_batches]
        while len(hidden_tiles) < self.stream_depth:
            hidden_tiles.append(to_tiled(self.zero_hidden))
        return np.concatenate(hidden_tiles), valid

    def process_hidden_chunk(self, hidden_batches):
        if not hidden_batches:
            return [], 0.0

        packed_input, valid = self._pad_and_pack_hidden(hidden_batches)
        self.npu_op.write_buffer("input", packed_input)
        elapsed = self.npu_op.run_stream()

        outputs = []
        if self.npu_head:
            logits_flat = self.npu_op.read_buffer(
                "output",
                (self.stream_depth * self.B * N_CLS_PADDED,),
                copy=True,
            )
            for idx in range(valid):
                start = idx * self.B * N_CLS_PADDED
                stop = (idx + 1) * self.B * N_CLS_PADDED
                outputs.append(
                    logits_flat[start:stop]
                    .reshape(self.B, N_CLS_PADDED)[:, : self.num_classes]
                    .astype(np.float32)
                )
        else:
            hidden_flat = self.npu_op.read_buffer(
                "output",
                (self.stream_depth * self.B * self.hidden_dim,),
                copy=True,
            )
            for idx in range(valid):
                start = idx * self.B * self.hidden_dim
                stop = (idx + 1) * self.B * self.hidden_dim
                hidden = from_tiled(hidden_flat[start:stop], self.B, self.hidden_dim).astype(np.float32)
                outputs.append(hidden @ self.cpu_head_weight.T + self.cpu_head_bias)
        return outputs, elapsed

    def process_hidden_stream(self, hidden_batch_iter):
        """Yield logits forever (or until the iterator ends)."""
        while True:
            batch_group = []
            try:
                while len(batch_group) < self.stream_depth:
                    batch_group.append(next(hidden_batch_iter))
            except StopIteration:
                pass

            if not batch_group:
                return

            logits_group, _ = self.process_hidden_chunk(batch_group)
            for logits in logits_group:
                yield logits

    def process_image_stream(self, image_batch_iter):
        """Yield logits/preds for an arbitrary-length image batch iterator."""
        def hidden_iter():
            for images in image_batch_iter:
                yield self._embed_images(images)

        for logits in self.process_hidden_stream(iter(hidden_iter())):
            yield torch.from_numpy(logits)

    def benchmark(self, repeated_images, num_images, warmup_calls=4):
        if repeated_images.shape[0] != self.B:
            raise ValueError(
                f"Expected repeated_images batch dimension {self.B}, got {repeated_images.shape[0]}"
            )

        calls = math.ceil(num_images / (self.B * self.stream_depth))
        processed_images = calls * self.B * self.stream_depth
        repeated_images = repeated_images.contiguous()

        for _ in range(warmup_calls):
            hidden_batches = [self._embed_images(repeated_images) for _ in range(self.stream_depth)]
            self.process_hidden_chunk(hidden_batches)

        cpu_embed_s = 0.0
        kernel_s = 0.0
        t_wall0 = time.perf_counter()
        for _ in range(calls):
            t_cpu0 = time.perf_counter()
            hidden_batches = [self._embed_images(repeated_images) for _ in range(self.stream_depth)]
            cpu_embed_s += time.perf_counter() - t_cpu0
            _, elapsed = self.process_hidden_chunk(hidden_batches)
            kernel_s += elapsed
        wall_s = time.perf_counter() - t_wall0

        return {
            "num_images_requested": num_images,
            "num_images_processed": processed_images,
            "wall_s": wall_s,
            "cpu_embed_s": cpu_embed_s,
            "kernel_s": kernel_s,
            "wall_img_s": processed_images / wall_s,
            "kernel_img_s": processed_images / kernel_s,
        }


def success_exit_code(observed_acc, expected_acc, *, partial_run=False):
    if partial_run or expected_acc is None:
        return 0
    return 0 if observed_acc >= max(0.0, expected_acc - 0.10) else 1


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


def main():
    parser = argparse.ArgumentParser(description="Streaming dataset inference on NPU")
    parser.add_argument("checkpoint", help="Path to trained .pt checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cpu-head", action="store_true",
                        help="Use the body-only streamer and run the classifier head on the host")
    parser.add_argument("--num-cols", type=int, default=8)
    parser.add_argument(
        "--stream-depth",
        type=int,
        default=32,
        help="Microbatches per NPU call; validated up to 32 on the full H=160, 32-tile path",
    )
    parser.add_argument("--bench", action="store_true", help="Show timing")
    parser.add_argument("--bench-images", type=int, default=None,
                        help="Run a repeated-image throughput benchmark over this many images")
    parser.add_argument("--eval-split", choices=["val", "test"], default=None)
    parser.add_argument("--preview-samples", type=int, default=0)
    parser.add_argument("--preview-out", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    print(f"Loading {args.checkpoint}...")
    t0 = time.time()
    service = ResidualStreamingInferenceService(
        args.checkpoint,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dataset=args.dataset,
        batch_size=args.batch_size,
        npu_head=not args.cpu_head,
        num_cols=args.num_cols,
        stream_depth=args.stream_depth,
    )
    print(
        f"  dataset={service.dataset}, pipeline={service.pipeline}, epoch={service.epoch}, "
        f"saved {service.eval_split} acc={service.saved_eval_acc}"
    )
    print(
        f"Compiled streaming NPU pipeline ({service.num_tiles} tiles, B={service.B}, H={service.hidden_dim}, "
        f"stream_depth={service.stream_depth}, head={'npu' if service.npu_head else 'cpu'}) in {time.time() - t0:.1f}s"
    )

    eval_split = args.eval_split or service.eval_split
    eval_ds = get_eval_dataset(
        service.dataset,
        split=eval_split,
        data_dir=args.data_dir,
        val_size=service.val_size,
        split_seed=service.split_seed,
    )

    if args.bench_images is not None:
        repeated_images, _, _ = next(dataset_batch_iterator(eval_ds, service.B))
        stats = service.benchmark(repeated_images, args.bench_images)
        print("\nRepeated-image benchmark:")
        for key in (
            "num_images_requested",
            "num_images_processed",
            "wall_s",
            "cpu_embed_s",
            "kernel_s",
            "wall_img_s",
            "kernel_img_s",
        ):
            print(f"  {key}: {stats[key]}")
        return 0

    correct = 0
    total = 0
    npu_time = 0.0
    npu_calls = 0
    cpu_time = 0.0
    preview_images = []
    preview_labels = []
    preview_preds = []

    batch_iter = dataset_batch_iterator(eval_ds, service.B)
    consumed_batches = 0
    while True:
        batch_group = []
        try:
            while len(batch_group) < service.stream_depth:
                if args.max_batches is not None and consumed_batches >= args.max_batches:
                    break
                images, labels, actual_b = next(batch_iter)
                batch_group.append((images, labels, actual_b))
                consumed_batches += 1
        except StopIteration:
            pass

        if not batch_group:
            break

        hidden_batches = []
        batch_meta = []
        t_cpu0 = time.perf_counter()
        for images, labels, actual_b in batch_group:
            hidden_batches.append(service._embed_images(images))
            batch_meta.append((images, labels, actual_b))
        cpu_time += time.perf_counter() - t_cpu0

        outputs, elapsed = service.process_hidden_chunk(hidden_batches)
        npu_time += elapsed
        npu_calls += 1

        for logits_np, (images, labels, actual_b) in zip(outputs, batch_meta):
            logits = torch.from_numpy(logits_np)
            preds = logits[:actual_b].argmax(1)
            correct += (preds == labels).sum().item()
            total += actual_b
            remaining = args.preview_samples - len(preview_labels)
            if remaining > 0:
                take = min(remaining, actual_b)
                preview_images.append(images[:take].clone())
                preview_labels.extend(labels[:take].tolist())
                preview_preds.extend(preds[:take].tolist())

    accuracy = correct / total if total else 0.0
    print(f"\n{'═' * 50}")
    print(f"Streaming NPU {eval_split} accuracy: {accuracy:.4f} ({correct}/{total})")

    preview_out = args.preview_out
    if args.preview_samples > 0 and preview_out is None:
        preview_out = str(
            Path("build")
            / f"streaming_{service.dataset}_{eval_split}_preview_h{service.hidden_dim}_b{service.B}.png"
        )
    if preview_out is not None and preview_labels:
        preview_path = save_prediction_preview(
            torch.cat(preview_images, dim=0),
            preview_labels,
            preview_preds,
            preview_out,
            dataset_name=service.dataset,
            max_items=args.preview_samples,
        )
        print(f"Preview: {preview_path}")

    if args.bench and npu_calls:
        print("\nTiming:")
        print(f"  NPU total:   {npu_time * 1000:.1f} ms ({npu_time / npu_calls * 1000:.3f} ms/call)")
        print(f"  CPU total:   {cpu_time * 1000:.1f} ms")
        print(f"  NPU calls:   {npu_calls}")
        print(f"  Throughput:  {total / npu_time:.0f} images/sec (NPU only)")

    return success_exit_code(
        accuracy,
        service.saved_eval_acc if eval_split == service.eval_split else None,
        partial_run=args.max_batches is not None,
    )


if __name__ == "__main__":
    sys.exit(main())
