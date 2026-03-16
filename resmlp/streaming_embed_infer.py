"""Streaming inference service for embed-on-NPU residual inference."""

import argparse
import math
import sys
import time

import numpy as np
import torch
from ml_dtypes import bfloat16
from torchvision import datasets, transforms

from iron.common.aie_context import AIEContext

from resmlp import from_tiled, to_tiled
from resmlp.design import ROWS_PER_COL
from resmlp.model import ResMLP
from resmlp.streaming_embed_design import EMBED_INPUT_DIM
from resmlp.streaming_embed_op import StreamingEmbedResMLP


def reference_embed_resmlp(x_raw, embed_weight, residual_weights):
    x = (x_raw.astype(np.float32) @ embed_weight.astype(np.float32)).astype(bfloat16).astype(np.float32)
    for w in residual_weights:
        matmul_out = (x @ w.astype(np.float32)).astype(bfloat16).astype(np.float32)
        relu_out = np.maximum(matmul_out, 0)
        x = (relu_out + x).astype(bfloat16).astype(np.float32)
    return x.astype(bfloat16)


class EmbedResidualStreamingInferenceService:
    """Streaming residual service with the first linear/embed layer on the NPU."""

    def __init__(
        self,
        checkpoint=None,
        *,
        hidden_dim=32,
        num_cols=8,
        stream_depth=32,
    ):
        self.B = 8
        self.num_cols = num_cols
        self.num_tiles = num_cols * ROWS_PER_COL
        self.num_layers = self.num_tiles - 1
        self.stream_depth = stream_depth
        self.hidden_dim = hidden_dim

        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            hidden_dim = ckpt.get("hidden_dim", hidden_dim)
            num_layers = ckpt.get("num_layers", self.num_layers)
            residual_bias = bool(ckpt.get("residual_bias", False))
            if hidden_dim != self.hidden_dim or num_layers != self.num_layers:
                raise ValueError(
                    f"Checkpoint/model mismatch: expected hidden_dim={self.hidden_dim}, "
                    f"num_layers={self.num_layers}, got hidden_dim={hidden_dim}, num_layers={num_layers}"
                )
            if residual_bias:
                raise ValueError(
                    "The embed-on-NPU residual prototype does not support residual-bias checkpoints yet."
                )
            self.pipeline = ckpt.get("pipeline", "hybrid")
            self.epoch = ckpt.get("epoch")
            self.test_acc = ckpt.get("test_acc")
            self.model = ResMLP(hidden_dim=self.hidden_dim, num_layers=self.num_layers)
            self.model.load_state_dict(ckpt["model"])
        else:
            self.pipeline = "random-init"
            self.epoch = None
            self.test_acc = None
            self.model = ResMLP(hidden_dim=self.hidden_dim, num_layers=self.num_layers)
            self.model.zero_linear_biases()

        self.model.eval()
        self.embed_weight = self.model.export_embed_weight()
        self.residual_weights = self.model.export_residual_weights()
        self.head_weight = self.model.head.weight.detach().float()
        self.head_bias = self.model.head.bias.detach().float()
        self.zero_x = np.zeros((self.B, EMBED_INPUT_DIM), dtype=bfloat16)

        packed_embed = np.asarray(to_tiled(self.embed_weight), dtype=bfloat16)
        packed_residual = np.stack(
            [np.asarray(to_tiled(w), dtype=bfloat16) for w in self.residual_weights]
        )

        ctx = AIEContext(use_runlist=False)
        self.npu_op = StreamingEmbedResMLP(
            packed_embed,
            packed_residual,
            H=self.hidden_dim,
            B=self.B,
            num_cols=self.num_cols,
            stream_depth=self.stream_depth,
            context=ctx,
        )
        ctx.compile_all()
        ctx.prepare_runtime()

    def _pad_and_pack_raw(self, raw_batches):
        valid = len(raw_batches)
        raw_tiles = [to_tiled(batch.astype(bfloat16, copy=False)) for batch in raw_batches]
        while len(raw_tiles) < self.stream_depth:
            raw_tiles.append(to_tiled(self.zero_x))
        return np.concatenate(raw_tiles), valid

    def process_raw_chunk(self, raw_batches):
        if not raw_batches:
            return [], 0.0

        packed_input, valid = self._pad_and_pack_raw(raw_batches)
        self.npu_op.write_buffer("input", packed_input)
        elapsed = self.npu_op.run_stream()

        y_flat = self.npu_op.read_buffer(
            "output",
            (self.stream_depth * self.B * self.hidden_dim,),
            copy=True,
        )
        outputs = []
        for idx in range(valid):
            start = idx * self.B * self.hidden_dim
            stop = (idx + 1) * self.B * self.hidden_dim
            outputs.append(from_tiled(y_flat[start:stop], self.B, self.hidden_dim).astype(np.float32))
        return outputs, elapsed

    def benchmark(self, num_images):
        zero_group = [self.zero_x for _ in range(self.stream_depth)]
        calls = math.ceil(num_images / (self.B * self.stream_depth))

        kernel_s = 0.0
        t0 = time.perf_counter()
        for _ in range(calls):
            _, elapsed = self.process_raw_chunk(zero_group)
            kernel_s += elapsed
        wall_s = time.perf_counter() - t0
        return {
            "num_images": num_images,
            "wall_s": wall_s,
            "kernel_s": kernel_s,
            "wall_img_s": num_images / wall_s,
            "kernel_img_s": num_images / kernel_s,
        }


def mnist_raw_batch_iterator(dataset, batch_size):
    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        actual_b = end - start
        images = torch.stack([dataset[i][0] for i in range(start, end)])
        labels = torch.tensor([dataset[i][1] for i in range(start, end)])
        x_raw = images.view(actual_b, -1).float().numpy().astype(bfloat16)
        if actual_b < batch_size:
            pad = np.zeros((batch_size - actual_b, EMBED_INPUT_DIM), dtype=bfloat16)
            x_raw = np.concatenate([x_raw, pad], axis=0)
        yield x_raw, labels, actual_b


def main():
    parser = argparse.ArgumentParser(description="Streaming residual inference with embed on NPU")
    parser.add_argument("checkpoint", nargs="?", default=None, help="Optional trained .pt checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-cols", type=int, default=8)
    parser.add_argument("--stream-depth", type=int, default=32)
    parser.add_argument("--bench-images", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    print("Building streaming embed+residual pipeline...")
    t0 = time.time()
    service = EmbedResidualStreamingInferenceService(
        args.checkpoint,
        hidden_dim=args.hidden_dim,
        num_cols=args.num_cols,
        stream_depth=args.stream_depth,
    )
    print(f"  pipeline={service.pipeline}, epoch={service.epoch}, test_acc={service.test_acc}")
    print(
        f"Compiled streaming NPU pipeline ({service.num_tiles} tiles, H={service.hidden_dim}, "
        f"num_layers={service.num_layers}, stream_depth={service.stream_depth}) in {time.time() - t0:.1f}s"
    )

    if args.bench_images is not None:
        stats = service.benchmark(args.bench_images)
        for key in ("num_images", "wall_s", "kernel_s", "wall_img_s", "kernel_img_s"):
            print(f"{key}={stats[key]}")
        return 0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)

    correct = 0
    total = 0
    kernel_s = 0.0
    calls = 0

    batch_iter = mnist_raw_batch_iterator(test_ds, service.B)
    consumed_batches = 0
    while True:
        batch_group = []
        labels_meta = []
        try:
            while len(batch_group) < service.stream_depth:
                if args.max_batches is not None and consumed_batches >= args.max_batches:
                    break
                x_raw, labels, actual_b = next(batch_iter)
                batch_group.append(x_raw)
                labels_meta.append((labels, actual_b))
                consumed_batches += 1
        except StopIteration:
            pass

        if not batch_group:
            break

        outputs, elapsed = service.process_raw_chunk(batch_group)
        kernel_s += elapsed
        calls += 1

        for hidden_out, (labels, actual_b) in zip(outputs, labels_meta):
            logits = torch.from_numpy(hidden_out) @ service.head_weight.T + service.head_bias
            preds = logits[:actual_b].argmax(1)
            correct += (preds == labels).sum().item()
            total += actual_b

    accuracy = correct / total if total else 0.0
    print(f"\n{'═' * 50}")
    print(f"Streaming embed+residual accuracy: {accuracy:.4f} ({correct}/{total})")
    if calls:
        print(f"Throughput: {total / kernel_s:.0f} images/sec (NPU kernel only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
