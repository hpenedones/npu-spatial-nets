"""Streaming inference service for the residual MLP NPU path."""

import argparse
import sys
import time

import numpy as np
import torch
from ml_dtypes import bfloat16
from torchvision import datasets, transforms

from iron.common.aie_context import AIEContext

from resmlp import from_tiled, to_tiled
from resmlp.model import ResMLP
from resmlp.design import ROWS_PER_COL
from resmlp.streaming_op import StreamingResMLP


class ResidualStreamingInferenceService:
    """Host-facing infinite stream API over repeated multi-microbatch NPU chunks."""

    def __init__(self, checkpoint, *, hidden_dim=None, num_layers=None, num_cols=8, stream_depth=6):
        self.B = 8
        self.num_cols = num_cols
        self.num_tiles = num_cols * ROWS_PER_COL
        self.stream_depth = stream_depth

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.hidden_dim = hidden_dim if hidden_dim is not None else ckpt.get("hidden_dim", 160)
        self.num_layers = num_layers if num_layers is not None else ckpt.get("num_layers", 32)
        self.pipeline = ckpt.get("pipeline", "hybrid")
        self.epoch = ckpt["epoch"]
        self.test_acc = ckpt.get("test_acc", None)

        if self.num_layers not in {self.num_tiles, self.num_tiles - 2}:
            raise ValueError(
                f"Checkpoint uses {self.num_layers} residual layers, but the {self.num_tiles}-tile "
                "streaming inference pipeline can only handle models with either all tiles "
                "used as residual layers or with 2 identity-padded endpoints."
            )

        self.model = ResMLP(hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.embed_weight = self.model.embed.weight.detach().float()
        self.embed_bias = self.model.embed.bias.detach().float()
        self.head_weight = self.model.head.weight.detach().float()
        self.head_bias = self.model.head.bias.detach().float()
        residual_weights = self.model.export_residual_weights()

        if self.num_layers == self.num_tiles - 2:
            zero_w = np.zeros((self.hidden_dim, self.hidden_dim), dtype=bfloat16)
            residual_weights = [zero_w] + residual_weights + [zero_w]

        self.packed_weights_by_tile = np.stack(
            [np.asarray(to_tiled(w), dtype=bfloat16) for w in residual_weights]
        )
        self.zero_hidden = np.zeros((self.B, self.hidden_dim), dtype=bfloat16)

        ctx = AIEContext(use_runlist=False)
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

    def process_hidden_stream(self, hidden_batch_iter):
        """Yield hidden outputs forever (or until the iterator ends)."""
        while True:
            batch_group = []
            try:
                while len(batch_group) < self.stream_depth:
                    batch_group.append(next(hidden_batch_iter))
            except StopIteration:
                pass

            if not batch_group:
                return

            outputs, _ = self.process_hidden_chunk(batch_group)
            for output in outputs:
                yield output

    def process_image_stream(self, image_batch_iter):
        """Yield logits/preds for an arbitrary-length image batch iterator."""
        def hidden_iter():
            for images in image_batch_iter:
                x_flat = images.view(self.B, -1).float()
                x_hidden = (x_flat @ self.embed_weight.T + self.embed_bias).numpy()
                yield x_hidden.astype(bfloat16)

        for hidden_out in self.process_hidden_stream(iter(hidden_iter())):
            logits = torch.from_numpy(hidden_out) @ self.head_weight.T + self.head_bias
            yield logits


def mnist_batch_iterator(dataset, batch_size):
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
    parser = argparse.ArgumentParser(description="Streaming MNIST inference on NPU")
    parser.add_argument("checkpoint", help="Path to trained .pt checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-cols", type=int, default=8)
    parser.add_argument(
        "--stream-depth",
        type=int,
        default=6,
        help="Microbatches per NPU call; validated up to 6 on the full H=160, 32-tile path",
    )
    parser.add_argument("--bench", action="store_true", help="Show timing")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading {args.checkpoint}...")
    t0 = time.time()
    service = ResidualStreamingInferenceService(
        args.checkpoint,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_cols=args.num_cols,
        stream_depth=args.stream_depth,
    )
    print(f"  pipeline={service.pipeline}, epoch={service.epoch}, test acc={service.test_acc}")
    print(
        f"Compiled streaming NPU pipeline ({service.num_tiles} tiles, H={service.hidden_dim}, "
        f"stream_depth={service.stream_depth}) in {time.time() - t0:.1f}s"
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)

    correct = 0
    total = 0
    npu_time = 0.0
    npu_calls = 0
    cpu_time = 0.0

    batch_iter = mnist_batch_iterator(test_ds, service.B)
    consumed_batches = 0
    while True:
        batch_group = []
        try:
            while len(batch_group) < service.stream_depth:
                if args.max_batches is not None and consumed_batches >= args.max_batches:
                    break
                batch_group.append(next(batch_iter))
                consumed_batches += 1
        except StopIteration:
            pass

        if not batch_group:
            break

        hidden_batches = []
        labels_meta = []
        t_cpu0 = time.perf_counter()
        for images, labels, actual_b in batch_group:
            x_flat = images.view(service.B, -1).float()
            x_hidden = (x_flat @ service.embed_weight.T + service.embed_bias).numpy()
            hidden_batches.append(x_hidden.astype(bfloat16))
            labels_meta.append((labels, actual_b))
        cpu_time += time.perf_counter() - t_cpu0

        outputs, elapsed = service.process_hidden_chunk(hidden_batches)
        npu_time += elapsed
        npu_calls += 1

        t_cpu0 = time.perf_counter()
        for hidden_out, (labels, actual_b) in zip(outputs, labels_meta):
            logits = torch.from_numpy(hidden_out) @ service.head_weight.T + service.head_bias
            preds = logits[:actual_b].argmax(1)
            correct += (preds == labels).sum().item()
            total += actual_b
        cpu_time += time.perf_counter() - t_cpu0

    accuracy = correct / total if total else 0.0
    print(f"\n{'═' * 50}")
    print(f"Streaming NPU accuracy: {accuracy:.4f} ({correct}/{total})")

    if args.bench and npu_calls:
        print("\nTiming:")
        print(f"  NPU total:   {npu_time * 1000:.1f} ms ({npu_time / npu_calls * 1000:.3f} ms/call)")
        print(f"  CPU total:   {cpu_time * 1000:.1f} ms")
        print(f"  NPU calls:   {npu_calls}")
        print(f"  Throughput:  {total / npu_time:.0f} images/sec (NPU only)")

    return 0 if accuracy > 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
