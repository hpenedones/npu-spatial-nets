"""Evaluate and benchmark the forward-only HIGGS conveyor-belt path on the NPU."""

import argparse
import math
import sys
import time

import numpy as np
import torch
from ml_dtypes import bfloat16

from resmlp.xrt_env import ensure_xrt_python_path

ensure_xrt_python_path()

from iron.common.aie_context import AIEContext
from iron.common.aie_device_manager import pyxrt

from resmlp import TILE_BLOCK, from_tiled, to_tiled
from resmlp.data_utils import DEFAULT_SPLIT_SEED, DEFAULT_VAL_SIZE, get_eval_dataset
from resmlp.design import ROWS_PER_COL
from resmlp.model import ResMLP
from resmlp.streaming_op import StreamingResMLP


class HiggsStreamingInferenceService:
    """Host-facing streaming service for the HIGGS residual body on the NPU."""

    def __init__(
        self,
        checkpoint,
        *,
        hidden_dim=None,
        num_layers=None,
        batch_size=8,
        num_cols=None,
        stream_depth=32,
    ):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        dataset = ckpt.get("dataset", "higgs")
        if dataset.lower() != "higgs":
            raise ValueError(f"This curated branch only supports HIGGS checkpoints, got '{dataset}'")

        self.dataset = "higgs"
        self.B = batch_size
        self.hidden_dim = hidden_dim if hidden_dim is not None else ckpt.get("hidden_dim", 64)
        self.num_layers = num_layers if num_layers is not None else ckpt.get("num_layers", 32)
        self.num_cols = num_cols if num_cols is not None else math.ceil(self.num_layers / ROWS_PER_COL)
        self.num_tiles = self.num_cols * ROWS_PER_COL
        self.stream_depth = stream_depth
        self.input_dim = ckpt.get("input_dim", 28)
        self.num_classes = ckpt.get("num_classes", 2)
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

        if self.B % 8 != 0:
            raise ValueError(f"batch_size must be divisible by 8, got {self.B}")
        if self.hidden_dim % 8 != 0:
            raise ValueError(f"hidden_dim must be divisible by 8, got {self.hidden_dim}")
        if self.num_layers != self.num_tiles:
            raise ValueError(
                f"Checkpoint uses {self.num_layers} residual layers, but {self.num_cols} columns provide "
                f"{self.num_tiles} tiles. Choose num_cols so they match exactly."
            )
        if self.residual_bias:
            raise ValueError("Streaming residual inference does not support residual-bias checkpoints.")
        if self.num_classes != 2:
            raise ValueError(
                f"This curated branch expects binary HIGGS checkpoints, got {self.num_classes} classes"
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

        self.embed_weight_np = self.model.embed.weight.detach().cpu().float().numpy()
        self.embed_bias_np = self.model.embed.bias.detach().cpu().float().numpy()
        residual_weights = self.model.export_residual_weights()
        self.packed_weights_by_tile = np.stack(
            [np.asarray(to_tiled(weight), dtype=bfloat16) for weight in residual_weights]
        )
        self.cpu_head_weight = self.model.head.weight.detach().cpu().float().numpy()
        self.cpu_head_bias = self.model.head.bias.detach().cpu().float().numpy()

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

    def _embed_stacked(self, features_SBD):
        """(S, B, D) float32 → (S, B, H) bf16 via a single (S*B, D)·(D, H) matmul."""
        S = self.stream_depth
        flat = features_SBD.reshape(S * self.B, self.input_dim)
        hidden = flat @ self.embed_weight_np.T + self.embed_bias_np
        return hidden.astype(bfloat16).reshape(S, self.B, self.hidden_dim)

    def _pack_stacked(self, hidden_SBH):
        """(S, B, H) bf16 → flat tiled buffer, shape (S*B*H,). Single reshape+transpose."""
        br = bc = TILE_BLOCK
        S, B, H = hidden_SBH.shape
        return (
            hidden_SBH.reshape(S, B // br, br, H // bc, bc)
            .transpose(0, 1, 3, 2, 4)
            .reshape(-1)
        )

    def _detile_stacked(self, flat_SBH):
        """(S*B*H,) tiled bf16 → (S, B, H) float32. Single reshape+transpose."""
        br = bc = TILE_BLOCK
        S, B, H = self.stream_depth, self.B, self.hidden_dim
        return (
            flat_SBH.reshape(S, B // br, H // bc, br, bc)
            .transpose(0, 1, 3, 2, 4)
            .reshape(S, B, H)
            .astype(np.float32)
        )

    def _head_stacked(self, hidden_SBH_f32):
        """(S, B, H) float32 hidden → (S, B, C) float32 logits in one matmul."""
        S, B, H = hidden_SBH_f32.shape
        flat = hidden_SBH_f32.reshape(S * B, H)
        logits = flat @ self.cpu_head_weight.T + self.cpu_head_bias
        return logits.reshape(S, B, self.num_classes)

    def process_features_stacked(self, features_SBD, valid=None, timings=None):
        """Run the full chunk path (embed + kernel + head) on stacked features.

        features_SBD: np.ndarray, shape (stream_depth, B, input_dim), float32.
        valid: number of leading slots that carry real data (others are padding).
               If None, assumes full depth.
        Returns (logits_list[valid entries of (B, C) float32], kernel_elapsed_s).
        """
        if valid is None:
            valid = self.stream_depth

        embed_t0 = time.perf_counter()
        hidden_SBH = self._embed_stacked(features_SBD)
        embed_s = time.perf_counter() - embed_t0

        pack_t0 = time.perf_counter()
        packed_input = self._pack_stacked(hidden_SBH)
        pack_s = time.perf_counter() - pack_t0

        write_t0 = time.perf_counter()
        self.npu_op.write_input_slot(0, packed_input)
        write_s = time.perf_counter() - write_t0

        elapsed = self.npu_op.run_stream(timings=timings, slot=0)

        read_t0 = time.perf_counter()
        hidden_flat = self.npu_op.read_output_slot(
            0,
            (self.stream_depth * self.B * self.hidden_dim,),
            copy=True,
        )
        read_s = time.perf_counter() - read_t0

        detile_t0 = time.perf_counter()
        hidden_out_SBH = self._detile_stacked(hidden_flat)
        detile_s = time.perf_counter() - detile_t0

        head_t0 = time.perf_counter()
        logits_SBC = self._head_stacked(hidden_out_SBH)
        head_s = time.perf_counter() - head_t0

        outputs = [logits_SBC[i] for i in range(valid)]

        if timings is not None:
            timings["cpu_embed_s"] = timings.get("cpu_embed_s", 0.0) + embed_s
            timings["pack_s"] = timings.get("pack_s", 0.0) + pack_s
            timings["write_buffer_s"] = timings.get("write_buffer_s", 0.0) + write_s
            timings["read_buffer_s"] = timings.get("read_buffer_s", 0.0) + read_s
            timings["detile_s"] = timings.get("detile_s", 0.0) + detile_s
            timings["head_s"] = timings.get("head_s", 0.0) + head_s
        return outputs, elapsed

    def _prepare_slot_host(self, slot, features_SBD, timings):
        """Run the pre-dispatch host work for one slot: embed, pack, write_buffer, H2D sync."""
        embed_t0 = time.perf_counter()
        hidden_SBH = self._embed_stacked(features_SBD)
        if timings is not None:
            timings["cpu_embed_s"] = timings.get("cpu_embed_s", 0.0) + (
                time.perf_counter() - embed_t0
            )

        pack_t0 = time.perf_counter()
        packed = self._pack_stacked(hidden_SBH)
        if timings is not None:
            timings["pack_s"] = timings.get("pack_s", 0.0) + (
                time.perf_counter() - pack_t0
            )

        write_t0 = time.perf_counter()
        self.npu_op.write_input_slot(slot, packed)
        if timings is not None:
            timings["write_buffer_s"] = timings.get("write_buffer_s", 0.0) + (
                time.perf_counter() - write_t0
            )

        h2d_t0 = time.perf_counter()
        self.npu_op.sync_input_h2d(slot)
        if timings is not None:
            timings["h2d_sync_s"] = timings.get("h2d_sync_s", 0.0) + (
                time.perf_counter() - h2d_t0
            )

    def _consume_slot_host(self, slot, timings):
        """Run post-wait host work for one slot: D2H sync, read, detile, head."""
        d2h_t0 = time.perf_counter()
        self.npu_op.sync_output_d2h(slot)
        if timings is not None:
            timings["d2h_sync_s"] = timings.get("d2h_sync_s", 0.0) + (
                time.perf_counter() - d2h_t0
            )

        read_t0 = time.perf_counter()
        hidden_flat = self.npu_op.read_output_slot(
            slot,
            (self.stream_depth * self.B * self.hidden_dim,),
            copy=True,
        )
        if timings is not None:
            timings["read_buffer_s"] = timings.get("read_buffer_s", 0.0) + (
                time.perf_counter() - read_t0
            )

        detile_t0 = time.perf_counter()
        hidden_SBH = self._detile_stacked(hidden_flat)
        if timings is not None:
            timings["detile_s"] = timings.get("detile_s", 0.0) + (
                time.perf_counter() - detile_t0
            )

        head_t0 = time.perf_counter()
        logits_SBC = self._head_stacked(hidden_SBH)
        if timings is not None:
            timings["head_s"] = timings.get("head_s", 0.0) + (
                time.perf_counter() - head_t0
            )
        return logits_SBC

    def benchmark(self, repeated_features, num_samples, warmup_calls=4, async_pipe=True):
        if repeated_features.shape[0] != self.B:
            raise ValueError(
                f"Expected repeated_features batch dimension {self.B}, got {repeated_features.shape[0]}"
            )

        calls = math.ceil(num_samples / (self.B * self.stream_depth))
        processed_samples = calls * self.B * self.stream_depth

        base_BD = repeated_features.contiguous().view(self.B, -1).float().cpu().numpy()
        stacked_SBD = np.ascontiguousarray(
            np.broadcast_to(base_BD, (self.stream_depth, self.B, self.input_dim))
        )

        for _ in range(warmup_calls):
            self.process_features_stacked(stacked_SBD)

        kernel_s = 0.0
        timings = {}
        if async_pipe and calls >= 2:
            # Double-buffered pipeline: one slot is dispatched to the NPU while
            # the host prepares the next slot's inputs and/or consumes the previous
            # slot's outputs. In steady state: wall = max(host, kernel).
            wall_t0 = time.perf_counter()
            # Prime the pipeline with slot 0.
            self._prepare_slot_host(0, stacked_SBD, timings)
            k0_start = time.perf_counter()
            run_prev = self.npu_op.dispatch(0)
            slot_prev = 0
            # Prepare slot 1 while slot 0 runs.
            self._prepare_slot_host(1, stacked_SBD, timings)
            for i in range(1, calls):
                slot_next = i % self.NPU_NUM_SLOTS
                # Wait for the currently-running dispatch, measure kernel_s only by wait time.
                result = run_prev.wait()
                kernel_s += time.perf_counter() - k0_start
                if result != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    raise RuntimeError(
                        f"Kernel resmlp_streaming did not complete correctly: {result}"
                    )
                # Immediately dispatch the next slot (already prepared).
                k0_start = time.perf_counter()
                run_prev = self.npu_op.dispatch(slot_next)
                # Consume the just-finished slot (overlapped with new kernel).
                self._consume_slot_host(slot_prev, timings)
                # Prepare the slot after next (overlapped with new kernel too).
                if i + 1 < calls:
                    self._prepare_slot_host(slot_prev, stacked_SBD, timings)
                slot_prev = slot_next
            # Drain: wait for the final dispatch and consume it.
            result = run_prev.wait()
            kernel_s += time.perf_counter() - k0_start
            if result != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(
                    f"Kernel resmlp_streaming did not complete correctly: {result}"
                )
            self._consume_slot_host(slot_prev, timings)
            wall_s = time.perf_counter() - wall_t0
        else:
            wall_t0 = time.perf_counter()
            for _ in range(calls):
                _, elapsed = self.process_features_stacked(stacked_SBD, timings=timings)
                kernel_s += elapsed
            wall_s = time.perf_counter() - wall_t0

        result = {
            "num_samples_requested": num_samples,
            "num_samples_processed": processed_samples,
            "calls": calls,
            "wall_s": wall_s,
            "cpu_embed_s": timings.get("cpu_embed_s", 0.0),
            "kernel_s": kernel_s,
            "wall_sample_s": processed_samples / wall_s,
            "kernel_sample_s": processed_samples / kernel_s,
            "async_pipe": async_pipe,
        }
        for key in (
            "pack_s",
            "write_buffer_s",
            "h2d_sync_s",
            "d2h_sync_s",
            "read_buffer_s",
            "detile_s",
            "head_s",
        ):
            result[key] = timings.get(key, 0.0)
        return result

    NPU_NUM_SLOTS = 2


def success_exit_code(observed_acc, expected_acc, *, partial_run=False):
    if partial_run or expected_acc is None:
        return 0
    return 0 if observed_acc >= max(0.0, expected_acc - 0.10) else 1


def dataset_batch_iterator(dataset, batch_size):
    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        actual_b = end - start
        features = torch.stack([dataset[i][0] for i in range(start, end)])
        labels = torch.tensor([dataset[i][1] for i in range(start, end)])
        if actual_b < batch_size:
            pad = torch.zeros(batch_size - actual_b, *features.shape[1:])
            features = torch.cat([features, pad])
        yield features, labels, actual_b


def main():
    parser = argparse.ArgumentParser(description="Streaming HIGGS inference on the NPU")
    parser.add_argument("checkpoint", help="Path to a trained .pt checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-cols", type=int, default=None)
    parser.add_argument("--stream-depth", type=int, default=32)
    parser.add_argument("--bench", action="store_true", help="Show timing for evaluation runs")
    parser.add_argument("--bench-samples", type=int, default=None)
    parser.add_argument(
        "--no-async-pipe",
        action="store_true",
        help="Disable the double-buffered host/kernel overlap (for A/B comparisons)",
    )
    parser.add_argument("--eval-split", choices=["val", "test"], default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default="data/higgs_full")
    args = parser.parse_args()

    print(f"Loading {args.checkpoint}...")
    t0 = time.time()
    service = HiggsStreamingInferenceService(
        args.checkpoint,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        num_cols=args.num_cols,
        stream_depth=args.stream_depth,
    )
    print(
        f"  dataset={service.dataset}, pipeline={service.pipeline}, epoch={service.epoch}, "
        f"saved {service.eval_split} acc={service.saved_eval_acc}"
    )
    print(
        f"Compiled streaming NPU pipeline ({service.num_tiles} tiles, B={service.B}, H={service.hidden_dim}, "
        f"stream_depth={service.stream_depth}) in {time.time() - t0:.1f}s"
    )

    eval_split = args.eval_split or service.eval_split
    eval_ds = get_eval_dataset(
        service.dataset,
        split=eval_split,
        data_dir=args.data_dir,
        val_size=service.val_size,
        split_seed=service.split_seed,
    )

    if args.bench_samples is not None:
        repeated_features, _, _ = next(dataset_batch_iterator(eval_ds, service.B))
        stats = service.benchmark(
            repeated_features,
            args.bench_samples,
            async_pipe=not args.no_async_pipe,
        )
        print("\nRepeated-sample benchmark:")
        for key in (
            "num_samples_requested",
            "num_samples_processed",
            "calls",
            "wall_s",
            "cpu_embed_s",
            "kernel_s",
            "wall_sample_s",
            "kernel_sample_s",
        ):
            print(f"  {key}: {stats[key]}")

        calls = stats["calls"]
        print("\nPer-call breakdown (µs):")
        stage_keys = [
            ("cpu_embed_s", "cpu_embed"),
            ("pack_s", "pack(to_tiled)"),
            ("write_buffer_s", "write_buffer"),
            ("h2d_sync_s", "h2d_sync(DMA)"),
            ("kernel_s", "kernel(wait)"),
            ("d2h_sync_s", "d2h_sync(DMA)"),
            ("read_buffer_s", "read_buffer"),
            ("detile_s", "detile(from_tiled)"),
            ("head_s", "cpu_head"),
        ]
        accounted = 0.0
        for key, label in stage_keys:
            us_per_call = stats[key] / calls * 1e6
            print(f"  {label:>20}: {us_per_call:>10.2f} µs/call ({stats[key] * 1000:>9.2f} ms total)")
            accounted += stats[key]
        wall_us_per_call = stats["wall_s"] / calls * 1e6
        unaccounted_us = (stats["wall_s"] - accounted) / calls * 1e6
        print(f"  {'wall_total':>20}: {wall_us_per_call:>10.2f} µs/call")
        print(f"  {'unaccounted':>20}: {unaccounted_us:>10.2f} µs/call (loop/Python overhead)")
        return 0

    correct = 0
    total = 0
    npu_time = 0.0
    npu_calls = 0
    cpu_time = 0.0

    batch_iter = dataset_batch_iterator(eval_ds, service.B)
    consumed_batches = 0
    while True:
        batch_group = []
        try:
            while len(batch_group) < service.stream_depth:
                if args.max_batches is not None and consumed_batches >= args.max_batches:
                    break
                features, labels, actual_b = next(batch_iter)
                batch_group.append((features, labels, actual_b))
                consumed_batches += 1
        except StopIteration:
            pass

        if not batch_group:
            break

        valid = len(batch_group)
        cpu_t0 = time.perf_counter()
        features_SBD = np.zeros(
            (service.stream_depth, service.B, service.input_dim), dtype=np.float32
        )
        batch_meta = []
        for idx, (features, labels, actual_b) in enumerate(batch_group):
            features_SBD[idx] = features.contiguous().view(service.B, -1).float().cpu().numpy()
            batch_meta.append((labels, actual_b))
        cpu_time += time.perf_counter() - cpu_t0

        outputs, elapsed = service.process_features_stacked(features_SBD, valid=valid)
        npu_time += elapsed
        npu_calls += 1

        for logits_np, (labels, actual_b) in zip(outputs, batch_meta):
            logits = torch.from_numpy(logits_np)
            preds = logits[:actual_b].argmax(1)
            correct += (preds == labels).sum().item()
            total += actual_b

    accuracy = correct / total if total else 0.0
    print(f"\n{'═' * 50}")
    print(f"Streaming NPU {eval_split} accuracy: {accuracy:.4f} ({correct}/{total})")

    if args.bench and npu_calls:
        print("\nTiming:")
        print(f"  NPU total:    {npu_time * 1000:.1f} ms ({npu_time / npu_calls * 1000:.3f} ms/call)")
        print(f"  CPU total:    {cpu_time * 1000:.1f} ms")
        print(f"  NPU calls:    {npu_calls}")
        print(f"  Throughput:   {total / npu_time:.0f} samples/sec (NPU only)")

    return success_exit_code(
        accuracy,
        service.saved_eval_acc if eval_split == service.eval_split else None,
        partial_run=args.max_batches is not None,
    )


if __name__ == "__main__":
    sys.exit(main())
