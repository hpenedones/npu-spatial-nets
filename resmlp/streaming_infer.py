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

from resmlp import TILE_BLOCK, round_up_to_tile_multiple, to_tiled
from resmlp.data_utils import DEFAULT_SPLIT_SEED, DEFAULT_VAL_SIZE, get_eval_dataset
from resmlp.design import ROWS_PER_COL
from resmlp.model import ResMLP
from resmlp.streaming_op import StreamingResMLP


class HiggsStreamingInferenceService:
    """Host-facing streaming service for the full-NPU HIGGS pipeline."""

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
        self.hidden_dim = hidden_dim if hidden_dim is not None else ckpt.get("hidden_dim", 32)
        self.num_layers = num_layers if num_layers is not None else ckpt.get("num_layers", 30)
        required_tiles = self.num_layers + 2
        self.num_cols = num_cols if num_cols is not None else math.ceil(required_tiles / ROWS_PER_COL)
        self.num_tiles = self.num_cols * ROWS_PER_COL
        self.stream_depth = stream_depth
        self.input_dim = ckpt.get("input_dim", 28)
        self.num_classes = ckpt.get("num_classes", 2)
        self.residual_bias = bool(ckpt.get("residual_bias", False))
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
        if self.num_layers != self.num_tiles - 2:
            raise ValueError(
                f"Checkpoint uses {self.num_layers} residual layers, but {self.num_cols} columns "
                f"provide {self.num_tiles} tiles; expected residual layers == tiles - 2."
            )
        if self.residual_bias:
            raise ValueError("Streaming residual inference does not support residual-bias checkpoints.")
        if self.num_classes != 2:
            raise ValueError(
                f"This curated branch expects binary HIGGS checkpoints, got {self.num_classes} classes"
            )

        self.npu_input_dim = round_up_to_tile_multiple(self.input_dim)
        self.npu_output_dim = round_up_to_tile_multiple(self.num_classes)

        self.model = ResMLP(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            residual_bias=self.residual_bias,
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        residual_weights = self.model.export_residual_weights()
        self.packed_weights_by_tile = np.stack(
            [np.asarray(to_tiled(weight), dtype=bfloat16) for weight in residual_weights]
        )

        embed_weight = self.model.export_embed_weight(padded_input_dim=self.npu_input_dim)
        embed_bias = self.model.export_embed_bias()
        head_weight = self.model.export_head_weight(padded_classes=self.npu_output_dim)
        head_bias = self.model.export_head_bias(padded_classes=self.npu_output_dim)

        self.packed_embed_weight = np.asarray(to_tiled(embed_weight), dtype=bfloat16)
        self.packed_embed_bias = self._pack_bias_vector(embed_bias)
        self.packed_head_weight = np.asarray(to_tiled(head_weight), dtype=bfloat16)
        self.packed_head_bias = self._pack_bias_vector(head_bias)

        ctx = AIEContext(use_runlist=False)
        self.npu_op = StreamingResMLP(
            self.packed_weights_by_tile,
            H=self.hidden_dim,
            B=self.B,
            num_cols=self.num_cols,
            stream_depth=self.stream_depth,
            context=ctx,
            input_dim=self.input_dim,
            output_dim=self.num_classes,
            packed_embed_weight=self.packed_embed_weight,
            packed_embed_bias=self.packed_embed_bias,
            packed_head_weight=self.packed_head_weight,
            packed_head_bias=self.packed_head_bias,
        )
        ctx.compile_all()
        ctx.prepare_runtime()

    @staticmethod
    def _pack_bias_vector(bias):
        bias = np.asarray(bias, dtype=bfloat16)
        tiled = np.broadcast_to(bias, (TILE_BLOCK, bias.shape[0])).copy()
        return np.asarray(to_tiled(tiled), dtype=bfloat16)

    def _pad_features_stacked(self, features_SBD):
        """(S, B, D) float32 → (S, B, D_padded) bf16 for the full-NPU path."""
        padded = np.zeros((self.stream_depth, self.B, self.npu_input_dim), dtype=bfloat16)
        padded[..., : self.input_dim] = features_SBD.astype(bfloat16)
        return padded

    def _pack_stacked(self, tensor_SBK):
        """(S, B, K) bf16 → flat tiled buffer, shape (S*B*K,)."""
        br = bc = TILE_BLOCK
        S, B, K = tensor_SBK.shape
        return (
            tensor_SBK.reshape(S, B // br, br, K // bc, bc)
            .transpose(0, 1, 3, 2, 4)
            .reshape(-1)
        )

    def _detile_stacked(self, flat_SBK, width):
        """(S*B*K,) tiled bf16 → (S, B, K) float32."""
        br = bc = TILE_BLOCK
        S, B = self.stream_depth, self.B
        return (
            flat_SBK.reshape(S, B // br, width // bc, br, bc)
            .transpose(0, 1, 3, 2, 4)
            .reshape(S, B, width)
            .astype(np.float32)
        )

    @staticmethod
    def _add_timing(timings, key, elapsed):
        if timings is not None:
            timings[key] = timings.get(key, 0.0) + elapsed

    def _prepare_packed_input(self, features_SBD, timings=None):
        tensor_SBK = self._pad_features_stacked(features_SBD)

        pack_t0 = time.perf_counter()
        packed_input = self._pack_stacked(tensor_SBK)
        self._add_timing(timings, "pack_s", time.perf_counter() - pack_t0)
        return packed_input

    def _postprocess_output(self, flat_output, timings=None):
        detile_t0 = time.perf_counter()
        tensor_SBK = self._detile_stacked(flat_output, self.npu_output_dim)
        self._add_timing(timings, "detile_s", time.perf_counter() - detile_t0)

        return tensor_SBK[..., : self.num_classes].astype(np.float32)

    def process_features_stacked(self, features_SBD, valid=None, timings=None):
        """Run one stacked chunk through the configured NPU pipeline.

        features_SBD: np.ndarray, shape (stream_depth, B, input_dim), float32.
        valid: number of leading slots that carry real data (others are padding).
               If None, assumes full depth.
        Returns (logits_list[valid entries of (B, C) float32], kernel_elapsed_s).
        """
        if valid is None:
            valid = self.stream_depth

        packed_input = self._prepare_packed_input(features_SBD, timings=timings)

        write_t0 = time.perf_counter()
        self.npu_op.write_input_slot(0, packed_input)
        self._add_timing(timings, "write_buffer_s", time.perf_counter() - write_t0)

        elapsed = self.npu_op.run_stream(timings=timings, slot=0)

        read_t0 = time.perf_counter()
        output_flat = self.npu_op.read_output_slot(
            0,
            (self.stream_depth * self.B * self.npu_output_dim,),
            copy=True,
        )
        self._add_timing(timings, "read_buffer_s", time.perf_counter() - read_t0)

        logits_SBC = self._postprocess_output(output_flat, timings=timings)
        outputs = [logits_SBC[i] for i in range(valid)]
        return outputs, elapsed

    def _prepare_slot_host(self, slot, features_SBD, timings):
        """Run the pre-dispatch host work for one slot: embed, pack, write_buffer, H2D sync."""
        packed = self._prepare_packed_input(features_SBD, timings=timings)

        write_t0 = time.perf_counter()
        self.npu_op.write_input_slot(slot, packed)
        self._add_timing(timings, "write_buffer_s", time.perf_counter() - write_t0)

        h2d_t0 = time.perf_counter()
        self.npu_op.sync_input_h2d(slot)
        self._add_timing(timings, "h2d_sync_s", time.perf_counter() - h2d_t0)

    def _consume_slot_host(self, slot, timings):
        """Run post-wait host work for one slot: D2H sync, read, detile, postprocess."""
        d2h_t0 = time.perf_counter()
        self.npu_op.sync_output_d2h(slot)
        self._add_timing(timings, "d2h_sync_s", time.perf_counter() - d2h_t0)

        read_t0 = time.perf_counter()
        output_flat = self.npu_op.read_output_slot(
            slot,
            (self.stream_depth * self.B * self.npu_output_dim,),
            copy=True,
        )
        self._add_timing(timings, "read_buffer_s", time.perf_counter() - read_t0)
        return self._postprocess_output(output_flat, timings=timings)

    def _measure_kernel_wait(self, features_SBD, calls):
        """Measure pure dispatch+wait time without overlapped host work."""
        packed_input = self._prepare_packed_input(features_SBD)
        self.npu_op.write_input_slot(0, packed_input)
        self.npu_op.sync_input_h2d(0)

        kernel_s = 0.0
        for _ in range(calls):
            wait_t0 = time.perf_counter()
            run = self.npu_op.dispatch(0)
            result = run.wait()
            kernel_s += time.perf_counter() - wait_t0
            if result != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(
                    f"Kernel resmlp_streaming did not complete correctly: {result}"
                )
        return kernel_s

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

        timings = {}
        used_async_pipe = async_pipe and calls >= 2
        if used_async_pipe:
            # Double-buffered pipeline: one slot is dispatched to the NPU while
            # the host prepares the next slot's inputs and/or consumes the previous
            # slot's outputs. In steady state: wall = max(host, kernel).
            wall_t0 = time.perf_counter()
            # Prime the pipeline with slot 0.
            self._prepare_slot_host(0, stacked_SBD, timings)
            run_prev = self.npu_op.dispatch(0)
            slot_prev = 0
            # Prepare slot 1 while slot 0 runs.
            self._prepare_slot_host(1, stacked_SBD, timings)
            for i in range(1, calls):
                slot_next = i % self.NPU_NUM_SLOTS
                result = run_prev.wait()
                if result != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    raise RuntimeError(
                        f"Kernel resmlp_streaming did not complete correctly: {result}"
                    )
                # Immediately dispatch the next slot (already prepared).
                run_prev = self.npu_op.dispatch(slot_next)
                # Consume the just-finished slot (overlapped with new kernel).
                self._consume_slot_host(slot_prev, timings)
                # Prepare the slot after next (overlapped with new kernel too).
                if i + 1 < calls:
                    self._prepare_slot_host(slot_prev, stacked_SBD, timings)
                slot_prev = slot_next
            # Drain: wait for the final dispatch and consume it.
            result = run_prev.wait()
            if result != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(
                    f"Kernel resmlp_streaming did not complete correctly: {result}"
                )
            self._consume_slot_host(slot_prev, timings)
            wall_s = time.perf_counter() - wall_t0
        else:
            kernel_s = 0.0
            wall_t0 = time.perf_counter()
            for _ in range(calls):
                _, elapsed = self.process_features_stacked(stacked_SBD, timings=timings)
                kernel_s += elapsed
            wall_s = time.perf_counter() - wall_t0

        if used_async_pipe:
            # In the overlapped loop, dispatch→wait spans host work too, so measure
            # kernel throughput separately with an isolated dispatch+wait calibration.
            kernel_s = self._measure_kernel_wait(stacked_SBD, calls)

        host_prepare_s = sum(
            timings.get(key, 0.0)
            for key in ("pack_s", "write_buffer_s", "h2d_sync_s")
        )
        host_consume_s = sum(
            timings.get(key, 0.0)
            for key in ("d2h_sync_s", "read_buffer_s", "detile_s")
        )

        result = {
            "num_samples_requested": num_samples,
            "num_samples_processed": processed_samples,
            "calls": calls,
            "wall_s": wall_s,
            "kernel_s": kernel_s,
            "host_prepare_s": host_prepare_s,
            "host_consume_s": host_consume_s,
            "host_total_s": host_prepare_s + host_consume_s,
            "wall_sample_s": processed_samples / wall_s,
            "kernel_sample_s": processed_samples / kernel_s,
            "async_pipe": used_async_pipe,
        }
        for key in (
            "pack_s",
            "write_buffer_s",
            "h2d_sync_s",
            "d2h_sync_s",
            "read_buffer_s",
            "detile_s",
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
        f"  dataset={service.dataset}, epoch={service.epoch}, "
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
            "kernel_s",
            "host_total_s",
            "wall_sample_s",
            "kernel_sample_s",
        ):
            print(f"  {key}: {stats[key]}")
        if stats["async_pipe"]:
            print("  kernel_s: isolated dispatch+wait calibration (not the overlapped wall loop)")

        calls = stats["calls"]
        print("\nPer-call breakdown (µs):")
        stage_keys = [
            ("pack_s", "pack(to_tiled)"),
            ("write_buffer_s", "write_buffer"),
            ("h2d_sync_s", "h2d_sync(DMA)"),
            ("d2h_sync_s", "d2h_sync(DMA)"),
            ("read_buffer_s", "read_buffer"),
            ("detile_s", "detile(from_tiled)"),
        ]
        for key, label in stage_keys:
            us_per_call = stats[key] / calls * 1e6
            print(f"  {label:>20}: {us_per_call:>10.2f} µs/call ({stats[key] * 1000:>9.2f} ms total)")
        print(
            f"  {'host_prepare_total':>20}: {stats['host_prepare_s'] / calls * 1e6:>10.2f} µs/call"
        )
        print(
            f"  {'host_consume_total':>20}: {stats['host_consume_s'] / calls * 1e6:>10.2f} µs/call"
        )
        print(f"  {'kernel_wait':>20}: {stats['kernel_s'] / calls * 1e6:>10.2f} µs/call")
        wall_us_per_call = stats["wall_s"] / calls * 1e6
        print(f"  {'wall_total':>20}: {wall_us_per_call:>10.2f} µs/call")
        if stats["async_pipe"]:
            print("  note: async host stages overlap with the kernel, so their totals do not add up to wall_total")
        else:
            accounted = stats["host_total_s"] + stats["kernel_s"]
            unaccounted_us = (stats["wall_s"] - accounted) / calls * 1e6
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
