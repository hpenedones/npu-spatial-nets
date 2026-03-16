"""
Train ResMLP on image datasets with NPU acceleration.

Two training modes are supported:

  1. `hybrid`   : CPU embed + CPU head, NPU residual stack.
  2. `full-npu` : NPU embed + NPU residual stack + NPU head/loss/update.

At the default 8 columns, the full-NPU pipeline consumes 32 compute tiles as:
    embed tile + 30 residual tiles + head tile

Reduced-shape full-NPU variants use fewer columns and therefore
`num_cols * 4 - 2` residual layers.
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ml_dtypes import bfloat16

from aie.utils import DefaultNPURuntime
from iron.common.aie_context import AIEContext
from resmlp import from_tiled, to_tiled
from resmlp.data_utils import (
    SUPPORTED_DATASETS,
    get_dataset_config,
    get_dataset_dataloaders,
    resolve_dataset_name,
)
from resmlp.model import ResMLP
from resmlp.training_design import ROWS_PER_COL
from resmlp.training_full_design import N_CLS_PADDED, residual_drainback_enabled
from resmlp.training_full_op import FullTrainingPipeline
from resmlp.training_op import TrainingPipeline

NPU_KERNEL_LR = 0.005
REDUCED_SHAPE_NPU_KERNEL_LR = 0.0005
REDUCED_SHAPE_EMBED_SCALE = 0.25
REDUCED_SHAPE_HEAD_SCALE = 0.25
REDUCED_SHAPE_WEIGHT_CLIP_MAX_ABS = 8.0
FULL_NPU_CONTEXT_GC_INTERVAL = 256
FULL_NPU_FRESH_RUNTIME_GC_INTERVAL = 64
FULL_NPU_CLI_HARD_EXIT = False
FULL_NPU_WEIGHT_CHECK_INTERVAL = 20
SMALLH_RESIDENT_MAX_H = 16
SMALLH_RESIDENT_DEFAULT_WINDOW_BATCHES = 64


def cpu_forward_residual(x_np, weights_bf16):
    """Run residual layers on CPU in bf16-truncated arithmetic."""
    current = x_np.astype(np.float32)
    for w in weights_bf16:
        w_f32 = w.astype(np.float32)
        mm = (current @ w_f32.T).astype(bfloat16).astype(np.float32)
        relu_out = np.maximum(mm, 0)
        current = (current + relu_out).astype(bfloat16).astype(np.float32)
    return current


def unpack_residual_weights(flat_w, H, num_layers):
    weights = []
    for i in range(num_layers):
        w_tile = flat_w[i * H * H: (i + 1) * H * H]
        weights.append(from_tiled(w_tile, H, H))
    return weights


def sync_full_npu_weights_to_model(model, embed_packed, residual_packed, head_packed, H):
    model.load_embed_weight(from_tiled(
        embed_packed, model.embed.in_features, model.embed.out_features
    ))
    model.load_residual_weights(
        unpack_residual_weights(residual_packed, H, len(model.layers))
    )
    model.load_head_weight(from_tiled(head_packed, H, N_CLS_PADDED))


def evict_full_npu_contexts():
    """Reset cached full-NPU hardware contexts without flushing inst BOs."""
    context_cache = getattr(DefaultNPURuntime, "_context_cache", None)
    cleanup_entry = getattr(DefaultNPURuntime, "_cleanup_entry", None)
    if context_cache is None or cleanup_entry is None:
        DefaultNPURuntime.cleanup()
        return

    while context_cache:
        _, entry = context_cache.popitem(last=False)
        cleanup_entry(entry)


def validate_full_npu_window_plan(H, window_batches, max_train_batches):
    if window_batches < 1:
        raise ValueError("--window-batches must be >= 1")


def resident_smallh_enabled(H, num_cols):
    return residual_drainback_enabled(H, num_cols) and H <= SMALLH_RESIDENT_MAX_H


def full_npu_uses_fresh_runtime(H, num_cols):
    return residual_drainback_enabled(H, num_cols) and not resident_smallh_enabled(H, num_cols)


def default_full_npu_window_batches(H, num_cols, max_train_batches):
    if resident_smallh_enabled(H, num_cols):
        window_batches = SMALLH_RESIDENT_DEFAULT_WINDOW_BATCHES
        if max_train_batches is not None:
            window_batches = min(window_batches, max_train_batches)
        if window_batches > 1 and window_batches % 2 == 1:
            window_batches -= 1
        return max(1, window_batches)
    if residual_drainback_enabled(H, num_cols):
        return 1
    return 10


def prepare_full_npu_operator(model, H, B, num_cols, window_batches, sgd_lr):
    evict_full_npu_contexts()
    ctx = AIEContext(use_runlist=False)
    npu_op = FullTrainingPipeline(
        H=H,
        B=B,
        K_EMBED=model.embed.in_features,
        num_cols=num_cols,
        window_batches=window_batches,
        sgd_lr=sgd_lr,
        context=ctx,
    )
    ctx.compile_all()
    ctx.prepare_runtime()
    return ctx, npu_op


def read_full_npu_weights(npu_op, model, H, *, check_finite=True, weights_already_synced=False):
    if not weights_already_synced:
        npu_op.sync_resident_weights_from_device()
    embed_out = npu_op.read_buffer(
        "embed_wt",
        (model.embed.in_features * H,),
        copy=True,
    )
    residual_out = npu_op.read_buffer(
        "res_wt",
        (npu_op.num_residual * H * H,),
        copy=True,
    )
    head_out = npu_op.read_buffer("head_wt", (H * N_CLS_PADDED,), copy=True)

    if check_finite:
        for name, buf in (
            ("embed_wt", embed_out),
            ("res_wt", residual_out),
            ("head_wt", head_out),
        ):
            if not np.isfinite(np.asarray(buf, dtype=np.float32)).all():
                raise RuntimeError(f"{name} became non-finite after full-NPU batch")

    return {
        "embed_packed": embed_out,
        "residual_packed": residual_out,
        "head_packed": head_out,
    }


def clip_packed_weights(packed, max_abs):
    clipped = np.clip(np.asarray(packed, dtype=np.float32), -max_abs, max_abs)
    return clipped.astype(bfloat16, copy=False)


def execute_full_npu_window(
    npu_op,
    model,
    embed_packed,
    residual_packed,
    head_packed,
    x_tiled,
    labels_buf,
    H,
    *,
    sync_weights_to_device=True,
    read_weights_back=True,
):
    npu_op.write_buffer("x_raw", x_tiled)
    if sync_weights_to_device:
        npu_op.write_buffer("embed_wt", embed_packed)
        npu_op.write_buffer("res_wt", residual_packed)
        npu_op.write_buffer("head_wt", head_packed)
    npu_op.write_buffer("labels", labels_buf)

    elapsed = npu_op.run_resident_window(
        sync_weights_to_device=sync_weights_to_device,
        sync_weights_from_device=read_weights_back,
    )

    window_batches = npu_op.window_batches
    B = npu_op.B
    labels_io = npu_op.read_buffer(
        "labels", (window_batches * 2 * B,), copy=True, dtype=np.int32
    )
    preds_np = labels_io.reshape(window_batches, 2 * B)[:, B:].reshape(-1)
    stats = {
        "preds": preds_np,
        "npu_time": elapsed,
    }
    if read_weights_back:
        stats.update(
            read_full_npu_weights(
                npu_op,
                model,
                H,
                weights_already_synced=True,
            )
        )
    return stats


def run_full_npu_batch(model, embed_packed, residual_packed, head_packed,
                       x_tiled, labels_buf, H, B, num_cols, window_batches=1,
                       sgd_lr=NPU_KERNEL_LR, weight_clip_max_abs=None):
    validate_full_npu_window_plan(H, window_batches, window_batches)
    evict_full_npu_contexts()
    ctx = None
    npu_op = None
    try:
        ctx, npu_op = prepare_full_npu_operator(
            model, H=H, B=B, num_cols=num_cols, window_batches=window_batches,
            sgd_lr=sgd_lr,
        )
        stats = execute_full_npu_window(
            npu_op,
            model,
            embed_packed,
            residual_packed,
            head_packed,
            x_tiled,
            labels_buf,
            H,
            sync_weights_to_device=True,
            read_weights_back=True,
        )
        if weight_clip_max_abs is not None:
            stats["embed_packed"] = clip_packed_weights(
                stats["embed_packed"], weight_clip_max_abs
            )
            stats["residual_packed"] = clip_packed_weights(
                stats["residual_packed"], weight_clip_max_abs
            )
            stats["head_packed"] = clip_packed_weights(
                stats["head_packed"], weight_clip_max_abs
            )
        return stats
    finally:
        npu_op = None
        ctx = None
        evict_full_npu_contexts()


def evaluate_model(model, loader, criterion, max_batches=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            logits = model(images.float())
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    return total_loss / total, correct / total


def run_hybrid_epoch(model, optimizer, criterion, train_loader, npu_op,
                     residual_weights, residual_packed, H, B,
                     max_batches=None):
    zero_buf = np.zeros(B * H, dtype=bfloat16)
    running_loss = 0.0
    correct = 0
    total = 0
    npu_time = 0.0
    npu_calls = 0

    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        x_flat = images.view(B, -1).float()
        x_embedded = torch.nn.functional.linear(
            x_flat, model.embed.weight, model.embed.bias
        )

        x_np = x_embedded.detach().numpy().astype(bfloat16)
        y_hidden_np = cpu_forward_residual(x_np, residual_weights)

        y_hidden_t = torch.from_numpy(y_hidden_np.astype(np.float32))
        y_hidden_t.requires_grad_(True)
        logits = model.head(y_hidden_t)
        loss = criterion(logits, labels)

        loss.backward()
        gy_np = y_hidden_t.grad.numpy().astype(bfloat16)

        npu_op.write_buffer("act_in", to_tiled(x_np))
        npu_op.write_buffer("weights_in", residual_packed)
        npu_op.write_buffer("act_out", zero_buf.copy())
        npu_op.write_buffer("grad_in", to_tiled(gy_np))
        npu_op.write_buffer("grad_out", zero_buf.copy())

        t_npu = time.perf_counter()
        try:
            npu_op.run_runlist()
        except Exception as exc:
            raise RuntimeError(
                f"hybrid NPU run failed at batch {batch_idx}"
            ) from exc
        npu_time += time.perf_counter() - t_npu
        npu_calls += 1

        flat_w = npu_op.read_buffer(
            "weights_in", (len(residual_weights) * H * H,), copy=True
        )
        residual_packed = flat_w.copy()
        residual_weights = unpack_residual_weights(flat_w, H, len(residual_weights))
        model.load_residual_weights(residual_weights)

        gx_flat = npu_op.read_buffer("grad_out", (B * H,), copy=True)
        gx_np = from_tiled(gx_flat, B, H).astype(np.float32)
        x_embedded.backward(torch.from_numpy(gx_np))
        optimizer.step()

        running_loss += loss.item() * B
        correct += (logits.detach().argmax(1) == labels).sum().item()
        total += B
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    return {
        "train_loss": running_loss / total,
        "train_acc": correct / total,
        "npu_time": npu_time,
        "npu_calls": npu_calls,
        "residual_weights": residual_weights,
        "residual_packed": residual_packed,
    }


def run_full_npu_epoch(model, train_loader,
                       embed_packed, residual_packed, head_packed, H, B,
                       num_cols, max_batches=None, window_batches=1,
                       sgd_lr=NPU_KERNEL_LR, weight_clip_max_abs=None):
    validate_full_npu_window_plan(H, window_batches, max_batches)
    total = 0
    correct = 0
    npu_time = 0.0
    npu_calls = 0

    model.train()

    K = model.embed.in_features
    zero_x_tiled = to_tiled(np.zeros((B, K), dtype=bfloat16))
    ctx = None
    npu_op = None
    resident_smallh_mode = resident_smallh_enabled(H, num_cols)
    use_fresh_runtime_per_window = full_npu_uses_fresh_runtime(H, num_cols)
    gc_interval = (
        FULL_NPU_FRESH_RUNTIME_GC_INTERVAL
        if use_fresh_runtime_per_window
        else FULL_NPU_CONTEXT_GC_INTERVAL
    )

    try:
        if not use_fresh_runtime_per_window:
            ctx, npu_op = prepare_full_npu_operator(
                model, H=H, B=B, num_cols=num_cols, window_batches=window_batches,
                sgd_lr=sgd_lr,
            )

        loader_iter = iter(train_loader)
        batch_idx = 0
        resident_weights = window_batches % 2 == 0
        weights_initialized = False
        resident_weight_resync_needed = False
        while True:
            batch_group = []
            while len(batch_group) < window_batches:
                if max_batches is not None and batch_idx >= max_batches:
                    break
                try:
                    batch_group.append(next(loader_iter))
                except StopIteration:
                    break
                batch_idx += 1

            if not batch_group:
                break

            if batch_idx > 0 and batch_idx % gc_interval == 0:
                gc.collect()

            x_tiles = []
            labels_list = []
            labels_buf = np.full(window_batches * 2 * B, -1, dtype=np.int32)
            for group_idx, (images, labels) in enumerate(batch_group):
                x_raw = images.view(B, -1).float().numpy().astype(bfloat16)
                x_tiles.append(to_tiled(x_raw))
                labels_np = labels.numpy().astype(np.int32, copy=False)
                labels_list.append(labels_np)
                offset = group_idx * 2 * B
                labels_buf[offset : offset + B] = labels_np

            while len(x_tiles) < window_batches:
                x_tiles.append(zero_x_tiled)

            x_tiled = np.concatenate(x_tiles)

            try:
                if use_fresh_runtime_per_window:
                    batch_stats = run_full_npu_batch(
                        model,
                        embed_packed,
                        residual_packed,
                        head_packed,
                        x_tiled,
                        labels_buf,
                        H,
                        B,
                        num_cols,
                        window_batches=window_batches,
                        sgd_lr=sgd_lr,
                        weight_clip_max_abs=weight_clip_max_abs,
                    )
                else:
                    sync_weights_to_device = True
                    read_weights_back = True
                    if resident_weights:
                        sync_weights_to_device = (
                            resident_weight_resync_needed or not weights_initialized
                        )
                        read_weights_back = (
                            resident_smallh_mode
                            and weight_clip_max_abs is not None
                        )
                    batch_stats = execute_full_npu_window(
                        npu_op,
                        model,
                        embed_packed,
                        residual_packed,
                        head_packed,
                        x_tiled,
                        labels_buf,
                        H,
                        sync_weights_to_device=sync_weights_to_device,
                        read_weights_back=read_weights_back,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"full-NPU run failed at batch {batch_idx - len(batch_group)}"
                ) from exc
            weights_initialized = True
            npu_time += batch_stats["npu_time"]
            npu_calls += 1
            resident_weight_resync_needed = False
            if use_fresh_runtime_per_window:
                embed_packed = batch_stats["embed_packed"]
                residual_packed = batch_stats["residual_packed"]
                head_packed = batch_stats["head_packed"]
            elif resident_smallh_mode and resident_weights and weight_clip_max_abs is not None:
                embed_packed = clip_packed_weights(
                    batch_stats["embed_packed"], weight_clip_max_abs
                )
                residual_packed = clip_packed_weights(
                    batch_stats["residual_packed"], weight_clip_max_abs
                )
                head_packed = clip_packed_weights(
                    batch_stats["head_packed"], weight_clip_max_abs
                )
                resident_weight_resync_needed = True
            elif (
                resident_weights
                and not resident_smallh_mode
                and npu_calls % FULL_NPU_WEIGHT_CHECK_INTERVAL == 0
            ):
                try:
                    weight_stats = read_full_npu_weights(npu_op, model, H)
                except Exception as exc:
                    raise RuntimeError(
                        f"full-NPU weights became invalid at batch {batch_idx - len(batch_group)}"
                    ) from exc
                embed_packed = weight_stats["embed_packed"]
                residual_packed = weight_stats["residual_packed"]
                head_packed = weight_stats["head_packed"]
            elif not resident_weights:
                embed_packed = batch_stats["embed_packed"]
                residual_packed = batch_stats["residual_packed"]
                head_packed = batch_stats["head_packed"]
            preds_np = batch_stats["preds"]
            for group_idx, labels_np in enumerate(labels_list):
                pred_slice = preds_np[group_idx * B : (group_idx + 1) * B]
                total += B
                correct += int((pred_slice == labels_np).sum())

        if use_fresh_runtime_per_window:
            final_weights = {
                "embed_packed": embed_packed,
                "residual_packed": residual_packed,
                "head_packed": head_packed,
            }
        elif resident_weights and weights_initialized:
            final_weights = read_full_npu_weights(npu_op, model, H)
        else:
            final_weights = {
                "embed_packed": embed_packed,
                "residual_packed": residual_packed,
                "head_packed": head_packed,
            }
        return {
            "train_loss": None,
            "train_acc": correct / total if total else None,
            "npu_time": npu_time,
            "npu_calls": npu_calls,
            "embed_packed": final_weights["embed_packed"],
            "residual_packed": final_weights["residual_packed"],
            "head_packed": final_weights["head_packed"],
        }
    finally:
        npu_op = None
        ctx = None
        evict_full_npu_contexts()


def main():
    global FULL_NPU_CLI_HARD_EXIT
    parser = argparse.ArgumentParser(description="Train ResMLP on a dataset with NPU acceleration")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pipeline", choices=["hybrid", "full-npu"], default="hybrid")
    parser.add_argument("--lr-head", type=float, default=1e-3,
                        help="Adam LR for CPU embed/head in hybrid mode")
    parser.add_argument("--lr-npu", type=float, default=None,
                        help="Compile-time SGD LR for the NPU kernels")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size (B=8 to match NPU microbatch)")
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-cols", type=int, default=8)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--weight-scale", type=float, default=0.02,
                        help="Initial weight scale for residual layers")
    parser.add_argument("--embed-scale", type=float, default=None,
                        help="Optional multiplier for the initial embed weights")
    parser.add_argument("--head-scale", type=float, default=None,
                        help="Optional multiplier for the initial head weights")
    parser.add_argument("--weight-clip-max-abs", type=float, default=None,
                        help="Optional host-side max-abs clipping for full-NPU weights after each invocation")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Optional cap on training batches per epoch")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="Optional cap on evaluation batches")
    parser.add_argument("--window-batches", type=int, default=None,
                        help="Microbatches per full-NPU invocation (default depends on full-NPU mode)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="resmlp/checkpoints")
    args = parser.parse_args()

    B = 8
    H = args.hidden_dim
    num_cols = args.num_cols
    num_tiles = num_cols * ROWS_PER_COL
    expected_layers = num_tiles - 2 if args.pipeline == "full-npu" else num_tiles
    num_layers = args.num_layers if args.num_layers is not None else expected_layers

    assert args.batch_size == B, (
        f"Batch size must be {B} to match NPU microbatch dimension"
    )
    if args.pipeline == "full-npu":
        assert 2 <= num_cols <= 8, "full-npu mode requires between 2 and 8 columns"
    assert num_layers == expected_layers, (
        f"{args.pipeline} mode expects num_layers={expected_layers}, got {num_layers}"
    )
    if args.window_batches is None:
        args.window_batches = (
            default_full_npu_window_batches(H, num_cols, args.max_train_batches)
            if args.pipeline == "full-npu"
            else 1
        )
    resident_smallh_mode = (
        args.pipeline == "full-npu" and resident_smallh_enabled(H, num_cols)
    )
    if resident_smallh_mode and args.window_batches > 1 and args.window_batches % 2 == 1:
        args.window_batches -= 1
    if args.pipeline == "full-npu":
        validate_full_npu_window_plan(H, args.window_batches, args.max_train_batches)
    reduced_shape_mode = (
        args.pipeline == "full-npu" and residual_drainback_enabled(H, num_cols)
    )
    if args.lr_npu is None:
        args.lr_npu = (
            REDUCED_SHAPE_NPU_KERNEL_LR
            if reduced_shape_mode
            else NPU_KERNEL_LR
        )
    if args.embed_scale is None:
        args.embed_scale = (
            REDUCED_SHAPE_EMBED_SCALE
            if reduced_shape_mode
            else 1.0
        )
    if args.head_scale is None:
        args.head_scale = (
            REDUCED_SHAPE_HEAD_SCALE
            if reduced_shape_mode
            else 1.0
        )
    if args.weight_clip_max_abs is None and reduced_shape_mode:
        args.weight_clip_max_abs = REDUCED_SHAPE_WEIGHT_CLIP_MAX_ABS
    FULL_NPU_CLI_HARD_EXIT = args.pipeline == "full-npu"

    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
    dataset_name = resolve_dataset_name(
        args.dataset,
        resume_ckpt.get("dataset") if resume_ckpt else None,
    )
    dataset_cfg = get_dataset_config(dataset_name)
    input_dim = resume_ckpt.get("input_dim", dataset_cfg["input_dim"]) if resume_ckpt else dataset_cfg["input_dim"]
    num_classes = (
        resume_ckpt.get("num_classes", dataset_cfg["num_classes"])
        if resume_ckpt
        else dataset_cfg["num_classes"]
    )
    residual_bias = bool(resume_ckpt.get("residual_bias", False)) if resume_ckpt else False
    if input_dim != dataset_cfg["input_dim"] or num_classes != dataset_cfg["num_classes"]:
        raise ValueError(
            f"Checkpoint/input metadata ({input_dim} inputs, {num_classes} classes) "
            f"does not match dataset '{dataset_name}'"
        )
    if residual_bias:
        raise ValueError(
            "train_npu.py does not support residual-bias checkpoints yet because the current NPU "
            "residual training kernels are bias-free."
        )
    args.dataset = dataset_name

    model = ResMLP(
        hidden_dim=H,
        num_layers=num_layers,
        input_dim=input_dim,
        num_classes=num_classes,
        residual_bias=False,
    )
    if resume_ckpt:
        model.load_state_dict(resume_ckpt["model"])
        print(f"Resumed from {args.resume}")
    else:
        with torch.no_grad():
            for layer in model.layers:
                layer.weight.mul_(args.weight_scale / 0.1)

    if args.pipeline == "full-npu":
        model.zero_linear_biases()
        with torch.no_grad():
            model.embed.weight.mul_(args.embed_scale)
            model.head.weight.mul_(args.head_scale)

    total_params = sum(p.numel() for p in model.parameters())
    residual_params = sum(layer.weight.numel() for layer in model.layers)
    print(f"Model: {total_params:,} parameters")
    print(f"  dataset: {dataset_name}")
    print(f"  pipeline: {args.pipeline}")
    print(f"  embed: {input_dim} → {H}")
    print(f"  head: {H} → {num_classes}")
    print(f"  residual layers: {num_layers}")
    if args.pipeline == "hybrid":
        print(f"  NPU residual: {residual_params:,}  ({num_layers} × {H}×{H})")
        print(f"  CPU embed+head: {total_params - residual_params:,}")
    else:
        print("  full-NPU training uses 1 embed tile + "
              f"{num_layers} residual tiles + 1 head tile")
        print("  embed/head biases are pinned to zero to match current kernels")
        if resident_smallh_mode:
            print("  reduced-shape residual drainback is enabled for this configuration")
            print("  small-H resident mode is enabled")
            print("  runtime is prepared once per epoch; no per-window runtime recreation")
            print("  weights stay on-device across microbatches within each window")
            print(f"  embed/head init scales: {args.embed_scale:g} / {args.head_scale:g}")
            if args.weight_clip_max_abs is not None:
                print(f"  host-side weight clip max abs: {args.weight_clip_max_abs:g}")
                print("  host weight sync/clip happens once per window for stability")
            else:
                print("  host weight sync is deferred until epoch end")
        elif residual_drainback_enabled(H, num_cols):
            print("  reduced-shape residual drainback is enabled for this configuration")
            print("  this mode recreates the runtime each invocation for stability")
            print(f"  host GC runs every {FULL_NPU_FRESH_RUNTIME_GC_INTERVAL} invocations")
            print(f"  embed/head init scales: {args.embed_scale:g} / {args.head_scale:g}")
            if args.weight_clip_max_abs is not None:
                print(f"  host-side weight clip max abs: {args.weight_clip_max_abs:g}")
        print(f"  window batches per invocation: {args.window_batches}")

    criterion = nn.CrossEntropyLoss()
    optimizer = None
    if args.pipeline == "hybrid":
        cpu_params = list(model.embed.parameters()) + list(model.head.parameters())
        optimizer = torch.optim.Adam(cpu_params, lr=args.lr_head)

    print(f"\nCompiling NPU pipeline ({num_tiles} tiles)...", flush=True)
    t0 = time.time()
    if args.pipeline == "full-npu" and full_npu_uses_fresh_runtime(H, num_cols):
        ctx = None
        npu_op = None
    else:
        ctx = AIEContext(use_runlist=(args.pipeline != "full-npu"))
        if args.pipeline == "hybrid":
            npu_op = TrainingPipeline(
                H=H, B=B, num_cols=num_cols, sgd_lr=args.lr_npu, context=ctx
            )
        else:
            npu_op = FullTrainingPipeline(
                H=H,
                B=B,
                K_EMBED=model.embed.in_features,
                num_cols=num_cols,
                window_batches=args.window_batches,
                sgd_lr=args.lr_npu,
                context=ctx,
            )
        ctx.compile_all()
        ctx.prepare_runtime()
        if args.pipeline == "full-npu":
            evict_full_npu_contexts()
            npu_op = None
    print(f"  Compiled in {time.time() - t0:.1f}s")

    residual_weights = model.export_residual_weights()
    residual_packed = np.concatenate([to_tiled(w) for w in residual_weights])
    embed_packed = None
    head_packed = None
    if args.pipeline == "full-npu":
        embed_packed = to_tiled(model.export_embed_weight())
        head_packed = to_tiled(model.export_head_weight(padded_classes=N_CLS_PADDED))

    loader_workers = 0 if args.pipeline == "full-npu" else 2
    loader_pin_memory = args.pipeline != "full-npu"
    train_loader, _, test_loader = get_dataset_dataloaders(
        dataset_name,
        args.batch_size,
        data_dir=args.data_dir,
        val_size=0,
        train_num_workers=loader_workers,
        eval_num_workers=loader_workers,
        pin_memory=loader_pin_memory,
        drop_last_train=True,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 70}")
    print(f"Training for {args.epochs} epochs  |  pipeline={args.pipeline}  "
          f"lr_npu={args.lr_npu}")
    print(f"{'═' * 70}\n")

    for epoch in range(args.epochs):
        ep_t0 = time.time()

        if args.pipeline == "hybrid":
            stats = run_hybrid_epoch(
                model, optimizer, criterion, train_loader, npu_op,
                residual_weights, residual_packed, H, B,
                max_batches=args.max_train_batches
            )
            residual_weights = stats["residual_weights"]
            residual_packed = stats["residual_packed"]
        else:
            stats = run_full_npu_epoch(
                model, train_loader,
                embed_packed, residual_packed, head_packed, H, B,
                num_cols=num_cols,
                max_batches=args.max_train_batches,
                window_batches=args.window_batches,
                sgd_lr=args.lr_npu,
                weight_clip_max_abs=args.weight_clip_max_abs,
            )
            embed_packed = stats["embed_packed"]
            residual_packed = stats["residual_packed"]
            head_packed = stats["head_packed"]
            sync_full_npu_weights_to_model(
                model, embed_packed, residual_packed, head_packed, H
            )

        if args.pipeline == "hybrid":
            test_loss, test_acc = evaluate_model(
                model, test_loader, criterion, max_batches=args.max_eval_batches
            )
        else:
            test_loss, test_acc = None, None
        ep_time = time.time() - ep_t0
        train_images = (
            len(train_loader.dataset)
            if args.max_train_batches is None
            else min(len(train_loader.dataset), args.max_train_batches * B)
        )
        imgs_per_sec = train_images / ep_time
        npu_ms_per_batch = (
            stats["npu_time"] / stats["npu_calls"] * 1000
            if stats["npu_calls"] else 0.0
        )

        if stats["train_acc"] is None:
            print(f"  Epoch {epoch:2d}:  "
                  f"train loss=n/a  |  "
                  f"test loss=n/a acc=n/a  |  "
                  f"{ep_time:.1f}s  {imgs_per_sec:.0f} img/s  "
                  f"({npu_ms_per_batch:.1f} ms/npu-call, {stats['npu_calls']} calls)")
        elif stats["train_loss"] is None or test_acc is None:
            print(f"  Epoch {epoch:2d}:  "
                  f"train loss=n/a acc={stats['train_acc']:.4f}  |  "
                  f"test loss=n/a acc=n/a  |  "
                  f"{ep_time:.1f}s  {imgs_per_sec:.0f} img/s  "
                  f"({npu_ms_per_batch:.1f} ms/npu-call, {stats['npu_calls']} calls)")
        else:
            print(f"  Epoch {epoch:2d}:  "
                  f"train loss={stats['train_loss']:.4f} acc={stats['train_acc']:.4f}  |  "
                  f"test loss={test_loss:.4f} acc={test_acc:.4f}  |  "
                  f"{ep_time:.1f}s  {imgs_per_sec:.0f} img/s  "
                  f"({npu_ms_per_batch:.1f} ms/npu-call, {stats['npu_calls']} calls)")

        if args.pipeline == "hybrid" and ((epoch + 1) % 5 == 0 or epoch == args.epochs - 1):
            tag = args.pipeline.replace("-", "_")
            path = save_dir / f"resmlp_{tag}_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "dataset": dataset_name,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "pipeline": args.pipeline,
                "hidden_dim": H,
                "num_layers": num_layers,
                "num_cols": num_cols,
                "input_dim": input_dim,
                "num_classes": num_classes,
                "residual_bias": False,
            }, path)
            print(f"    → saved {path}")
        elif args.pipeline == "full-npu" and epoch == args.epochs - 1:
            tag = args.pipeline.replace("-", "_")
            path = save_dir / f"resmlp_{tag}_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "dataset": dataset_name,
                "pipeline": args.pipeline,
                "hidden_dim": H,
                "num_layers": num_layers,
                "num_cols": num_cols,
                "input_dim": input_dim,
                "num_classes": num_classes,
                "residual_bias": False,
            }, path)
            print(f"    → saved {path}")

    print(f"\n{'═' * 70}")
    if args.pipeline == "hybrid":
        print(f"Final test accuracy: {test_acc:.4f}")
    else:
        print("Computing final full-NPU accuracies with current host-synced weights...")
        _, final_train_acc = evaluate_model(
            model, train_loader, criterion, max_batches=args.max_train_batches
        )
        _, final_test_acc = evaluate_model(
            model, test_loader, criterion, max_batches=args.max_eval_batches
        )
        print(f"Final train accuracy: {final_train_acc:.4f}")
        print(f"Final test accuracy:  {final_test_acc:.4f}")
    print(f"{'═' * 70}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    if FULL_NPU_CLI_HARD_EXIT and exit_code == 0:
        # Work around an external XRT/pyxrt teardown crash that can trigger
        # during interpreter shutdown after successful full-NPU runs.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    sys.exit(exit_code)
