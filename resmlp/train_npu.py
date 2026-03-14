"""
Train ResMLP on MNIST with NPU acceleration.

Two training modes are supported:

  1. `hybrid`   : CPU embed + CPU head, NPU residual stack.
  2. `full-npu` : NPU embed + NPU residual stack + NPU head/loss/update.

The full-NPU pipeline consumes 32 compute tiles as:
    embed tile + 30 residual tiles + head tile

so `full-npu` mode trains a 30-layer residual stack rather than the original
32-layer residual-only snake.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ml_dtypes import bfloat16
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from aie.utils import DefaultNPURuntime
from iron.common.aie_context import AIEContext
from resmlp import from_tiled, to_tiled
from resmlp.model import ResMLP
from resmlp.training_design import ROWS_PER_COL
from resmlp.training_full_design import N_CLS_PADDED, NUM_RESIDUAL
from resmlp.training_full_op import FullTrainingPipeline
from resmlp.training_op import TrainingPipeline

NPU_KERNEL_LR = 0.01


def get_dataloaders(batch_size, data_dir="data", num_workers=2, pin_memory=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True,
                              transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


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


def run_full_npu_batch(model, embed_packed, residual_packed, head_packed,
                       x_tiled, labels_buf, H, B, num_cols):
    evict_full_npu_contexts()
    try:
        ctx = AIEContext(use_runlist=False)
        npu_op = FullTrainingPipeline(
            H=H,
            B=B,
            K_EMBED=model.embed.in_features,
            num_cols=num_cols,
            context=ctx,
        )
        ctx.compile_all()
        ctx.prepare_runtime()

        npu_op.write_buffer("x_raw", x_tiled)
        npu_op.write_buffer("embed_wt", embed_packed)
        npu_op.write_buffer("res_wt", residual_packed)
        npu_op.write_buffer("head_wt", head_packed)
        npu_op.write_buffer("labels", labels_buf)

        t_npu = time.perf_counter()
        npu_op.run_runlist()
        elapsed = time.perf_counter() - t_npu

        preds_np = npu_op.read_buffer("labels", (2 * B,), copy=True, dtype=np.int32)[B:]
        embed_out = npu_op.read_buffer(
            "embed_wt", (model.embed.in_features * H,), copy=True
        )
        residual_out = npu_op.read_buffer(
            "res_wt", (NUM_RESIDUAL * H * H,), copy=True
        )
        head_out = npu_op.read_buffer(
            "head_wt", (H * N_CLS_PADDED,), copy=True
        )
    finally:
        evict_full_npu_contexts()

    for name, buf in (
        ("embed_wt", embed_out),
        ("res_wt", residual_out),
        ("head_wt", head_out),
    ):
        if not np.isfinite(np.asarray(buf, dtype=np.float32)).all():
            raise RuntimeError(f"{name} became non-finite after full-NPU batch")

    return {
        "preds": preds_np,
        "embed_packed": embed_out,
        "residual_packed": residual_out,
        "head_packed": head_out,
        "npu_time": elapsed,
    }


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
                       num_cols, max_batches=None):
    total = 0
    correct = 0
    npu_time = 0.0
    npu_calls = 0

    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):
        x_raw = images.view(B, -1).float().numpy().astype(bfloat16)
        x_tiled = to_tiled(x_raw)
        labels_np = labels.numpy().astype(np.int32, copy=False)
        labels_buf = np.empty(2 * B, dtype=np.int32)
        labels_buf[:B] = labels_np
        labels_buf[B:] = -1

        try:
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
            )
        except Exception as exc:
            raise RuntimeError(
                f"full-NPU run failed at batch {batch_idx}"
            ) from exc
        embed_packed = batch_stats["embed_packed"]
        residual_packed = batch_stats["residual_packed"]
        head_packed = batch_stats["head_packed"]
        npu_time += batch_stats["npu_time"]
        npu_calls += 1
        total += B
        preds_np = batch_stats["preds"]
        correct += int((preds_np[:B] == labels_np).sum())
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    return {
        "train_loss": None,
        "train_acc": correct / total if total else None,
        "npu_time": npu_time,
        "npu_calls": npu_calls,
        "embed_packed": embed_packed,
        "residual_packed": residual_packed,
        "head_packed": head_packed,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ResMLP on MNIST with NPU")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pipeline", choices=["hybrid", "full-npu"], default="hybrid")
    parser.add_argument("--lr-head", type=float, default=1e-3,
                        help="Adam LR for CPU embed/head in hybrid mode")
    parser.add_argument("--lr-npu", type=float, default=NPU_KERNEL_LR,
                        help="Fixed SGD LR baked into the current NPU kernels")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size (B=8 to match NPU microbatch)")
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-cols", type=int, default=8)
    parser.add_argument("--weight-scale", type=float, default=0.02,
                        help="Initial weight scale for residual layers")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Optional cap on training batches per epoch")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="Optional cap on evaluation batches")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="resmlp/checkpoints")
    args = parser.parse_args()

    B = 8
    H = args.hidden_dim
    num_cols = args.num_cols
    num_tiles = num_cols * ROWS_PER_COL
    expected_layers = num_tiles - 2 if args.pipeline == "full-npu" else num_tiles
    num_layers = args.num_layers if args.num_layers is not None else expected_layers

    if not np.isclose(args.lr_npu, NPU_KERNEL_LR):
        raise ValueError(
            f"--lr-npu={args.lr_npu} is not supported yet; current kernels use "
            f"a fixed SGD learning rate of {NPU_KERNEL_LR}"
        )

    assert args.batch_size == B, (
        f"Batch size must be {B} to match NPU microbatch dimension"
    )
    if args.pipeline == "full-npu":
        assert num_cols == 8, "full-npu mode currently requires all 8 columns (32 tiles)"
    assert num_layers == expected_layers, (
        f"{args.pipeline} mode expects num_layers={expected_layers}, got {num_layers}"
    )

    model = ResMLP(hidden_dim=H, num_layers=num_layers)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")
    else:
        with torch.no_grad():
            for layer in model.layers:
                layer.weight.mul_(args.weight_scale / 0.1)

    if args.pipeline == "full-npu":
        model.zero_linear_biases()

    total_params = sum(p.numel() for p in model.parameters())
    residual_params = sum(layer.weight.numel() for layer in model.layers)
    print(f"Model: {total_params:,} parameters")
    print(f"  pipeline: {args.pipeline}")
    print(f"  residual layers: {num_layers}")
    if args.pipeline == "hybrid":
        print(f"  NPU residual: {residual_params:,}  ({num_layers} × {H}×{H})")
        print(f"  CPU embed+head: {total_params - residual_params:,}")
    else:
        print("  full-NPU training uses 1 embed tile + "
              f"{num_layers} residual tiles + 1 head tile")
        print("  embed/head biases are pinned to zero to match current kernels")
        print("  fresh-hw-context mode currently persists head weights across batches")

    criterion = nn.CrossEntropyLoss()
    optimizer = None
    if args.pipeline == "hybrid":
        cpu_params = list(model.embed.parameters()) + list(model.head.parameters())
        optimizer = torch.optim.Adam(cpu_params, lr=args.lr_head)

    print(f"\nCompiling NPU pipeline ({num_tiles} tiles)...", flush=True)
    t0 = time.time()
    ctx = AIEContext(use_runlist=(args.pipeline != "full-npu"))
    if args.pipeline == "hybrid":
        npu_op = TrainingPipeline(H=H, B=B, num_cols=num_cols, context=ctx)
    else:
        npu_op = FullTrainingPipeline(H=H, B=B, K_EMBED=model.embed.in_features,
                                      num_cols=num_cols, context=ctx)
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
    train_loader, test_loader = get_dataloaders(
        args.batch_size,
        num_workers=loader_workers,
        pin_memory=loader_pin_memory,
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
                "test_acc": test_acc,
                "test_loss": test_loss,
                "pipeline": args.pipeline,
                "hidden_dim": H,
                "num_layers": num_layers,
                "num_cols": num_cols,
            }, path)
            print(f"    → saved {path}")
        elif args.pipeline == "full-npu" and epoch == args.epochs - 1:
            tag = args.pipeline.replace("-", "_")
            path = save_dir / f"resmlp_{tag}_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "pipeline": args.pipeline,
                "hidden_dim": H,
                "num_layers": num_layers,
                "num_cols": num_cols,
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
    sys.exit(main())
