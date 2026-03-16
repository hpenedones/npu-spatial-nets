"""
Train a simple CIFAR-10 baseline model with a standard local recipe.

Usage:
    python -m cifar_baseline.train --device cuda
"""

from __future__ import annotations

import argparse
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from cifar_baseline.preact_resnet import PreActResNet18
from resmlp.data_utils import (
    DEFAULT_SPLIT_SEED,
    get_dataset_dataloaders,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested, but torch.cuda.is_available() is false")
    return requested


def optimizer_to_device(optimizer: optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def maybe_channels_last(images: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled:
        return images.contiguous(memory_format=torch.channels_last)
    return images.contiguous()


def autocast_context(device: str, enabled: bool):
    if device != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler,
    device: str,
    *,
    amp: bool,
    channels_last: bool,
    max_batches: int | None = None,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = maybe_channels_last(images, channels_last).to(device, non_blocking=(device == "cuda"))
        labels = labels.to(device, non_blocking=(device == "cuda"))

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at train batch {batch_idx}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_seen += labels.size(0)
    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    *,
    amp: bool,
    channels_last: bool,
    max_batches: int | None = None,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = maybe_channels_last(images, channels_last).to(device, non_blocking=(device == "cuda"))
        labels = labels.to(device, non_blocking=(device == "cuda"))
        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_seen += labels.size(0)
    return total_loss / total_seen, total_correct / total_seen


def build_checkpoint(args, epoch, model, optimizer, scheduler, scaler, *, eval_name, eval_loss, eval_acc):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "dataset": "cifar10",
        "pipeline": "preactresnet18-gpu",
        "model_name": "PreActResNet18",
        "seed": args.seed,
        "train_aug": args.train_aug,
        "val_size": args.val_size,
        "split_seed": args.split_seed,
        "train_batch_size": args.batch_size,
        "optimizer_name": "sgd",
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "scheduler_name": "cosine",
        "epochs": args.epochs,
        "amp": args.amp,
        "channels_last": args.channels_last,
    }
    checkpoint[f"{eval_name}_loss"] = eval_loss
    checkpoint[f"{eval_name}_acc"] = eval_acc
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PreActResNet18 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--val-size", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-aug", choices=("none", "basic", "strong"), default="basic")
    parser.add_argument("--train-num-workers", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="cifar_baseline/checkpoints/preactresnet18")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-mode", choices=("full", "weights_only"), default="full")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")

    set_seed(args.seed)
    device = resolve_device(args.device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = args.benchmark

    resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    model = PreActResNet18(num_classes=10)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    if resume_ckpt:
        model.load_state_dict(resume_ckpt["model"])
    if args.compile:
        model = torch.compile(model)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device == "cuda"))

    start_epoch = 0
    if resume_ckpt and args.resume_mode == "full":
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        optimizer_to_device(optimizer, device)
        if resume_ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        if resume_ckpt.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(resume_ckpt["scaler"])
        start_epoch = resume_ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    elif resume_ckpt:
        print("Loaded model weights from checkpoint; optimizer/scheduler reinitialized")

    train_loader, val_loader, test_loader = get_dataset_dataloaders(
        "cifar10",
        args.batch_size,
        data_dir=args.data_dir,
        train_aug=args.train_aug,
        val_size=args.val_size,
        split_seed=args.split_seed,
        train_num_workers=args.train_num_workers,
        eval_num_workers=args.eval_num_workers,
        pin_memory=(device == "cuda"),
    )
    eval_loader = val_loader if val_loader is not None else test_loader
    eval_name = "val" if val_loader is not None else "test"

    print(f"Model: PreActResNet18 ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"  device: {device}")
    print(f"  train_aug: {args.train_aug}")
    print(f"  val_size: {args.val_size:,}")
    print(f"  optimizer: SGD lr={args.lr:g} momentum={args.momentum:g} wd={args.weight_decay:g}")
    print(f"  label_smoothing: {args.label_smoothing:g}")
    print(f"  amp: {args.amp and device == 'cuda'}")
    print(f"  channels_last: {args.channels_last}")
    print(f"  benchmark: {args.benchmark and device == 'cuda'}")
    print(f"  eval_every: {args.eval_every}")
    print(f"Data split (cifar10): train={len(train_loader.dataset):,}, {eval_name}={len(eval_loader.dataset):,}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "preactresnet18_best.pt"
    final_path = save_dir / "preactresnet18_final.pt"
    best_eval_acc = float("-inf")
    last_epoch = start_epoch + args.epochs - 1

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            amp=args.amp,
            channels_last=args.channels_last,
            max_batches=args.max_train_batches,
        )
        should_eval = ((epoch - start_epoch + 1) % args.eval_every == 0) or epoch == last_epoch
        eval_loss = None
        eval_acc = None
        if should_eval:
            eval_loss, eval_acc = evaluate(
                model,
                eval_loader,
                criterion,
                device,
                amp=args.amp,
                channels_last=args.channels_last,
                max_batches=args.max_eval_batches,
            )
        scheduler.step()
        elapsed = time.time() - t0

        if should_eval:
            print(
                f"  Epoch {epoch:3d}: "
                f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"{eval_name} loss={eval_loss:.4f} acc={eval_acc:.4f} | "
                f"lr={epoch_lr:.3e} | "
                f"{elapsed:.1f}s"
            )

            checkpoint = build_checkpoint(
                args,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                eval_name=eval_name,
                eval_loss=eval_loss,
                eval_acc=eval_acc,
            )
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                torch.save(checkpoint, best_path)
                print(f"    -> saved {best_path} (best {eval_name})")
        else:
            print(
                f"  Epoch {epoch:3d}: "
                f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"lr={epoch_lr:.3e} | "
                f"{elapsed:.1f}s"
            )

    torch.save(checkpoint, final_path)
    print(f"Saved final checkpoint to {final_path}")

    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best_ckpt["model"])
        model = model.to(device)
        best_loss, best_acc = evaluate(
            model,
            eval_loader,
            criterion,
            device,
            amp=args.amp,
            channels_last=args.channels_last,
            max_batches=args.max_eval_batches,
        )
        print(f"Best {eval_name} accuracy: {best_acc:.4f} (loss={best_loss:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
