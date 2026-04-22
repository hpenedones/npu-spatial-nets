"""Train the residual MLP on the HIGGS dataset."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from resmlp.data_utils import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_SIZE,
    SUPPORTED_DATASETS,
    SUPPORTED_TRAIN_AUGS,
    get_dataset_config,
    get_dataset_dataloaders,
    resolve_dataset_name,
)
from resmlp.model import ResMLP


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    non_blocking = device == "cuda"

    for features, labels in loader:
        features = features.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    non_blocking = device == "cuda"

    for features, labels in loader:
        features = features.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested, but torch.cuda.is_available() is false")
    return requested


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_optimizer(args, parameters):
    if args.optimizer == "adam":
        return optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    raise AssertionError(f"Unhandled optimizer: {args.optimizer}")


def build_scheduler(args, optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.min_lr,
        )
    raise AssertionError(f"Unhandled scheduler: {args.scheduler}")


def build_checkpoint(args, epoch, model, optimizer, scheduler, eval_name, eval_loss, eval_acc):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "dataset": args.dataset,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "input_dim": model.embed.in_features,
        "num_classes": model.head.out_features,
        "residual_bias": args.residual_bias,
        "eval_split": eval_name,
        "val_size": args.val_size,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "train_aug": getattr(args, "train_aug", "none"),
        "optimizer_name": args.optimizer,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "scheduler_name": args.scheduler,
        "momentum": args.momentum,
        "min_lr": args.min_lr,
        "npu_batch_size": getattr(args, "npu_batch_size", getattr(args, "batch_size", None)),
    }
    if eval_name == "val":
        checkpoint["val_loss"] = eval_loss
        checkpoint["val_acc"] = eval_acc
    else:
        checkpoint["test_loss"] = eval_loss
        checkpoint["test_acc"] = eval_acc
    return checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the residual MLP on HIGGS")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--min-lr", type=float, default=8.0e-5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--residual-bias", action="store_true")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="higgs")
    parser.add_argument("--data-dir", type=str, default="data/higgs_full")
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=3.0e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--train-aug", choices=SUPPORTED_TRAIN_AUGS, default="none")
    parser.add_argument("--train-num-workers", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-mode", choices=("full", "weights_only"), default="full")
    parser.add_argument("--save-dir", type=str, default="build/higgs_checkpoints")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.npu_batch_size = args.batch_size

    set_seed(args.seed)
    device = resolve_device(args.device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

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
    if resume_ckpt:
        args.hidden_dim = resume_ckpt.get("hidden_dim", args.hidden_dim)
        args.num_layers = resume_ckpt.get("num_layers", args.num_layers)
    ckpt_residual_bias = bool(resume_ckpt.get("residual_bias", False)) if resume_ckpt else False
    if resume_ckpt and args.residual_bias and not ckpt_residual_bias:
        raise ValueError(
            "Cannot resume a checkpoint without residual bias into a residual-bias model. "
            "Start from scratch or resume a checkpoint that was trained with --residual-bias."
        )
    args.residual_bias = ckpt_residual_bias or args.residual_bias
    if input_dim != dataset_cfg["input_dim"] or num_classes != dataset_cfg["num_classes"]:
        raise ValueError(
            f"Checkpoint/input metadata ({input_dim} inputs, {num_classes} classes) "
            f"does not match dataset '{dataset_name}'"
        )
    args.dataset = dataset_name

    model = ResMLP(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        input_dim=input_dim,
        num_classes=num_classes,
        residual_bias=args.residual_bias,
    ).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  device: {device}")
    print(f"  dataset: {dataset_name}")
    print(f"  embed: {input_dim} -> {args.hidden_dim}")
    print(f"  hidden: {args.num_layers} x ResidualLinear({args.hidden_dim})")
    print(f"  head: {args.hidden_dim} -> {num_classes}")
    print(f"  residual_bias: {args.residual_bias}")
    print(f"  optimizer: {args.optimizer} lr={args.lr:g} wd={args.weight_decay:g}")
    print(f"  scheduler: {args.scheduler} min_lr={args.min_lr:g}")
    print(f"  label_smoothing: {args.label_smoothing:g}")

    optimizer = build_optimizer(args, model.parameters())
    scheduler = build_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    start_epoch = 0
    if resume_ckpt:
        model.load_state_dict(resume_ckpt["model"])
        if args.resume_mode == "full":
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            optimizer_to_device(optimizer, device)
            if scheduler is not None and resume_ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(resume_ckpt["scheduler"])
            start_epoch = resume_ckpt["epoch"] + 1
            print(f"Resumed from epoch {start_epoch}")
        else:
            print("Loaded model weights from checkpoint; optimizer/scheduler reinitialized")

    train_loader, val_loader, test_loader = get_dataset_dataloaders(
        dataset_name,
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
    print(f"Data split ({dataset_name}): train={len(train_loader.dataset):,}, {eval_name}={len(eval_loader.dataset):,}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_eval_acc = float("-inf")
    if resume_ckpt and args.resume_mode == "full" and resume_ckpt.get("eval_split") == eval_name:
        best_eval_acc = resume_ckpt.get(f"{eval_name}_acc", float("-inf"))
    best_path = save_dir / "resmlp_best.pt"

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)
        elapsed = time.time() - t0
        if scheduler is not None:
            scheduler.step()

        print(
            f"  Epoch {epoch:3d}: train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"{eval_name} loss={eval_loss:.4f} acc={eval_acc:.4f} | "
            f"lr={epoch_lr:.3e} | {elapsed:.1f}s"
        )

        checkpoint = build_checkpoint(args, epoch, model, optimizer, scheduler, eval_name, eval_loss, eval_acc)
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(checkpoint, best_path)
            print(f"    -> saved {best_path} (best {eval_name})")

        if (epoch + 1) % 5 == 0 or epoch == start_epoch + args.epochs - 1:
            path = save_dir / f"resmlp_epoch{epoch:03d}.pt"
            torch.save(checkpoint, path)
            print(f"    -> saved {path}")

    print(f"\nFinal {eval_name} accuracy: {eval_acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
