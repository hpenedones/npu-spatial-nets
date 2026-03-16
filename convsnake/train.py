"""
Train the convolutional snake on image classification datasets with CPU/GPU training.

Usage:
    python -m convsnake.train --dataset cifar10 --device cuda
    python -m convsnake.train --resume convsnake/checkpoints/convsnake_best.pt
"""

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from convsnake.config import SUPPORTED_DATASETS, build_config, num_blocks_for_cols
from convsnake.model import StreamingConvNet
from resmlp.data_utils import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_SIZE,
    SUPPORTED_TRAIN_AUGS,
    get_dataset_dataloaders,
    resolve_dataset_name,
)


def train_epoch(model, loader, optimizer, criterion, device, *, max_batches=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, *, max_batches=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
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


def build_checkpoint(args, epoch, model, optimizer, scheduler, eval_name, eval_loss, eval_acc, *, cfg, num_cols, num_blocks):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "dataset": args.dataset,
        "pipeline": "convsnake-gpu",
        "num_cols": num_cols,
        "num_blocks": num_blocks,
        "config": cfg.to_dict(),
        "eval_split": eval_name,
        "val_size": args.val_size,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "train_aug": args.train_aug,
        "optimizer_name": args.optimizer,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "scheduler_name": args.scheduler,
        "momentum": args.momentum,
        "min_lr": args.min_lr,
        "train_batch_size": args.batch_size,
        "npu_batch_size": cfg.batch_size,
        "conv_scale": args.conv_scale,
        "head_scale": args.head_scale,
        "residual_blocks": True,
    }
    if eval_name == "val":
        checkpoint["val_loss"] = eval_loss
        checkpoint["val_acc"] = eval_acc
    else:
        checkpoint["test_loss"] = eval_loss
        checkpoint["test_acc"] = eval_acc
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train ConvSnake on an image dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-cols", type=int, default=8)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--train-aug", choices=SUPPORTED_TRAIN_AUGS, default="none")
    parser.add_argument("--train-num-workers", type=int, default=2)
    parser.add_argument("--eval-num-workers", type=int, default=None)
    parser.add_argument("--conv-scale", type=float, default=1.0)
    parser.add_argument("--head-scale", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-mode", choices=("full", "weights_only"), default="full")
    parser.add_argument("--save-dir", type=str, default="convsnake/checkpoints")
    args = parser.parse_args()

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
    num_cols = resume_ckpt.get("num_cols", args.num_cols) if resume_ckpt else args.num_cols
    if resume_ckpt and args.num_cols != num_cols:
        raise ValueError(
            f"Checkpoint was trained with num_cols={num_cols}, but --num-cols={args.num_cols} was requested"
        )
    num_blocks = num_blocks_for_cols(num_cols)
    if resume_ckpt and resume_ckpt.get("num_blocks", num_blocks) != num_blocks:
        raise ValueError(
            f"Checkpoint num_blocks={resume_ckpt.get('num_blocks')} does not match num_cols={num_cols}"
        )

    cfg = build_config(dataset_name)
    args.dataset = dataset_name

    model = StreamingConvNet(num_same_blocks=num_blocks, config=cfg).to(device)
    if resume_ckpt:
        model.load_state_dict(resume_ckpt["model"])
    else:
        model.scale_initial_weights(conv_scale=args.conv_scale, head_scale=args.head_scale)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  device: {device}")
    print(f"  dataset: {dataset_name}")
    print(f"  image: {cfg.img_c} x {cfg.img_h} x {cfg.img_w}")
    print(f"  stem: {cfg.c1}/{cfg.c2}/{cfg.c3}")
    print(f"  blocks: {num_blocks} (num_cols={num_cols})")
    print(f"  head: {cfg.flat_dim} -> {cfg.num_classes}")
    print(f"  optimizer: {args.optimizer} lr={args.lr:g} wd={args.weight_decay:g}")
    print(f"  scheduler: {args.scheduler} min_lr={args.min_lr:g}")
    print(f"  label_smoothing: {args.label_smoothing:g}")
    print(f"  train_aug: {args.train_aug}")

    optimizer = build_optimizer(args, model.parameters())
    scheduler = build_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    start_epoch = 0
    if resume_ckpt and args.resume_mode == "full":
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        optimizer_to_device(optimizer, device)
        if scheduler is not None and resume_ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        start_epoch = resume_ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    elif resume_ckpt:
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
    print(
        f"Data split ({dataset_name}): train={len(train_loader.dataset):,}, "
        f"{eval_name}={len(eval_loader.dataset):,}"
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_eval_acc = float("-inf")
    best_path = save_dir / "convsnake_best.pt"

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_batches=args.max_train_batches,
        )
        eval_loss, eval_acc = evaluate(
            model,
            eval_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
        )
        elapsed = time.time() - t0
        if scheduler is not None:
            scheduler.step()

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
            eval_name,
            eval_loss,
            eval_acc,
            cfg=cfg,
            num_cols=num_cols,
            num_blocks=num_blocks,
        )
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(checkpoint, best_path)
            print(f"    -> saved {best_path} (best {eval_name})")

        if (epoch + 1) % 5 == 0 or epoch == start_epoch + args.epochs - 1:
            path = save_dir / f"convsnake_epoch{epoch:03d}.pt"
            torch.save(checkpoint, path)
            print(f"    -> saved {path}")

    print(f"\nFinal {eval_name} accuracy: {eval_acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
