"""
Train the 32-layer residual MLP on MNIST.

Usage:
    python resmlp/train.py                    # train from scratch
    python resmlp/train.py --epochs 20        # more epochs
    python resmlp/train.py --resume ckpt.pt   # resume from checkpoint

Checkpoints are saved to resmlp/checkpoints/.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from resmlp.mnist_utils import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_SIZE,
    get_mnist_dataloaders,
)
from resmlp.model import ResMLP


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def build_checkpoint(args, epoch, model, optimizer, eval_name, eval_loss, eval_acc):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "pipeline": "hybrid",
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "eval_split": eval_name,
        "val_size": args.val_size,
        "split_seed": args.split_seed,
        "npu_batch_size": args.batch_size,
    }
    if eval_name == "val":
        checkpoint["val_loss"] = eval_loss
        checkpoint["val_acc"] = eval_acc
    else:
        checkpoint["test_loss"] = eval_loss
        checkpoint["test_acc"] = eval_acc
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train ResMLP on MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="resmlp/checkpoints")
    args = parser.parse_args()

    device = "cpu"  # MNIST is small enough for CPU training

    # Model
    model = ResMLP(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  embed: {784} → {args.hidden_dim}")
    print(f"  hidden: {args.num_layers} × ResidualLinear({args.hidden_dim})")
    print(f"  head: {args.hidden_dim} → 10")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        args.batch_size,
        data_dir=args.data_dir,
        val_size=args.val_size,
        split_seed=args.split_seed,
    )
    eval_loader = val_loader if val_loader is not None else test_loader
    eval_name = "val" if val_loader is not None else "test"
    print(f"Data split: train={len(train_loader.dataset):,}, {eval_name}={len(eval_loader.dataset):,}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_eval_acc = float("-inf")
    best_path = save_dir / "resmlp_best.pt"

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:3d}: "
              f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"{eval_name} loss={eval_loss:.4f} acc={eval_acc:.4f} | "
              f"{elapsed:.1f}s")

        checkpoint = build_checkpoint(
            args, epoch, model, optimizer, eval_name, eval_loss, eval_acc
        )
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(checkpoint, best_path)
            print(f"    → saved {best_path} (best {eval_name})")

        # Save checkpoint every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or epoch == start_epoch + args.epochs - 1:
            path = save_dir / f"resmlp_epoch{epoch:03d}.pt"
            torch.save(checkpoint, path)
            print(f"    → saved {path}")

    print(f"\nFinal {eval_name} accuracy: {eval_acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
