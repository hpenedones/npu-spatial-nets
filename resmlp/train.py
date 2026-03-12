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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resmlp.model import ResMLP


def get_dataloaders(batch_size=128, data_dir="data"):
    """Standard MNIST train/test loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True,
                              transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2)
    return train_loader, test_loader


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


def main():
    parser = argparse.ArgumentParser(description="Train ResMLP on MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--num-layers", type=int, default=32)
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

    # Data
    train_loader, test_loader = get_dataloaders(args.batch_size)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:3d}: "
              f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"test loss={test_loss:.4f} acc={test_acc:.4f} | "
              f"{elapsed:.1f}s")

        # Save checkpoint every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or epoch == start_epoch + args.epochs - 1:
            path = save_dir / f"resmlp_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "test_acc": test_acc,
            }, path)
            print(f"    → saved {path}")

    print(f"\nFinal test accuracy: {test_acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
