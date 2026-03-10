#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Train the recurrent character-level language model on tiny Shakespeare.

Usage::

    python -m char_lm.train [--depth 500] [--epochs 10] [--lr 1e-3]

The trained checkpoint is saved to ``data/charlm_checkpoint.pt``.
"""

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from char_lm.data import load_shakespeare, Vocabulary
from char_lm.model import RecurrentCharLM

CHECKPOINT_DIR = Path(__file__).parent.parent / "data"


def _clamp_spectral_norm(model: RecurrentCharLM, max_norm: float = 1.0):
    """Project each W to have spectral norm ≤ max_norm."""
    with torch.no_grad():
        for W in model.weights:
            sigma = torch.linalg.norm(W, ord=2)
            if sigma > max_norm:
                W.mul_(max_norm / sigma)


def train_epoch(
    model: RecurrentCharLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)  # (batch, seq_len, vocab)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        _clamp_spectral_norm(model)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def eval_loss(model: RecurrentCharLM, loader: DataLoader, device: torch.device) -> float:
    """Compute average cross-entropy loss on a dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def save_checkpoint(
    model: RecurrentCharLM, vocab: Vocabulary, path: Path, metadata: dict
):
    """Save model weights (always on CPU), vocabulary, and training metadata."""
    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "vocab_chars": vocab.chars,
            "hidden_size": model.hidden_size,
            "depth": model.depth,
            "bptt_depth": model.bptt_depth,
            "num_layers": model.num_layers,
            "vocab_size": vocab.size,
            **metadata,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[RecurrentCharLM, Vocabulary]:
    """Load model and vocabulary from a checkpoint."""
    ckpt = torch.load(path, weights_only=False)
    vocab = Vocabulary(ckpt["vocab_chars"])
    model = RecurrentCharLM(
        vocab_size=ckpt["vocab_size"],
        hidden_size=ckpt["hidden_size"],
        depth=ckpt["depth"],
        bptt_depth=ckpt["bptt_depth"],
        num_layers=ckpt.get("num_layers", 1),
    )
    model.load_state_dict(ckpt["model_state"])
    return model, vocab


def main():
    parser = argparse.ArgumentParser(description="Train character LM")
    parser.add_argument("--depth", type=int, default=500,
                        help="Recurrence depth per character (default: 500)")
    parser.add_argument("--bptt-depth", type=int, default=20,
                        help="Backprop-through-time depth (default: 20)")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="Hidden dimension (default: 128)")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Number of distinct W matrices (default: 1)")
    parser.add_argument("--seq-len", type=int, default=64,
                        help="Training sequence length (default: 64)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Training device: cpu, cuda, or auto (default: auto)")
    args = parser.parse_args()

    # Select device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("TileFlow Character Language Model — Training")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    train_ds, val_ds, vocab = load_shakespeare(seq_len=args.seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    print(f"Vocabulary: {vocab.size} characters")
    print(f"Training:   {len(train_ds)} sequences of length {args.seq_len}")
    print(f"Validation: {len(val_ds)} sequences")
    print()

    # Create model
    model = RecurrentCharLM(
        vocab_size=vocab.size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        bptt_depth=args.bptt_depth,
        num_layers=args.num_layers,
    )
    params = model.count_parameters()
    print(f"Model parameters:")
    print(f"  Embedding:    {params['embedding']:,}")
    print(f"  Recurrent W:  {params['recurrent_W']:,} ({params['num_layers']} layer(s) × {args.hidden_size}²)")
    print(f"  Readout:      {params['readout']:,}")
    print(f"  Total:        {params['total']:,}")
    print(f"  Depth:        {args.depth} ({model.depth_per_layer}/layer, {args.bptt_depth} with gradients)")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model = model.to(device)

    # Training loop
    best_val_loss = float("inf")
    ckpt_path = CHECKPOINT_DIR / "charlm_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, args.grad_clip)
        val_loss = eval_loss(model, val_loader, device)
        elapsed = time.time() - t0

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, vocab, ckpt_path, {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
            marker = " ← saved"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"{elapsed:.1f}s{marker}"
        )

    # Generate a sample (always on CPU for compatibility)
    print("\n" + "=" * 60)
    print("Sample generation (CPU, temperature=0.8):")
    print("=" * 60)
    model_cpu, vocab = load_checkpoint(ckpt_path)
    prompt = "\nTo be, or not to be"
    prompt_ids = torch.tensor([vocab.encode(prompt)])
    generated = model_cpu.generate(prompt_ids, num_chars=200, temperature=0.8)
    print(vocab.decode(generated))
    print()
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
