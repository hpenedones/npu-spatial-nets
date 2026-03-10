#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Minimal Transformer baseline for character-level Shakespeare.

This is a standard GPT-style model (causal attention + feedforward) used
as a quality reference.  It trains on GPU in minutes and reaches val loss
~1.5, which is the target our NPU-friendly architecture should approach.

Usage::

    python -m char_lm.transformer_baseline [--epochs 10] [--device auto]
"""

import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from char_lm.data import load_shakespeare, Vocabulary

CHECKPOINT_DIR = Path(__file__).parent.parent / "data"


class TransformerCharLM(nn.Module):
    """Minimal GPT-style character-level language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits (batch, seq_len, vocab_size)."""
        B, T = chars.shape
        assert T <= self.max_seq_len

        pos = torch.arange(T, device=chars.device).unsqueeze(0)
        x = self.drop(self.embed(chars) + self.pos_embed(pos))

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=chars.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.readout(x)

    @torch.no_grad()
    def generate(
        self,
        start_chars: torch.Tensor,
        num_chars: int = 200,
        temperature: float = 0.8,
    ) -> list[int]:
        """Autoregressively generate characters."""
        self.eval()
        device = next(self.parameters()).device
        generated = start_chars[0].tolist()

        for _ in range(num_chars):
            context = torch.tensor(
                [generated[-self.max_seq_len:]],
                device=device
            )
            logits = self(context)
            probs = F.softmax(logits[0, -1] / temperature, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            generated.append(next_idx)

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def main():
    parser = argparse.ArgumentParser(
        description="Transformer baseline for char-level Shakespeare")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("Transformer Baseline — Character LM on Shakespeare")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_ds, val_ds, vocab = load_shakespeare(seq_len=args.seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = TransformerCharLM(
        vocab_size=vocab.size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=0.1,
    ).to(device)

    params = model.count_parameters()
    print(f"Vocabulary: {vocab.size} characters")
    print(f"Training:   {len(train_ds)} sequences of length {args.seq_len}")
    print(f"Model:      {args.n_layers}L, {args.d_model}d, {args.n_heads}h, "
          f"{args.d_ff}ff")
    print(f"Parameters: {params:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    ckpt_path = CHECKPOINT_DIR / "transformer_baseline.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss = 0.0
        num_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        train_loss = total_loss / num_batches
        scheduler.step()

        # Eval
        model.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                val_total += loss.item()
                val_batches += 1
        val_loss = val_total / val_batches

        elapsed = time.time() - t0
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "vocab_chars": vocab.chars,
                "config": {
                    "d_model": args.d_model, "n_heads": args.n_heads,
                    "n_layers": args.n_layers, "d_ff": args.d_ff,
                    "max_seq_len": args.seq_len, "vocab_size": vocab.size,
                },
                "val_loss": val_loss,
            }, ckpt_path)
            marker = " ← saved"

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | "
              f"ppl {math.exp(val_loss):.1f} | {elapsed:.1f}s{marker}")

    # Generate sample
    print("\n" + "=" * 60)
    print("Sample generation (temperature=0.8):")
    print("=" * 60)
    prompt = "\nTo be, or not to be"
    prompt_ids = torch.tensor([vocab.encode(prompt)], device=device)
    generated = model.generate(prompt_ids, num_chars=300, temperature=0.8)
    print(vocab.decode(generated))
    print()
    print(f"Best val loss: {best_val_loss:.4f} "
          f"(ppl {math.exp(best_val_loss):.1f})")
    print(f"Parameters:    {params:,}")
    print(f"Checkpoint:    {ckpt_path}")


if __name__ == "__main__":
    main()
