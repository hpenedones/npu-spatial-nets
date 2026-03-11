#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Interactive text completion with a trained character LM.

Usage::

    python -m char_lm.complete                         # latest checkpoint
    python -m char_lm.complete --checkpoint data/wikipedia_transformer_checkpoint.pt
    python -m char_lm.complete --chars 500 --temperature 0.6
"""

import argparse
import sys
import torch
from pathlib import Path

from char_lm.data import Vocabulary
from char_lm.model import RecurrentCharLM
from char_lm.transformer_baseline import TransformerCharLM

DATA_DIR = Path(__file__).parent.parent / "data"


def load_any_checkpoint(path: Path):
    """Load either a recurrent or transformer checkpoint."""
    ckpt = torch.load(path, weights_only=False)
    vocab = Vocabulary(ckpt["vocab_chars"])

    if "config" in ckpt:
        # Transformer checkpoint
        cfg = ckpt["config"]
        model = TransformerCharLM(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ff=cfg["d_ff"],
            max_seq_len=cfg["max_seq_len"],
        )
        kind = f"Transformer {cfg['n_layers']}L {cfg['d_model']}d"
    else:
        # Recurrent checkpoint
        model = RecurrentCharLM(
            vocab_size=ckpt.get("vocab_size", len(vocab.chars)),
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            block_size=ckpt.get("block_size", 4),
            bptt_blocks=ckpt.get("bptt_blocks", None),
        )
        kind = f"Recurrent {ckpt['num_layers']} layers"

    model.load_state_dict(ckpt["model_state"])
    return model, vocab, kind


def main():
    parser = argparse.ArgumentParser(description="Interactive char LM completion")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--chars", type=int, default=200,
                        help="Characters to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        # Find the most recently modified checkpoint
        candidates = sorted(
            DATA_DIR.glob("*_checkpoint.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Also check legacy names
        for legacy_name in ["charlm_checkpoint.pt", "transformer_baseline.pt"]:
            legacy = DATA_DIR / legacy_name
            if legacy.exists():
                candidates.append(legacy)
        if not candidates:
            print("No checkpoint found in data/.")
            print("Run `python -m char_lm.train` or "
                  "`python -m char_lm.transformer_baseline` first.")
            sys.exit(1)
        ckpt_path = candidates[0]

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model, vocab, kind = load_any_checkpoint(ckpt_path)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {kind}, {params:,} params")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Vocab: {vocab.size} characters")
    print(f"Type a prompt and press Enter. Ctrl-C to quit.\n")

    try:
        while True:
            prompt = input(">>> ")
            if not prompt:
                continue
            # Filter prompt to known vocab chars
            safe = "".join(c for c in prompt if c in vocab.char_to_idx
                           or c.lower() in vocab.char_to_idx
                           or c.isupper())
            prompt_ids = torch.tensor([vocab.encode(safe)])
            generated = model.generate(
                prompt_ids, num_chars=args.chars, temperature=args.temperature
            )
            text = vocab.decode(generated)
            # Print only the generated part (after prompt)
            print(text[len(safe):])
            print()
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
