#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Dataset and vocabulary for character-level language modelling.

The vocabulary maps each unique character to an integer index.
SequenceDataset produces fixed-length windows of (input, target) pairs
where target[t] = input[t+1] (next-character prediction).
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


class Vocabulary:
    """Bidirectional mapping between characters and integer indices."""

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.char_to_idx = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        return cls(sorted(set(text)))

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(path.read_text()))

    def save(self, path: Path):
        path.write_text(json.dumps(self.chars))

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text]

    def decode(self, indices) -> str:
        return "".join(self.chars[i] for i in indices)

    @property
    def size(self) -> int:
        return len(self.chars)


class SequenceDataset(Dataset):
    """Fixed-length windows over a character corpus.

    Each item is (input_ids, target_ids) where target_ids are shifted
    by one position: target[t] = input[t+1].
    """

    def __init__(self, text: str, vocab: Vocabulary, seq_len: int = 64):
        self.data = torch.tensor(vocab.encode(text), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


def load_shakespeare(seq_len: int = 64, val_fraction: float = 0.1):
    """Load tiny Shakespeare and return train/val datasets + vocabulary."""
    text = (DATA_DIR / "shakespeare.txt").read_text()
    vocab = Vocabulary.from_text(text)

    split = int(len(text) * (1 - val_fraction))
    train_ds = SequenceDataset(text[:split], vocab, seq_len)
    val_ds = SequenceDataset(text[split:], vocab, seq_len)

    return train_ds, val_ds, vocab
