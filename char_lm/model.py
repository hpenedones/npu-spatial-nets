#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Recurrent character-level language model.

Architecture:
    1. Embedding:  char index  →  128-dim vector          (CPU)
    2. Injection:  h = h + embed(char)                    (CPU)
    3. Recurrence: h = ReLU(h @ W) repeated `depth` times (NPU at inference)
    4. Readout:    h  →  logits over vocabulary            (CPU)

The weight matrix W (128×128) is the core learnable parameter.  It lives
in each tile's 64 KB SRAM during NPU inference, enabling thousands of
matmul iterations without DDR traffic.

During training the recurrence runs on CPU with truncated backpropagation:
only the last `bptt_depth` iterations carry gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentCharLM(nn.Module):
    """Character-level language model with a weight-tied recurrent core.

    Parameters
    ----------
    vocab_size : int
        Number of distinct characters.
    hidden_size : int
        Dimension of the hidden state and weight matrix (must be multiple of 8).
    depth : int
        Number of ReLU(h @ W) iterations per character.  Effective neural
        network depth per token.
    bptt_depth : int
        How many of the `depth` iterations carry gradients during training.
        The first (depth - bptt_depth) run with torch.no_grad().
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        depth: int = 500,
        bptt_depth: int = 20,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.bptt_depth = bptt_depth

        self.embed = nn.Embedding(vocab_size, hidden_size)
        # The shared weight matrix — this is what gets loaded onto NPU tiles
        self.W = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.readout = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialise W with spectral norm < 1 to keep recurrence contractive."""
        nn.init.orthogonal_(self.W)
        # Scale down so repeated application doesn't explode
        with torch.no_grad():
            self.W.mul_(0.8)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def _apply_recurrence(self, h: torch.Tensor) -> torch.Tensor:
        """Apply h = ReLU(h @ W) for `depth` iterations with truncated BPTT."""
        no_grad_iters = self.depth - self.bptt_depth

        # Phase 1: forward-only (no gradients)
        if no_grad_iters > 0:
            with torch.no_grad():
                for _ in range(no_grad_iters):
                    h = F.relu(h @ self.W)
            h = h.detach().requires_grad_(True)

        # Phase 2: with gradients (for backprop)
        for _ in range(self.bptt_depth):
            h = F.relu(h @ self.W)

        return h

    def forward(
        self, chars: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence of characters and predict the next one at each step.

        Parameters
        ----------
        chars : (batch, seq_len) long tensor of character indices.
        hidden : (batch, hidden_size) initial hidden state, or None for zeros.

        Returns
        -------
        logits : (batch, seq_len, vocab_size) — next-character predictions.
        hidden : (batch, hidden_size) — final hidden state (detached).
        """
        batch_size, seq_len = chars.shape
        if hidden is None:
            hidden = torch.zeros(
                batch_size, self.hidden_size, device=chars.device
            )

        embeds = self.embed(chars)  # (batch, seq_len, hidden_size)
        all_logits = []

        for t in range(seq_len):
            hidden = hidden + embeds[:, t]
            hidden = self._apply_recurrence(hidden)
            all_logits.append(self.readout(hidden))

        logits = torch.stack(all_logits, dim=1)
        return logits, hidden.detach()

    @torch.no_grad()
    def generate(
        self,
        start_chars: torch.Tensor,
        num_chars: int = 200,
        temperature: float = 0.8,
        hidden: torch.Tensor | None = None,
    ) -> list[int]:
        """Autoregressively generate characters (CPU-only path).

        Parameters
        ----------
        start_chars : (1, prompt_len) long tensor — the prompt.
        num_chars : how many characters to generate after the prompt.
        temperature : sampling temperature (lower = more deterministic).
        hidden : optional initial hidden state.

        Returns
        -------
        List of generated character indices (prompt + generated).
        """
        self.eval()
        device = next(self.parameters()).device

        # Process prompt
        _, hidden = self.forward(start_chars, hidden)

        generated = start_chars[0].tolist()
        current_char = start_chars[0, -1].unsqueeze(0).unsqueeze(0)

        for _ in range(num_chars):
            logits, hidden = self.forward(current_char, hidden)
            probs = F.softmax(logits[0, 0] / temperature, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            generated.append(next_idx)
            current_char = torch.tensor([[next_idx]], device=device)

        return generated

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts by component."""
        return {
            "embedding": self.embed.weight.numel(),
            "recurrent_W": self.W.numel(),
            "readout": sum(p.numel() for p in self.readout.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
