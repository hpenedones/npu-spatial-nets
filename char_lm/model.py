#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Recurrent character-level language model.

Architecture:
    1. Embedding:  char index  →  128-dim vector                      (CPU)
    2. Injection:  h = h + embed(char)                                (CPU)
    3. Recurrence: for each layer W_i:                                (NPU)
                       h = ReLU(h @ W_i)  ×  (depth / num_layers)
    4. Readout:    h  →  logits over vocabulary                       (CPU)

With num_layers=1 this is a weight-tied recurrent net (same W every step).
With num_layers>1, each layer has its own W — giving more parameters while
reusing the same NPU design (one NPU call per layer per character).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentCharLM(nn.Module):
    """Character-level language model with recurrent core.

    Parameters
    ----------
    vocab_size : int
        Number of distinct characters.
    hidden_size : int
        Dimension of the hidden state and weight matrices (multiple of 8).
    depth : int
        Total ReLU(h @ W) iterations per character, split across layers.
    bptt_depth : int
        How many of the final iterations carry gradients (truncated BPTT).
    num_layers : int
        Number of distinct weight matrices.  depth is divided evenly.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        depth: int = 500,
        bptt_depth: int = 20,
        num_layers: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.bptt_depth = bptt_depth
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_size, hidden_size))
            for _ in range(num_layers)
        ])
        self.readout = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    @property
    def depth_per_layer(self) -> int:
        return self.depth // self.num_layers

    def _init_weights(self):
        """Initialise weights for stable deep recurrence.

        Each W is orthogonal (spectral norm = 1.0).  ReLU provides natural
        contraction by zeroing ~50% of activations each step, so explicit
        scaling below 1.0 is unnecessary and kills signal in deep models.
        """
        for W in self.weights:
            nn.init.orthogonal_(W)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def _apply_recurrence(self, h: torch.Tensor) -> torch.Tensor:
        """Apply all layers sequentially with truncated BPTT.

        Each layer applies h = ReLU(h @ W_i) for depth_per_layer iterations.
        Only the last bptt_depth iterations (across all layers) carry gradients.
        """
        dpl = self.depth_per_layer
        total_iters = dpl * self.num_layers
        no_grad_iters = total_iters - self.bptt_depth

        iter_count = 0
        for W in self.weights:
            for _ in range(dpl):
                if iter_count < no_grad_iters:
                    with torch.no_grad():
                        h = F.relu(h @ W)
                else:
                    if iter_count == no_grad_iters:
                        h = h.detach().requires_grad_(True)
                    h = F.relu(h @ W)
                iter_count += 1

        return h

    def forward(
        self, chars: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence of characters, predict next char at each step.

        Parameters
        ----------
        chars : (batch, seq_len) long tensor of character indices.
        hidden : (batch, hidden_size) initial hidden state, or None for zeros.

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
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
        """Autoregressively generate characters (CPU-only path)."""
        self.eval()
        device = next(self.parameters()).device

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
        recurrent = sum(W.numel() for W in self.weights)
        return {
            "embedding": self.embed.weight.numel(),
            "recurrent_W": recurrent,
            "num_layers": self.num_layers,
            "readout": sum(p.numel() for p in self.readout.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
