"""
PyTorch model for the residual MLP.

Architecture:
    D (flattened image input) → Linear → H
    [y = relu(x @ W_i) + x]  × N layers   (NPU)
    H → Linear → C classes

    The number of residual layers is configurable so the same class can model
    both the 32-layer hybrid pipeline and the 30-layer full-NPU pipeline.
"""

import numpy as np
import torch
import torch.nn as nn


class ResidualLinear(nn.Module):
    """One residual layer:  y = relu(x @ W) + x"""

    def __init__(self, dim, *, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        # Scale down to prevent activation explosion across 32 layers
        with torch.no_grad():
            self.weight.mul_(0.1)

    def forward(self, x):
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return torch.relu(y) + x


class ResMLP(nn.Module):
    """Residual MLP for dense tabular classification (e.g. HIGGS).

    Args:
        hidden_dim: Width of all hidden layers (must match NPU tile dimension).
        num_layers: Number of residual layers (must match number of NPU tiles).
        num_classes: Number of output classes.
        input_dim: Input feature dimension (e.g. 28 for native HIGGS features).
    """

    def __init__(self, hidden_dim=160, num_layers=32,
                 num_classes=10, input_dim=784, residual_bias=False):
        super().__init__()
        self.residual_bias = residual_bias
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [ResidualLinear(hidden_dim, bias=residual_bias) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def zero_linear_biases(self):
        with torch.no_grad():
            self.embed.bias.zero_()
            self.head.bias.zero_()

    def export_residual_weights(self):
        """Export residual-layer weights as numpy bfloat16 arrays.

        Returns a list of `(H, H)` arrays, one per residual layer.
        """
        from ml_dtypes import bfloat16

        weights = []
        for layer in self.layers:
            W = layer.weight.detach().cpu().float().numpy()
            weights.append(W.astype(bfloat16))
        return weights

    def export_npu_weights(self):
        """Backward-compatible alias for residual-layer export."""
        return self.export_residual_weights()

    def export_embed_weight(self):
        """Export embed weights in NPU layout: `[input_dim, hidden_dim]`."""
        from ml_dtypes import bfloat16

        return self.embed.weight.detach().cpu().float().numpy().T.astype(bfloat16)

    def export_head_weight(self, padded_classes=None):
        """Export head weights in NPU layout: `[hidden_dim, num_classes]`.

        When `padded_classes` is provided, zero-pad the class dimension to match
        the NPU head kernel's 8-wide tiling requirement.
        """
        from ml_dtypes import bfloat16

        weight = self.head.weight.detach().cpu().float().numpy().T
        if padded_classes is not None:
            if padded_classes < weight.shape[1]:
                raise ValueError(
                    f"padded_classes={padded_classes} is smaller than "
                    f"num_classes={weight.shape[1]}"
                )
            padded = np.zeros((weight.shape[0], padded_classes), dtype=np.float32)
            padded[:, : weight.shape[1]] = weight
            weight = padded
        return weight.astype(bfloat16)

    def load_residual_weights(self, weights):
        """Load residual-layer weights from numpy arrays in `[H, H]` layout."""
        if len(weights) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} residual weights, got {len(weights)}"
            )

        with torch.no_grad():
            for layer, weight in zip(self.layers, weights):
                array = np.asarray(weight, dtype=np.float32)
                if array.shape != tuple(layer.weight.shape):
                    raise ValueError(
                        f"Residual weight has shape {array.shape}, expected "
                        f"{tuple(layer.weight.shape)}"
                    )
                layer.weight.copy_(torch.from_numpy(array))

    def load_embed_weight(self, weight):
        """Load embed weights from NPU layout `[input_dim, hidden_dim]`."""
        array = np.asarray(weight, dtype=np.float32)
        expected = (self.embed.in_features, self.embed.out_features)
        if array.shape != expected:
            raise ValueError(
                f"Embed weight has shape {array.shape}, expected {expected}"
            )

        with torch.no_grad():
            self.embed.weight.copy_(torch.from_numpy(array.T))

    def load_head_weight(self, weight):
        """Load head weights from NPU layout `[hidden_dim, padded_classes]`."""
        array = np.asarray(weight, dtype=np.float32)
        if array.shape[0] != self.head.in_features:
            raise ValueError(
                f"Head weight has hidden dim {array.shape[0]}, expected "
                f"{self.head.in_features}"
            )

        trimmed = array[:, : self.head.out_features]
        with torch.no_grad():
            self.head.weight.copy_(torch.from_numpy(trimmed.T))
