"""
PyTorch model for the 32-layer residual MLP.

Architecture:
    784 (MNIST) → Linear → H
    [y = relu(x @ W_i) + x]  × 32 layers   (NPU)
    H → Linear → 10 classes

The 32 hidden layers match the 32 NPU tiles exactly.
"""

import torch
import torch.nn as nn


class ResidualLinear(nn.Module):
    """One residual layer:  y = relu(x @ W) + x"""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        # Scale down to prevent activation explosion across 32 layers
        with torch.no_grad():
            self.weight.mul_(0.1)

    def forward(self, x):
        return torch.relu(x @ self.weight) + x


class ResMLP(nn.Module):
    """32-layer residual MLP for MNIST classification.

    Args:
        hidden_dim: Width of all hidden layers (must match NPU tile dimension).
        num_layers: Number of residual layers (must match number of NPU tiles).
        num_classes: Number of output classes.
        input_dim: Input feature dimension (784 for MNIST).
    """

    def __init__(self, hidden_dim=160, num_layers=32,
                 num_classes=10, input_dim=784):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [ResidualLinear(hidden_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten MNIST images
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def export_npu_weights(self):
        """Export the 32 hidden-layer weight matrices as numpy bf16 arrays.

        Returns list of (H, H) numpy arrays in bfloat16, one per layer.
        """
        import numpy as np
        from ml_dtypes import bfloat16
        weights = []
        for layer in self.layers:
            W = layer.weight.detach().cpu().float().numpy()
            weights.append(W.astype(bfloat16))
        return weights
