from __future__ import annotations

import numpy as np
import torch
from ml_dtypes import bfloat16

from simplecnn.model import TinyConvNet


def quantized_model_copy(model: TinyConvNet) -> TinyConvNet:
    quantized = TinyConvNet()
    quantized.load_packed_weights(model.export_packed_weights())
    quantized.eval()
    return quantized


def forward_reference_logits(model: TinyConvNet, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x = images.float()
        x = torch.relu(model.conv1(x)).to(torch.bfloat16).float()
        x = torch.relu(model.conv2(x)).to(torch.bfloat16).float()
        x = torch.relu(model.conv3(x)).to(torch.bfloat16).float()
        x = x.mean(dim=(2, 3)).to(torch.bfloat16).float()
        return model.head(x).to(torch.bfloat16).float()


def pack_images(images: torch.Tensor) -> np.ndarray:
    return (
        images.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        .astype(bfloat16, copy=False)
        .reshape(-1)
    )
