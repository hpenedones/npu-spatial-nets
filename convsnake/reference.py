from __future__ import annotations

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16
except ModuleNotFoundError:
    bfloat16 = None

from convsnake.model import StreamingConvNet


def _require_bfloat16():
    if bfloat16 is None:
        raise RuntimeError("ml_dtypes is required for ConvSnake NPU reference packing")
    return bfloat16


def quantized_model_copy(model: StreamingConvNet) -> StreamingConvNet:
    quantized = StreamingConvNet(num_same_blocks=model.num_same_blocks, config=model.config)
    quantized.load_embedded_weights(model.export_embedded_weights())
    quantized.eval()
    return quantized


def forward_reference_logits(model: StreamingConvNet, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x = images.float()
        x = torch.relu(model.conv1(x)).to(torch.bfloat16).float()
        x = torch.relu(model.conv2(x)).to(torch.bfloat16).float()
        x = torch.relu(model.conv3(x)).to(torch.bfloat16).float()
        for conv in model.blocks:
            x = (torch.relu(conv(x)) + x).to(torch.bfloat16).float()
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
        x = x.to(torch.bfloat16).float()
        return model.head(x).to(torch.bfloat16).float()


def pack_image_batches(images: torch.Tensor) -> np.ndarray:
    bf16 = _require_bfloat16()
    if images.ndim == 4:
        packed = images.detach().cpu().permute(0, 2, 3, 1).contiguous()
    elif images.ndim == 5:
        packed = images.detach().cpu().permute(0, 1, 3, 4, 2).contiguous()
    else:
        raise ValueError(f"expected 4D or 5D image tensor, got shape {tuple(images.shape)}")
    return (
        packed
        .numpy()
        .astype(np.float32)
        .astype(bf16, copy=False)
        .reshape(-1)
    )
