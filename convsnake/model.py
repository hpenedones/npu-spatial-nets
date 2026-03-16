from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from ml_dtypes import bfloat16
except ModuleNotFoundError:
    bfloat16 = None

from resmlp import to_tiled

from convsnake.config import DEFAULT_CONFIG, DEFAULT_NUM_BLOCKS, ConvSnakeConfig


def _require_bfloat16():
    if bfloat16 is None:
        raise RuntimeError("ml_dtypes is required for ConvSnake NPU weight export")
    return bfloat16


class StreamingConvNet(nn.Module):
    def __init__(
        self,
        num_same_blocks: int = DEFAULT_NUM_BLOCKS,
        *,
        config: ConvSnakeConfig | None = None,
    ):
        if num_same_blocks < 0:
            raise ValueError("num_same_blocks must be non-negative")
        super().__init__()
        self.config = DEFAULT_CONFIG if config is None else config
        self.num_same_blocks = num_same_blocks
        self.conv1 = nn.Conv2d(
            self.config.img_c,
            self.config.c1,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            self.config.c1,
            self.config.c2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv3 = nn.Conv2d(
            self.config.c2,
            self.config.c3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.blocks = nn.ModuleList(
            nn.Conv2d(self.config.c3, self.config.c3, kernel_size=3, stride=1, padding=1, bias=False)
            for _ in range(num_same_blocks)
        )
        self.head = nn.Linear(self.config.flat_dim, self.config.num_classes, bias=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        for conv in self.blocks:
            x = torch.relu(conv(x)) + x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
        return self.head(x)

    def scale_initial_weights(self, conv_scale: float = 0.25, head_scale: float = 0.25) -> None:
        with torch.no_grad():
            self.conv1.weight.mul_(conv_scale)
            self.conv2.weight.mul_(conv_scale)
            self.conv3.weight.mul_(conv_scale)
            for conv in self.blocks:
                conv.weight.mul_(conv_scale)
            self.head.weight.mul_(head_scale)

    @staticmethod
    def _export_conv_weight(weight: torch.Tensor) -> np.ndarray:
        bf16 = _require_bfloat16()
        return (
            weight.detach()
            .cpu()
            .float()
            .permute(0, 2, 3, 1)
            .contiguous()
            .numpy()
            .astype(bf16)
            .reshape(-1)
        )

    @staticmethod
    def _load_conv_weight(array: np.ndarray, shape: torch.Size) -> torch.Tensor:
        out_c, in_c, k_h, k_w = shape
        return torch.from_numpy(
            np.asarray(array, dtype=np.float32).reshape(out_c, k_h, k_w, in_c).transpose(0, 3, 1, 2)
        )

    def export_head_weight(self) -> np.ndarray:
        bf16 = _require_bfloat16()
        return self.head.weight.detach().cpu().float().numpy().T.astype(bf16).reshape(-1)

    @staticmethod
    def _export_conv_weight_tiled(weight: torch.Tensor, *, out_c_pad: int, k_pad: int) -> np.ndarray:
        bf16 = _require_bfloat16()
        out_c = weight.shape[0]
        w = (
            weight.detach()
            .cpu()
            .float()
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(out_c, -1)
            .numpy()
        )
        k = w.shape[1]
        if out_c > out_c_pad or k > k_pad:
            raise ValueError(f"cannot pad matrix {(out_c, k)} into {(out_c_pad, k_pad)}")
        padded = np.zeros((out_c_pad, k_pad), dtype=bf16)
        padded[:out_c, :k] = w.astype(bf16)
        return to_tiled(padded)

    def export_embedded_weights(self) -> dict[str, np.ndarray]:
        blocks = (
            np.stack([self._export_conv_weight(block.weight) for block in self.blocks])
            if self.blocks
            else np.zeros((0, self.config.block_w_elems), dtype=_require_bfloat16())
        )
        return {
            "conv1": self._export_conv_weight(self.conv1.weight),
            "conv2": self._export_conv_weight(self.conv2.weight),
            "conv3": self._export_conv_weight(self.conv3.weight),
            "blocks": blocks,
            "head": self.export_head_weight(),
        }

    def export_npu_weights(self) -> dict[str, np.ndarray]:
        return self.export_embedded_weights()

    def load_embedded_weights(self, weights: dict[str, np.ndarray]) -> None:
        conv1 = np.asarray(weights["conv1"], dtype=np.float32).reshape(-1)
        conv2 = np.asarray(weights["conv2"], dtype=np.float32).reshape(-1)
        conv3 = np.asarray(weights["conv3"], dtype=np.float32).reshape(-1)
        blocks = np.asarray(weights["blocks"], dtype=np.float32).reshape(
            self.num_same_blocks,
            self.config.block_w_elems,
        )
        head = np.asarray(weights["head"], dtype=np.float32).reshape(
            self.config.flat_dim,
            self.config.num_classes,
        )

        if conv1.size != self.config.conv1_w_elems:
            raise ValueError(f"conv1 has {conv1.size} elements, expected {self.config.conv1_w_elems}")
        if conv2.size != self.config.conv2_w_elems:
            raise ValueError(f"conv2 has {conv2.size} elements, expected {self.config.conv2_w_elems}")
        if conv3.size != self.config.conv3_w_elems:
            raise ValueError(f"conv3 has {conv3.size} elements, expected {self.config.conv3_w_elems}")
        if blocks.shape != (self.num_same_blocks, self.config.block_w_elems):
            raise ValueError(
                f"blocks has shape {blocks.shape}, expected {(self.num_same_blocks, self.config.block_w_elems)}"
            )
        if head.size != self.config.head_w_elems:
            raise ValueError(f"head has {head.size} elements, expected {self.config.head_w_elems}")

        with torch.no_grad():
            self.conv1.weight.copy_(self._load_conv_weight(conv1, self.conv1.weight.shape))
            self.conv2.weight.copy_(self._load_conv_weight(conv2, self.conv2.weight.shape))
            self.conv3.weight.copy_(self._load_conv_weight(conv3, self.conv3.weight.shape))
            for block, flat in zip(self.blocks, blocks):
                block.weight.copy_(self._load_conv_weight(flat, block.weight.shape))
            self.head.weight.copy_(torch.from_numpy(head.T))

        expected_feat_shape = (self.config.c3, self.config.conv3_out_h, self.config.conv3_out_w)
        probe = torch.zeros(1, self.config.img_c, self.config.img_h, self.config.img_w)
        if tuple(self.forward_features(probe).shape[1:]) != expected_feat_shape:
            raise RuntimeError(f"unexpected feature shape, expected {expected_feat_shape}")
