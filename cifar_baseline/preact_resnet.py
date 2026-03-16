from __future__ import annotations

import torch
import torch.nn as nn


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + shortcut


class PreActResNet(nn.Module):
    def __init__(
        self,
        *,
        block_counts: tuple[int, int, int, int] = (2, 2, 2, 2),
        widths: tuple[int, int, int, int] = (64, 128, 256, 512),
        num_classes: int = 10,
    ):
        super().__init__()
        if len(block_counts) != 4 or len(widths) != 4:
            raise ValueError("Expected four residual stages")

        self.in_channels = widths[0]
        self.stem = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(widths[0], block_counts[0], stride=1)
        self.layer2 = self._make_layer(widths[1], block_counts[1], stride=2)
        self.layer3 = self._make_layer(widths[2], block_counts[2], stride=2)
        self.layer4 = self._make_layer(widths[3], block_counts[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(widths[3] * PreActBasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_stride in strides:
            blocks.append(PreActBasicBlock(self.in_channels, out_channels, stride=block_stride))
            self.in_channels = out_channels * PreActBasicBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def PreActResNet18(*, num_classes: int = 10) -> PreActResNet:
    return PreActResNet(block_counts=(2, 2, 2, 2), num_classes=num_classes)
