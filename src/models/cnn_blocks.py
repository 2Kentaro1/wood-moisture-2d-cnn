from __future__ import annotations

import torch
from torch import nn


class ConvBNGELU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], dropout: float = 0.0) -> None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNGELU(channels, channels, (3, 5), dropout),
            nn.Conv2d(channels, channels, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))

