from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.cnn_blocks import ConvBNGELU, ResidualBlock


@dataclass
class CNNOutput:
    logits: torch.Tensor
    embedding: torch.Tensor


class MultiViewCNN2D(nn.Module):
    """Multi-kernel CNN for input shape (batch, 1, views=6, wavelengths)."""

    def __init__(
        self,
        num_outputs: int = 1,
        task_type: str = "regression",
        base_channels: int = 32,
        embedding_dim: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.branch_local_wavelength = ConvBNGELU(1, base_channels, (1, 9), dropout)
        self.branch_view_local = ConvBNGELU(1, base_channels, (3, 9), dropout)
        self.branch_all_views = ConvBNGELU(1, base_channels, (6, 15), dropout)
        merged_channels = base_channels * 3
        self.fuse = nn.Sequential(
            ConvBNGELU(merged_channels, merged_channels, (3, 7), dropout),
            ResidualBlock(merged_channels, dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.embedding = nn.Sequential(
            nn.Linear(merged_channels, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
        )
        self.head = nn.Linear(embedding_dim, num_outputs)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor | CNNOutput:
        features = torch.cat(
            [
                self.branch_local_wavelength(x),
                self.branch_view_local(x),
                self.branch_all_views(x),
            ],
            dim=1,
        )
        pooled = self.fuse(features)
        embedding = self.embedding(pooled)
        logits = self.head(embedding)
        if self.task_type == "regression":
            logits = logits.squeeze(-1)
        if return_embedding:
            return CNNOutput(logits=logits, embedding=embedding)
        return logits

