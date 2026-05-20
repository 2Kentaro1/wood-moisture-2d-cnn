from __future__ import annotations

import torch
from torch import nn


def get_loss(task_type: str) -> nn.Module:
    if task_type == "regression":
        return nn.SmoothL1Loss()
    if task_type == "classification":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown task_type: {task_type}")


def prepare_target(y: torch.Tensor, task_type: str) -> torch.Tensor:
    return y.float() if task_type == "regression" else y.long()

