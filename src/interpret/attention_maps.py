from __future__ import annotations

import numpy as np
import torch


def normalize_importance(importance: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(importance, dtype=np.float32)
    arr = np.abs(arr)
    return arr / (arr.max() + eps)


def batch_mean_importance(maps: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = maps.detach().cpu().numpy() if isinstance(maps, torch.Tensor) else np.asarray(maps)
    return normalize_importance(arr.mean(axis=0))

