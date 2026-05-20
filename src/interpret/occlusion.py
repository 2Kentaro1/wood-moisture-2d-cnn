from __future__ import annotations

import numpy as np
import torch

from src.config.settings import OCCLUSION_BANDS, VIEW_NAMES


def _score(model: torch.nn.Module, x: torch.Tensor, task_type: str, target_index: int | None) -> torch.Tensor:
    pred = model(x)
    if task_type == "classification":
        probs = torch.softmax(pred, dim=1)
        if target_index is None:
            target_index = int(probs.argmax(dim=1)[0].item())
        return probs[:, target_index]
    return pred.reshape(-1)


@torch.no_grad()
def band_occlusion(
    model: torch.nn.Module,
    x: torch.Tensor,
    wavelengths: np.ndarray,
    task_type: str = "regression",
    bands: list[tuple[float, float]] | None = None,
    target_index: int | None = None,
    fill_value: float = 0.0,
) -> dict[str, float]:
    model.eval()
    bands = bands or OCCLUSION_BANDS
    base = _score(model, x, task_type, target_index)
    results: dict[str, float] = {}
    for low, high in bands:
        mask = (wavelengths >= low) & (wavelengths < high)
        xo = x.clone()
        xo[:, :, :, mask] = fill_value
        delta = base - _score(model, xo, task_type, target_index)
        results[f"{low:g}-{high:g}"] = float(delta.mean().detach().cpu())
    return results


@torch.no_grad()
def channel_occlusion(
    model: torch.nn.Module,
    x: torch.Tensor,
    task_type: str = "regression",
    target_index: int | None = None,
    fill_value: float = 0.0,
) -> dict[str, float]:
    model.eval()
    base = _score(model, x, task_type, target_index)
    results: dict[str, float] = {}
    for i, name in enumerate(VIEW_NAMES):
        xo = x.clone()
        xo[:, :, i, :] = fill_value
        delta = base - _score(model, xo, task_type, target_index)
        results[name] = float(delta.mean().detach().cpu())
    return results

