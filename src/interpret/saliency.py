from __future__ import annotations

import torch


def gradient_saliency(
    model: torch.nn.Module,
    x: torch.Tensor,
    task_type: str = "regression",
    target_index: int | None = None,
) -> torch.Tensor:
    model.eval()
    x = x.detach().clone().requires_grad_(True)
    pred = model(x)
    if task_type == "classification":
        if target_index is None:
            target_index = int(pred.argmax(dim=1)[0].item())
        score = pred[:, target_index].sum()
    else:
        score = pred.sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    return x.grad.detach().abs().squeeze(1)

