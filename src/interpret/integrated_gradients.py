from __future__ import annotations

import torch


def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor | None = None,
    steps: int = 32,
    task_type: str = "regression",
    target_index: int | None = None,
) -> torch.Tensor:
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)
    total_grad = torch.zeros_like(x)
    for alpha in torch.linspace(0, 1, steps, device=x.device):
        xi = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        pred = model(xi)
        if task_type == "classification":
            if target_index is None:
                score = pred.gather(1, pred.argmax(dim=1, keepdim=True)).sum()
            else:
                score = pred[:, target_index].sum()
        else:
            score = pred.sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        total_grad += xi.grad.detach()
    avg_grad = total_grad / steps
    return ((x - baseline) * avg_grad).detach().squeeze(1)
