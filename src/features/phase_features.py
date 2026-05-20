from __future__ import annotations

import numpy as np
import pandas as pd


def derivative_energy(view_tensor: np.ndarray, view_names: list[str]) -> pd.DataFrame:
    """Summarize derivative-like views to quantify phase/shape shifts."""
    x = np.asarray(view_tensor)
    if x.ndim == 4:
        x = x[:, 0]
    data = {f"{name}_energy": (x[:, i, :] ** 2).mean(axis=1) for i, name in enumerate(view_names)}
    return pd.DataFrame(data)

