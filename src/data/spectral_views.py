from __future__ import annotations

import numpy as np

from src.config.settings import VIEW_NAMES
from src.data.preprocessing import savitzky_golay_derivative, snv


def make_spectral_views(
    raw: np.ndarray,
    window_length: int = 21,
    polyorder: int = 3,
) -> np.ndarray:
    """Return shape (n_samples, 6, n_wavelengths) in the required view order."""
    raw = np.asarray(raw, dtype=np.float32)
    snv_x = snv(raw)
    raw_sg1 = savitzky_golay_derivative(raw, 1, window_length, polyorder)
    raw_sg2 = savitzky_golay_derivative(raw, 2, window_length, polyorder)
    snv_sg1 = savitzky_golay_derivative(snv_x, 1, window_length, polyorder)
    snv_sg2 = savitzky_golay_derivative(snv_x, 2, window_length, polyorder)
    views = [raw, snv_x, raw_sg1, snv_sg1, raw_sg2, snv_sg2]
    return np.stack(views, axis=1).astype(np.float32)


def view_index(name: str) -> int:
    return VIEW_NAMES.index(name)

