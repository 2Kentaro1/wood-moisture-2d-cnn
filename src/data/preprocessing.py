from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def snv(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + eps)


def savitzky_golay_derivative(
    x: np.ndarray,
    deriv: int,
    window_length: int = 21,
    polyorder: int = 3,
) -> np.ndarray:
    return savgol_filter(
        x,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        axis=1,
        mode="interp",
    )


def standardize_train_valid(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 3), keepdims=True)
    std = x_train.std(axis=(0, 3), keepdims=True)
    return (x_train - mean) / (std + eps), (x_valid - mean) / (std + eps), mean, std


def apply_standardization(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - mean) / (std + eps)
