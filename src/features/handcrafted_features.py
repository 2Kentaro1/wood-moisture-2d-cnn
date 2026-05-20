from __future__ import annotations

import numpy as np
import pandas as pd


def band_means(spectra: np.ndarray, wavelengths: np.ndarray, bands: list[tuple[float, float]]) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for low, high in bands:
        mask = (wavelengths >= low) & (wavelengths < high)
        data[f"mean_{low:g}_{high:g}"] = spectra[:, mask].mean(axis=1) if mask.any() else np.nan
    return pd.DataFrame(data)


def spectral_slopes(spectra: np.ndarray, wavelengths: np.ndarray, bands: list[tuple[float, float]]) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for low, high in bands:
        mask = (wavelengths >= low) & (wavelengths < high)
        if mask.sum() < 2:
            data[f"slope_{low:g}_{high:g}"] = np.nan
            continue
        x = wavelengths[mask]
        y = spectra[:, mask]
        data[f"slope_{low:g}_{high:g}"] = (y[:, -1] - y[:, 0]) / (x[-1] - x[0])
    return pd.DataFrame(data)

