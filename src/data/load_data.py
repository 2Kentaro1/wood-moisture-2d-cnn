from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import DataConfig
from src.data.spectral_views import make_spectral_views


@dataclass
class SpectralDatasetFrame:
    metadata: pd.DataFrame
    spectra: np.ndarray
    wavenumbers: np.ndarray
    wavelengths: np.ndarray
    spectral_columns: list[str]


def detect_spectral_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        try:
            float(col)
            cols.append(col)
        except ValueError:
            continue
    return cols


def load_spectral_csv(path: str | Path, config: DataConfig | None = None) -> SpectralDatasetFrame:
    config = config or DataConfig()
    df = pd.read_csv(path, encoding=config.encoding)
    spectral_columns = detect_spectral_columns(df)
    if not spectral_columns:
        raise ValueError(f"No numeric wavenumber columns found in {path}")

    wavenumbers = np.asarray([float(c) for c in spectral_columns], dtype=np.float64)
    wavelengths = 1e7 / wavenumbers
    order = np.argsort(wavelengths)

    sorted_columns = [spectral_columns[i] for i in order]
    metadata = df.drop(columns=spectral_columns).copy()
    spectra = df[sorted_columns].to_numpy(dtype=np.float32)
    return SpectralDatasetFrame(
        metadata=metadata,
        spectra=spectra,
        wavenumbers=wavenumbers[order],
        wavelengths=wavelengths[order],
        spectral_columns=sorted_columns,
    )


def load_train_test(root: str | Path = ".", config: DataConfig | None = None) -> tuple[SpectralDatasetFrame, SpectralDatasetFrame]:
    config = config or DataConfig()
    root = Path(root)
    return (
        load_spectral_csv(root / "data" / config.train_csv, config),
        load_spectral_csv(root / "data" / config.test_csv, config),
    )


def build_multiview_input(frame: SpectralDatasetFrame, config: DataConfig | None = None) -> np.ndarray:
    config = config or DataConfig()
    views = make_spectral_views(frame.spectra, config.sg_window_length, config.sg_polyorder)
    return views[:, None, :, :]
