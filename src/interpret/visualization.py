from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config.settings import MOISTURE_BINS, VIEW_NAMES


def plot_view_wavelength_heatmap(
    importance: np.ndarray,
    wavelengths: np.ndarray,
    title: str,
    output_path: str | Path,
    view_names: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    view_names = view_names or VIEW_NAMES
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(importance, cmap="mako", yticklabels=view_names, xticklabels=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(f"wavelength nm ({wavelengths.min():.0f}-{wavelengths.max():.0f})")
    ax.set_ylabel("view")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_task_comparison_heatmap(task_importance: dict[str, np.ndarray], output_path: str | Path, overwrite: bool = False) -> None:
    rows = {task: values.mean(axis=0) for task, values in task_importance.items()}
    df = pd.DataFrame(rows).T
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, max(3, len(rows) * 0.55)))
    sns.heatmap(df, cmap="viridis", xticklabels=False, ax=ax)
    ax.set_xlabel("wavelength index")
    ax.set_ylabel("task")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_difference_map(a: np.ndarray, b: np.ndarray, name: str, output_path: str | Path, overwrite: bool = False) -> None:
    diff = a - b
    lim = float(np.abs(diff).max())
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(diff, cmap="coolwarm", center=0, vmin=-lim, vmax=lim, yticklabels=VIEW_NAMES, xticklabels=False, ax=ax)
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def aggregate_species_importance(meta: pd.DataFrame, maps: np.ndarray, species_col: str = "species number") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for species, idx in meta.groupby(species_col).groups.items():
        out[str(species)] = maps[np.asarray(idx)].mean(axis=0)
    return out


def aggregate_moisture_bins(meta: pd.DataFrame, maps: np.ndarray, mc_col: str = "含水率") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    mc = meta[mc_col].to_numpy()
    for low, high in MOISTURE_BINS:
        mask = (mc >= low) & (mc < high)
        if mask.any():
            label = f"{low:g}-{high:g}" if np.isfinite(high) else f"{low:g}+"
            out[label] = maps[mask].mean(axis=0)
    return out
