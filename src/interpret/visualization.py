from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config.settings import MOISTURE_BINS, VIEW_NAMES
from src.utils.plotting import configure_matplotlib_japanese

configure_matplotlib_japanese()

def _wavelength_ticks(wavelengths: np.ndarray, n_ticks: int = 9) -> tuple[np.ndarray, list[str]]:
    positions = np.linspace(0, len(wavelengths) - 1, min(n_ticks, len(wavelengths)), dtype=int)
    labels = [f"{wavelengths[i]:.0f}" for i in positions]
    return positions, labels


def plot_view_wavelength_heatmap(
    importance: np.ndarray,
    wavelengths: np.ndarray,
    title: str,
    output_path: str | Path,
    view_names: list[str] | None = None,
    cbar_label: str = "normalized importance",
    cmap: str = "mako",
    center: float | None = None,
    overwrite: bool = False,
) -> None:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    view_names = view_names or VIEW_NAMES
    xticks, xlabels = _wavelength_ticks(wavelengths)

    fig, ax = plt.subplots(figsize=(16, 4.8))
    sns.heatmap(
        importance,
        cmap=cmap,
        center=center,
        yticklabels=view_names,
        xticklabels=False,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Spectral view")
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels(xlabels, rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_task_comparison_heatmap(
    task_importance: dict[str, np.ndarray],
    wavelengths: np.ndarray,
    output_path: str | Path,
    title: str = "Task comparison: wavelength importance",
    overwrite: bool = False,
) -> None:
    rows = {task: values.mean(axis=0) for task, values in task_importance.items()}
    df = pd.DataFrame(rows).T
    xticks, xlabels = _wavelength_ticks(wavelengths)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, max(3.2, len(rows) * 0.55)))
    sns.heatmap(df, cmap="viridis", xticklabels=False, cbar_kws={"label": "mean importance across views"}, ax=ax)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Task")
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels(xlabels, rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_difference_map(
    a: np.ndarray,
    b: np.ndarray,
    wavelengths: np.ndarray,
    title: str,
    output_path: str | Path,
    overwrite: bool = False,
) -> None:
    diff = a - b
    lim = float(np.abs(diff).max()) if np.abs(diff).max() > 0 else 1.0
    xticks, xlabels = _wavelength_ticks(wavelengths)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 4.8))
    sns.heatmap(
        diff,
        cmap="coolwarm",
        center=0,
        vmin=-lim,
        vmax=lim,
        yticklabels=VIEW_NAMES,
        xticklabels=False,
        cbar_kws={"label": "importance difference"},
        ax=ax,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Spectral view")
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels(xlabels, rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_occlusion_bar(values: dict[str, float], title: str, output_path: str | Path, overwrite: bool = False) -> None:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(values)
    scores = [values[k] for k in names]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(x=names, y=scores, ax=ax, color="#4C78A8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Occluded region")
    ax.set_ylabel("Prediction drop")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
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
