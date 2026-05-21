from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config.settings import VIEW_NAMES
from src.data.load_data import build_multiview_input, load_train_test
from src.interpret.attention_maps import batch_mean_importance
from src.interpret.integrated_gradients import integrated_gradients
from src.interpret.occlusion import band_occlusion, channel_occlusion
from src.interpret.saliency import gradient_saliency
from src.interpret.visualization import (
    plot_difference_map,
    plot_occlusion_bar,
    plot_task_comparison_heatmap,
    plot_view_wavelength_heatmap,
)
from src.training.trainer import load_model_checkpoint
from src.utils.io import save_table
from src.utils.paths import ensure_output_dirs


DEFAULT_TASKS = ["mc", "species", "woodtype", "wood_structure", "index_norm", "mc_norm"]


def default_output_dir() -> str:
    drive_dir = "/content/drive/MyDrive/wood-moisture-2d-cnn-outputs"
    if os.path.isdir("/content/drive/MyDrive"):
        return os.environ.get("OUTPUT_DIR", drive_dir)
    return os.environ.get("OUTPUT_DIR", "outputs")


def task_model_path(output_dir: Path, task: str) -> Path | None:
    candidates = [output_dir / "models" / f"{task}_best.pt", output_dir / "models" / f"{task}_fold0.pt"]
    for path in candidates:
        if path.exists():
            return path
    return None


def _standardized_batch(x: np.ndarray, ckpt: dict[str, object], n_samples: int, device: torch.device) -> torch.Tensor:
    xb = x[:n_samples]
    mean = np.asarray(ckpt["mean"])
    std = np.asarray(ckpt["std"])
    xb = (xb - mean) / (std + 1e-8)
    return torch.tensor(xb, dtype=torch.float32, device=device)


def save_importance_table(importance: np.ndarray, wavelengths: np.ndarray, output_path: Path) -> None:
    rows = []
    for view_idx, view_name in enumerate(VIEW_NAMES):
        for wave_idx, wavelength in enumerate(wavelengths):
            rows.append(
                {
                    "view": view_name,
                    "wavelength": float(wavelength),
                    "importance": float(importance[view_idx, wave_idx]),
                }
            )
    save_table(pd.DataFrame(rows), output_path)


def run_task(
    task: str,
    x: np.ndarray,
    wavelengths: np.ndarray,
    output_dir: Path,
    n_samples: int,
    ig_steps: int,
    overwrite: bool,
    device: torch.device,
) -> np.ndarray | None:
    model_path = task_model_path(output_dir, task)
    if model_path is None:
        print(f"skip {task}: model not found")
        return None

    saliency_npy = output_dir / "saliency" / f"{task}_saliency.npy"
    ig_npy = output_dir / "integrated_gradients" / f"{task}_integrated_gradients.npy"
    importance_npy = output_dir / "heatmaps" / f"{task}_importance.npy"

    if importance_npy.exists() and saliency_npy.exists() and ig_npy.exists() and not overwrite:
        print(f"skip {task}: interpretability outputs already exist")
        return np.load(importance_npy)

    model, ckpt = load_model_checkpoint(model_path, map_location=device)
    model = model.to(device)
    task_type = str(ckpt["task_type"])
    xb = _standardized_batch(x, ckpt, n_samples, device)

    saliency = gradient_saliency(model, xb, task_type=task_type)
    saliency_mean = batch_mean_importance(saliency)
    saliency_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(saliency_npy, saliency.detach().cpu().numpy())
    save_importance_table(saliency_mean, wavelengths, output_dir / "saliency" / f"{task}_saliency_mean.csv")

    ig = integrated_gradients(model, xb, steps=ig_steps, task_type=task_type)
    ig_mean = batch_mean_importance(ig)
    ig_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(ig_npy, ig.detach().cpu().numpy())
    save_importance_table(ig_mean, wavelengths, output_dir / "integrated_gradients" / f"{task}_integrated_gradients_mean.csv")

    importance = (saliency_mean + ig_mean) / 2.0
    importance_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(importance_npy, importance)
    save_importance_table(importance, wavelengths, output_dir / "heatmaps" / f"{task}_importance.csv")

    plot_view_wavelength_heatmap(
        saliency_mean,
        wavelengths,
        f"{task}: gradient saliency",
        output_dir / "heatmaps" / f"{task}_saliency_heatmap.png",
        cbar_label="normalized saliency",
        overwrite=overwrite,
    )
    plot_view_wavelength_heatmap(
        ig_mean,
        wavelengths,
        f"{task}: integrated gradients",
        output_dir / "heatmaps" / f"{task}_integrated_gradients_heatmap.png",
        cbar_label="normalized IG attribution",
        overwrite=overwrite,
    )
    plot_view_wavelength_heatmap(
        importance,
        wavelengths,
        f"{task}: combined importance",
        output_dir / "heatmaps" / f"{task}_importance_heatmap.png",
        cbar_label="mean normalized importance",
        overwrite=overwrite,
    )

    band_scores = band_occlusion(model, xb, wavelengths, task_type=task_type)
    channel_scores = channel_occlusion(model, xb, task_type=task_type)
    (output_dir / "occlusion").mkdir(parents=True, exist_ok=True)
    (output_dir / "occlusion" / f"{task}_band_occlusion.json").write_text(json.dumps(band_scores, indent=2), encoding="utf-8")
    (output_dir / "occlusion" / f"{task}_channel_occlusion.json").write_text(json.dumps(channel_scores, indent=2), encoding="utf-8")
    save_table(pd.DataFrame({"band": list(band_scores), "prediction_drop": list(band_scores.values())}), output_dir / "occlusion" / f"{task}_band_occlusion.csv")
    save_table(pd.DataFrame({"view": list(channel_scores), "prediction_drop": list(channel_scores.values())}), output_dir / "occlusion" / f"{task}_channel_occlusion.csv")
    plot_occlusion_bar(band_scores, f"{task}: wavelength band occlusion", output_dir / "figures" / f"{task}_band_occlusion.png", overwrite=overwrite)
    plot_occlusion_bar(channel_scores, f"{task}: view/channel occlusion", output_dir / "figures" / f"{task}_channel_occlusion.png", overwrite=overwrite)

    print(f"done {task}: {model_path}")
    return importance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", default=default_output_dir())
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_output_dirs(args.output_dir)
    train_frame, _ = load_train_test(".")
    x = build_multiview_input(train_frame)
    n_samples = min(args.n_samples, len(x))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_importance: dict[str, np.ndarray] = {}
    for task in args.tasks:
        importance = run_task(task, x, train_frame.wavelengths, output_dir, n_samples, args.ig_steps, args.overwrite, device)
        if importance is not None:
            task_importance[task] = importance

    if task_importance:
        plot_task_comparison_heatmap(task_importance, train_frame.wavelengths, output_dir / "figures" / "task_comparison_heatmap.png", overwrite=True)

    pairs = [
        ("mc", "species", "MC - species", "diff_mc_species.png"),
        ("index_norm", "mc_norm", "index_norm - mc_norm", "diff_index_mc_norm.png"),
        ("species", "woodtype", "species - woodtype", "diff_species_woodtype.png"),
        ("species", "wood_structure", "species - wood_structure", "diff_species_wood_structure.png"),
    ]
    for a, b, title, filename in pairs:
        if a in task_importance and b in task_importance:
            plot_difference_map(task_importance[a], task_importance[b], train_frame.wavelengths, title, output_dir / "figures" / filename, overwrite=True)


if __name__ == "__main__":
    main()
