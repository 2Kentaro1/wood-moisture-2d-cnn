from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config.settings import DataConfig, OCCLUSION_BANDS, VIEW_NAMES
from src.data.load_data import build_multiview_input, load_train_test
from src.interpret.attention_maps import batch_mean_importance, normalize_importance
from src.interpret.integrated_gradients import integrated_gradients
from src.interpret.occlusion import band_occlusion, channel_occlusion
from src.interpret.run_interpretability import default_output_dir as default_model_output_dir
from src.interpret.run_interpretability import save_importance_table, task_model_path
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
FIXED_BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 100), (100, math.inf)]


def default_moisture_bin_output_dir() -> str:
    drive_dir = "/content/drive/MyDrive/wood-moisture-2d-cnn-outputs-moisture-bins"
    if os.path.isdir("/content/drive/MyDrive"):
        return os.environ.get("MOISTURE_BIN_OUTPUT_DIR", drive_dir)
    return os.environ.get("MOISTURE_BIN_OUTPUT_DIR", "outputs-moisture-bins")


@dataclass(frozen=True)
class MoistureBin:
    name: str
    low: float
    high: float
    indices: np.ndarray
    kind: str = "moisture"


def safe_name(name: str) -> str:
    return name.replace("+", "plus").replace("-", "_").replace(" ", "_").replace("<", "lt").replace(">=", "ge").replace(".", "p")


def make_fixed_bins(mc: np.ndarray) -> list[MoistureBin]:
    bins: list[MoistureBin] = []
    for low, high in FIXED_BINS:
        if math.isinf(high):
            mask = mc >= low
            label = f"{low:g}+"
        else:
            mask = (mc >= low) & (mc < high)
            label = f"{low:g}-{high:g}"
        bins.append(MoistureBin(label, low, high, np.flatnonzero(mask), "moisture"))
    return bins


def make_quantile_bins(mc: np.ndarray, q: int) -> list[MoistureBin]:
    edges = np.unique(np.quantile(mc, np.linspace(0, 1, q + 1)))
    bins: list[MoistureBin] = []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (mc >= low) & (mc <= high) if i == len(edges) - 2 else (mc >= low) & (mc < high)
        label = f"q{i + 1}_{low:.1f}-{high:.1f}"
        bins.append(MoistureBin(label, float(low), float(high), np.flatnonzero(mask), "moisture"))
    return bins


def fsp_bins(mc: np.ndarray, threshold: float = 30.0) -> list[MoistureBin]:
    return [
        MoistureBin(f"FSP_lt_{threshold:g}", float("-inf"), threshold, np.flatnonzero(mc < threshold), "fsp"),
        MoistureBin(f"FSP_ge_{threshold:g}", threshold, float("inf"), np.flatnonzero(mc >= threshold), "fsp"),
    ]


def standardized_subset(x: np.ndarray, indices: np.ndarray, ckpt: dict[str, object], max_samples: int, device: torch.device) -> torch.Tensor:
    chosen = indices[:max_samples]
    xb = x[chosen]
    mean = np.asarray(ckpt["mean"])
    std = np.asarray(ckpt["std"])
    xb = (xb - mean) / (std + 1e-8)
    return torch.tensor(xb, dtype=torch.float32, device=device)


def summarize_top_wavelengths(importance: np.ndarray, wavelengths: np.ndarray, k: int = 8) -> str:
    wave_importance = importance.mean(axis=0)
    top = np.argsort(wave_importance)[-k:][::-1]
    return ", ".join(f"{wavelengths[i]:.0f}nm" for i in top)


def summarize_top_views(importance: np.ndarray, k: int = 3) -> str:
    scores = importance.mean(axis=1)
    top = np.argsort(scores)[-k:][::-1]
    return ", ".join(f"{VIEW_NAMES[i]}({scores[i]:.3f})" for i in top)


def band_importance(importance: np.ndarray, wavelengths: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    wave_importance = importance.mean(axis=0)
    for low, high in OCCLUSION_BANDS:
        mask = (wavelengths >= low) & (wavelengths < high)
        out[f"{low:g}-{high:g}"] = float(wave_importance[mask].mean()) if mask.any() else float("nan")
    return out


def save_bin_result(
    task: str,
    bin_info: MoistureBin,
    saliency_mean: np.ndarray,
    ig_mean: np.ndarray,
    importance: np.ndarray,
    wavelengths: np.ndarray,
    band_scores: dict[str, float],
    channel_scores: dict[str, float],
    output_dir: Path,
    overwrite: bool,
) -> None:
    bin_name = safe_name(bin_info.name)
    base = output_dir / "moisture_bins" / task / bin_name
    base.mkdir(parents=True, exist_ok=True)
    np.save(base / "saliency_mean.npy", saliency_mean)
    np.save(base / "integrated_gradients_mean.npy", ig_mean)
    np.save(base / "combined_importance.npy", importance)
    save_importance_table(saliency_mean, wavelengths, base / "saliency_mean.csv")
    save_importance_table(ig_mean, wavelengths, base / "integrated_gradients_mean.csv")
    save_importance_table(importance, wavelengths, base / "combined_importance.csv")
    (base / "band_occlusion.json").write_text(json.dumps(band_scores, indent=2), encoding="utf-8")
    (base / "channel_occlusion.json").write_text(json.dumps(channel_scores, indent=2), encoding="utf-8")
    save_table(pd.DataFrame({"band": list(band_scores), "prediction_drop": list(band_scores.values())}), base / "band_occlusion.csv")
    save_table(pd.DataFrame({"view": list(channel_scores), "prediction_drop": list(channel_scores.values())}), base / "channel_occlusion.csv")

    title_prefix = f"{task} | MC bin {bin_info.name} | n={len(bin_info.indices)}"
    plot_view_wavelength_heatmap(saliency_mean, wavelengths, f"{title_prefix}: saliency", base / "saliency_heatmap.png", cbar_label="normalized saliency", overwrite=overwrite)
    plot_view_wavelength_heatmap(ig_mean, wavelengths, f"{title_prefix}: integrated gradients", base / "integrated_gradients_heatmap.png", cbar_label="normalized IG", overwrite=overwrite)
    plot_view_wavelength_heatmap(importance, wavelengths, f"{title_prefix}: combined importance", base / "combined_importance_heatmap.png", cbar_label="combined importance", overwrite=overwrite)
    plot_occlusion_bar(band_scores, f"{title_prefix}: wavelength band occlusion", base / "band_occlusion.png", overwrite=overwrite)
    plot_occlusion_bar(channel_scores, f"{title_prefix}: view occlusion", base / "channel_occlusion.png", overwrite=overwrite)


def run_task_bins(
    task: str,
    bins: list[MoistureBin],
    x: np.ndarray,
    wavelengths: np.ndarray,
    model_output_dir: Path,
    output_dir: Path,
    max_samples_per_bin: int,
    ig_steps: int,
    overwrite: bool,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model_path = task_model_path(model_output_dir, task)
    if model_path is None:
        print(f"skip {task}: model not found")
        return {}
    model, ckpt = load_model_checkpoint(model_path, map_location=device)
    model = model.to(device)
    task_type = str(ckpt["task_type"])
    task_results: dict[str, np.ndarray] = {}

    for bin_info in bins:
        if len(bin_info.indices) == 0:
            print(f"skip {task} bin {bin_info.name}: empty")
            continue
        bin_name = safe_name(bin_info.name)
        combined_path = output_dir / "moisture_bins" / task / bin_name / "combined_importance.npy"
        if combined_path.exists() and not overwrite:
            task_results[bin_info.name] = np.load(combined_path)
            print(f"skip {task} bin {bin_info.name}: exists")
            continue

        xb = standardized_subset(x, bin_info.indices, ckpt, max_samples_per_bin, device)
        saliency = gradient_saliency(model, xb, task_type=task_type)
        ig = integrated_gradients(model, xb, steps=ig_steps, task_type=task_type)
        saliency_mean = batch_mean_importance(saliency)
        ig_mean = batch_mean_importance(ig)
        importance = normalize_importance((saliency_mean + ig_mean) / 2.0)
        band_scores = band_occlusion(model, xb, wavelengths, task_type=task_type)
        channel_scores = channel_occlusion(model, xb, task_type=task_type)
        save_bin_result(task, bin_info, saliency_mean, ig_mean, importance, wavelengths, band_scores, channel_scores, output_dir, overwrite)
        task_results[bin_info.name] = importance
        print(f"done {task} bin {bin_info.name}")
    return task_results


def save_difference_maps(
    task: str,
    results: dict[str, np.ndarray],
    bins: list[MoistureBin],
    wavelengths: np.ndarray,
    output_dir: Path,
    overwrite: bool,
) -> list[str]:
    notes: list[str] = []
    if not results:
        return notes
    diff_dir = output_dir / "moisture_bins" / task / "differences"
    diff_dir.mkdir(parents=True, exist_ok=True)

    moisture_keys = [b.name for b in bins if b.kind == "moisture" and b.name in results]
    comparisons = []
    if len(moisture_keys) >= 2:
        comparisons.append((moisture_keys[0], moisture_keys[-1], "low_vs_high"))
    keys = list(results)
    fsp_low = next((k for k in keys if k.startswith("FSP_lt")), None)
    fsp_high = next((k for k in keys if k.startswith("FSP_ge")), None)
    if fsp_low and fsp_high:
        comparisons.append((fsp_low, fsp_high, "fsp_ge30_minus_lt30"))

    for low, high, name in comparisons:
        low_imp = results[low]
        high_imp = results[high]
        diff = high_imp - low_imp
        norm_diff = diff / (np.abs(high_imp) + np.abs(low_imp) + 1e-8)
        ratio = (high_imp + 1e-6) / (low_imp + 1e-6)
        np.save(diff_dir / f"{name}_difference.npy", diff)
        np.save(diff_dir / f"{name}_normalized_difference.npy", norm_diff)
        np.save(diff_dir / f"{name}_ratio.npy", ratio)
        plot_difference_map(high_imp, low_imp, wavelengths, f"{task}: {high} - {low}", diff_dir / f"{name}_difference.png", overwrite=overwrite)
        plot_view_wavelength_heatmap(norm_diff, wavelengths, f"{task}: normalized difference {high} vs {low}", diff_dir / f"{name}_normalized_difference.png", cmap="coolwarm", center=0, cbar_label="normalized difference", overwrite=overwrite)
        plot_view_wavelength_heatmap(np.log2(ratio), wavelengths, f"{task}: log2 ratio {high} / {low}", diff_dir / f"{name}_log2_ratio.png", cmap="coolwarm", center=0, cbar_label="log2 ratio", overwrite=overwrite)
        notes.append(f"- {task} {name}: top changed views {summarize_top_views(np.abs(diff))}; top wavelengths {summarize_top_wavelengths(np.abs(diff), wavelengths)}")
    return notes


def save_bin_task_comparisons(
    all_results: dict[str, dict[str, np.ndarray]],
    bins: list[MoistureBin],
    wavelengths: np.ndarray,
    output_dir: Path,
    overwrite: bool,
) -> None:
    comparison_dir = output_dir / "moisture_bins" / "task_comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for bin_info in bins:
        task_importance = {
            task: results[bin_info.name]
            for task, results in all_results.items()
            if bin_info.name in results
        }
        if len(task_importance) < 2:
            continue
        plot_task_comparison_heatmap(
            task_importance,
            wavelengths,
            comparison_dir / f"{safe_name(bin_info.name)}_task_comparison_heatmap.png",
            title=f"Task comparison in MC bin {bin_info.name}",
            overwrite=overwrite,
        )
        rows = []
        for task, importance in task_importance.items():
            rows.append(
                {
                    "bin": bin_info.name,
                    "task": task,
                    "top_views": summarize_top_views(importance),
                    "top_wavelengths": summarize_top_wavelengths(importance, wavelengths),
                }
            )
        save_table(pd.DataFrame(rows), comparison_dir / f"{safe_name(bin_info.name)}_task_summary.csv")


def write_summary(
    all_results: dict[str, dict[str, np.ndarray]],
    bin_counts: pd.DataFrame,
    wavelengths: np.ndarray,
    diff_notes: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Moisture Bin Interpretability Summary",
        "",
        "## Bin Counts",
        "",
        bin_counts.to_markdown(index=False),
        "",
        "## Top Features By Task And Moisture Bin",
        "",
    ]
    for task, results in all_results.items():
        lines.append(f"### {task}")
        if not results:
            lines.append("- no model or no bin results")
            lines.append("")
            continue
        for bin_name, importance in results.items():
            bands = band_importance(importance, wavelengths)
            top_band = max(bands, key=lambda k: -np.inf if np.isnan(bands[k]) else bands[k])
            lines.append(
                f"- `{bin_name}`: top views {summarize_top_views(importance)}; "
                f"top wavelengths {summarize_top_wavelengths(importance, wavelengths)}; "
                f"top band {top_band} ({bands[top_band]:.3f})"
            )
        lines.append("")

    lines.extend(
        [
            "## Difference Highlights",
            "",
            *(diff_notes or ["- no difference maps generated"]),
            "",
            "## Physical Interpretation Guide",
            "",
            "- Low moisture bins are expected to emphasize bound water, structure exposure, cellulose/lignin related shape, and optical transport.",
            "- High moisture bins are expected to emphasize free water, scattering, optical clearing, and transport effects.",
            "- A strong change around `FSP_lt_30` vs `FSP_ge_30` supports the hypothesis that the CNN switches observation mode near the fiber saturation point.",
            "- Increased reliance on derivative views (`raw_sg1`, `snv_sg1`, `raw_sg2`, `snv_sg2`) suggests phase/shape or structure-resolution cues.",
            "- Increased reliance on `raw` can indicate absolute absorption/scattering level; increased reliance on `snv` can indicate shape-normalized or scatter-corrected information.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", default=default_moisture_bin_output_dir())
    parser.add_argument("--model-output-dir", default=None)
    parser.add_argument("--binning", choices=["fixed", "quantile"], default="fixed")
    parser.add_argument("--quantiles", type=int, default=7)
    parser.add_argument("--fsp-threshold", type=float, default=30.0)
    parser.add_argument("--max-samples-per-bin", type=int, default=96)
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_output_dirs(args.output_dir.strip() if isinstance(args.output_dir, str) and args.output_dir.strip() else default_moisture_bin_output_dir())
    model_output_dir = Path(args.model_output_dir.strip()) if isinstance(args.model_output_dir, str) and args.model_output_dir.strip() else Path(default_model_output_dir())
    print(f"model_output_dir={model_output_dir}")
    print(f"moisture_bin_output_dir={output_dir}")
    config = DataConfig()
    train_frame, _ = load_train_test(".")
    if config.mc_col not in train_frame.metadata.columns:
        raise ValueError("Moisture bin analysis requires train metadata column '含水率'. test.csv has no moisture target.")

    x = build_multiview_input(train_frame)
    mc = train_frame.metadata[config.mc_col].to_numpy(dtype=float)
    bins = make_fixed_bins(mc) if args.binning == "fixed" else make_quantile_bins(mc, args.quantiles)
    bins = [b for b in bins if len(b.indices) > 0]
    bins.extend([b for b in fsp_bins(mc, args.fsp_threshold) if len(b.indices) > 0])
    counts = pd.DataFrame([{"bin": b.name, "kind": b.kind, "low": b.low, "high": b.high, "n_samples": len(b.indices)} for b in bins])
    save_table(counts, output_dir / "moisture_bins" / "bin_counts.csv")
    sample_rows = []
    for b in bins:
        sample_rows.extend({"bin": b.name, "kind": b.kind, "row_index": int(i), "mc": float(mc[i])} for i in b.indices)
    save_table(pd.DataFrame(sample_rows), output_dir / "moisture_bins" / "bin_sample_indices.csv")
    print(counts.to_string(index=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results: dict[str, dict[str, np.ndarray]] = {}
    diff_notes: list[str] = []
    for task in args.tasks:
        results = run_task_bins(
            task,
            bins,
            x,
            train_frame.wavelengths,
            model_output_dir,
            output_dir,
            args.max_samples_per_bin,
            args.ig_steps,
            args.overwrite,
            device,
        )
        all_results[task] = results
        diff_notes.extend(save_difference_maps(task, results, bins, train_frame.wavelengths, output_dir, args.overwrite))

    save_bin_task_comparisons(all_results, bins, train_frame.wavelengths, output_dir, args.overwrite)
    write_summary(all_results, counts, train_frame.wavelengths, diff_notes, output_dir / "moisture_bins" / "summary.md")


if __name__ == "__main__":
    main()
