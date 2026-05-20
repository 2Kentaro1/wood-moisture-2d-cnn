from __future__ import annotations

from pathlib import Path


OUTPUT_SUBDIRS = [
    "models",
    "oof",
    "predictions",
    "embeddings",
    "saliency",
    "integrated_gradients",
    "occlusion",
    "heatmaps",
    "figures",
    "metrics",
    "temp",
]


def ensure_output_dirs(output_dir: str | Path = "outputs") -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_SUBDIRS:
        (output / name).mkdir(parents=True, exist_ok=True)
    return output
