from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT_OVERRIDE = os.environ.get("PROJECT_ROOT", "").strip()


def find_project_root() -> Path:
    if PROJECT_ROOT_OVERRIDE:
        root = Path(PROJECT_ROOT_OVERRIDE).expanduser().resolve()
        if (root / "src" / "training" / "train_regression.py").exists():
            return root
        raise FileNotFoundError(f"PROJECT_ROOT was set, but src was not found: {root}")

    starts = [
        Path.cwd(),
        *Path.cwd().parents,
        Path("/content"),
        Path("/content/drive/MyDrive"),
        Path("/workspace"),
        Path("/mnt/data"),
    ]
    for start in starts:
        if (start / "src" / "training" / "train_regression.py").exists():
            return start

    for base in [Path("/content"), Path("/content/drive/MyDrive"), Path("/workspace"), Path("/mnt/data")]:
        if not base.exists():
            continue
        for hit in base.rglob("train_regression.py"):
            if hit.as_posix().endswith("src/training/train_regression.py"):
                return hit.parents[2]

    raise FileNotFoundError(
        "Project root not found. Clone or copy the full repository to Colab, "
        "or set os.environ['PROJECT_ROOT'] to the repository path."
    )


PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"PROJECT_ROOT={PROJECT_ROOT}")
if not (PROJECT_ROOT / "data" / "train.csv").exists():
    print("WARNING: data/train.csv was not found under PROJECT_ROOT.")
