from __future__ import annotations

import argparse
import logging
import os

from src.data.load_data import build_multiview_input, load_train_test
from src.data.targets import build_target
from src.training.cv import group_kfold_indices
from src.training.trainer import train_cv
from src.utils.seed import seed_everything


def default_output_dir() -> str:
    drive_dir = "/content/drive/MyDrive/wood-moisture-2d-cnn-outputs"
    if os.path.isdir("/content/drive/MyDrive"):
        return os.environ.get("OUTPUT_DIR", drive_dir)
    return os.environ.get("OUTPUT_DIR", "outputs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mc", choices=["mc", "index_norm", "mc_norm"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output-dir", default=default_output_dir())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    seed_everything(42)
    train_frame, _ = load_train_test(".")
    x = build_multiview_input(train_frame)
    target = build_target(args.task, train_frame.metadata)
    groups = train_frame.metadata["species number"].to_numpy()
    folds = group_kfold_indices(groups, args.n_splits)
    train_cv(
        x,
        target.y,
        groups,
        folds,
        train_frame.metadata,
        args.task,
        target.task_type,
        target.num_outputs,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
