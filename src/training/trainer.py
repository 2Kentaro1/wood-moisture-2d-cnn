from __future__ import annotations

import logging
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.preprocessing import standardize_train_valid
from src.models.cnn2d_multiview import MultiViewCNN2D
from src.models.losses import get_loss, prepare_target
from src.utils.io import save_json, save_table
from src.utils.metrics import classification_metrics, regression_metrics
from src.utils.paths import ensure_output_dirs

LOGGER = logging.getLogger(__name__)


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, object]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class SpectraDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None = None) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


@dataclass
class FoldResult:
    fold: int
    best_score: float
    model_path: Path
    metrics: dict[str, float]


def _fold_temp_paths(output_dir: Path, task: str, fold: int) -> dict[str, Path]:
    temp_dir = output_dir / "temp" / task
    temp_dir.mkdir(parents=True, exist_ok=True)
    return {
        "checkpoint": temp_dir / f"fold{fold}_checkpoint.pt",
        "predictions": temp_dir / f"fold{fold}_valid_predictions.npz",
        "metrics": temp_dir / f"fold{fold}_metrics.json",
    }


def _load_completed_fold(output_dir: Path, task: str, fold: int) -> tuple[FoldResult, np.ndarray, np.ndarray] | None:
    model_path = output_dir / "models" / f"{task}_fold{fold}.pt"
    paths = _fold_temp_paths(output_dir, task, fold)
    if not (model_path.exists() and paths["predictions"].exists() and paths["metrics"].exists()):
        return None
    arr = np.load(paths["predictions"])
    with paths["metrics"].open("r", encoding="utf-8") as f:
        payload = json.load(f)
    LOGGER.info("skip completed fold: task=%s fold=%s", task, fold)
    return (
        FoldResult(
            fold=fold,
            best_score=float(payload["best_score"]),
            model_path=model_path,
            metrics={k: float(v) for k, v in payload["metrics"].items()},
        ),
        arr["pred"],
        arr["embedding"],
    )


def _load_completed_task(output_dir: Path, task: str) -> list[FoldResult] | None:
    metrics_path = output_dir / "metrics" / f"{task}_metrics.json"
    oof_path = output_dir / "oof" / f"{task}_oof.csv"
    embeddings_path = output_dir / "embeddings" / f"{task}_embeddings.parquet"
    best_path = output_dir / "models" / f"{task}_best.pt"
    if not (metrics_path.exists() and oof_path.exists() and embeddings_path.exists() and best_path.exists()):
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    LOGGER.info("skip completed task: %s", task)
    return [
        FoldResult(
            fold=int(item["fold"]),
            best_score=float(item.get("best_score", 0.0)),
            model_path=Path(item["model_path"]),
            metrics={k: float(v) for k, v in item["metrics"].items()},
        )
        for item in payload["folds"]
    ]


def _score_for_early_stop(y_true: np.ndarray, pred: np.ndarray, task_type: str) -> float:
    if task_type == "regression":
        return regression_metrics(y_true, pred)["rmse"]
    return -classification_metrics(y_true, pred)["macro_f1"]


def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            out = model(x.to(device), return_embedding=True)
            pred = out.logits
            if task_type == "classification":
                pred = torch.softmax(pred, dim=1)
            preds.append(pred.detach().cpu().numpy())
            embeddings.append(out.embedding.detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(embeddings)


def train_one_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    fold: int,
    task: str,
    task_type: str,
    num_outputs: int,
    output_dir: str | Path,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    amp: bool = True,
    device: str = "cuda",
    num_workers: int = 2,
) -> tuple[FoldResult, np.ndarray, np.ndarray]:
    output_dir = ensure_output_dirs(output_dir)
    completed = _load_completed_fold(output_dir, task, fold)
    if completed is not None:
        return completed

    temp_paths = _fold_temp_paths(output_dir, task, fold)
    device_obj = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    x_train, x_valid, mean, std = standardize_train_valid(x[train_idx], x[valid_idx])
    y_train, y_valid = y[train_idx], y[valid_idx]

    train_loader = DataLoader(SpectraDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(SpectraDataset(x_valid, y_valid), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model = MultiViewCNN2D(num_outputs=num_outputs, task_type=task_type).to(device_obj)
    criterion = get_loss(task_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(device_obj.type, enabled=amp and device_obj.type == "cuda")

    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    start_epoch = 0
    if temp_paths["checkpoint"].exists():
        ckpt = _torch_load(temp_paths["checkpoint"], map_location=device_obj)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_score = float(ckpt["best_score"])
        best_state = ckpt["best_state_dict"]
        stale = int(ckpt["stale"])
        start_epoch = int(ckpt["next_epoch"])
        mean = ckpt["mean"]
        std = ckpt["std"]
        LOGGER.info("resume checkpoint: task=%s fold=%s epoch=%s", task, fold, start_epoch)

    for epoch in range(start_epoch, epochs):
        model.train()
        losses: list[float] = []
        for xb, yb in tqdm(train_loader, desc=f"{task} fold {fold} epoch {epoch + 1}", leave=False):
            xb = xb.to(device_obj)
            yb = prepare_target(yb.to(device_obj), task_type)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_obj.type, enabled=amp and device_obj.type == "cuda"):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        valid_pred, _ = predict_loader(model, valid_loader, device_obj, task_type)
        score = _score_for_early_stop(y_valid, valid_pred, task_type)
        LOGGER.info("fold=%s epoch=%s loss=%.5f score=%.5f", fold, epoch + 1, np.mean(losses), score)
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        torch.save(
            {
                "next_epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_score": best_score,
                "best_state_dict": best_state,
                "stale": stale,
                "mean": mean,
                "std": std,
                "task": task,
                "task_type": task_type,
                "num_outputs": num_outputs,
            },
            temp_paths["checkpoint"],
        )
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_pred, valid_embeddings = predict_loader(model, valid_loader, device_obj, task_type)
    metrics = regression_metrics(y_valid, valid_pred) if task_type == "regression" else classification_metrics(y_valid, valid_pred)
    model_path = output_dir / "models" / f"{task}_fold{fold}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "task": task,
            "task_type": task_type,
            "num_outputs": num_outputs,
        },
        model_path,
    )
    np.savez_compressed(temp_paths["predictions"], pred=valid_pred.reshape(len(valid_idx), -1), embedding=valid_embeddings)
    save_json(
        {
            "fold": fold,
            "best_score": best_score,
            "metrics": metrics,
            "model_path": str(model_path),
        },
        temp_paths["metrics"],
    )
    return FoldResult(fold=fold, best_score=best_score, model_path=model_path, metrics=metrics), valid_pred, valid_embeddings


def train_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    meta: pd.DataFrame,
    task: str,
    task_type: str,
    num_outputs: int,
    output_dir: str | Path = "outputs",
    **train_kwargs: object,
) -> list[FoldResult]:
    output_dir = ensure_output_dirs(output_dir)
    completed_task = _load_completed_task(output_dir, task)
    if completed_task is not None:
        return completed_task

    oof_pred = np.zeros((len(y), num_outputs if task_type == "classification" else 1), dtype=np.float32)
    embeddings = np.zeros((len(y), 16), dtype=np.float32)
    results: list[FoldResult] = []
    for fold, (train_idx, valid_idx) in enumerate(folds):
        result, pred, emb = train_one_fold(
            x=x,
            y=y,
            train_idx=train_idx,
            valid_idx=valid_idx,
            fold=fold,
            task=task,
            task_type=task_type,
            num_outputs=num_outputs,
            output_dir=output_dir,
            **train_kwargs,
        )
        results.append(result)
        oof_pred[valid_idx] = pred.reshape(len(valid_idx), -1)
        embeddings[valid_idx] = emb

    best = min(results, key=lambda r: r.best_score)
    shutil.copyfile(best.model_path, output_dir / "models" / f"{task}_best.pt")
    pred_cols = [f"pred_{i}" for i in range(oof_pred.shape[1])]
    oof = meta.reset_index(drop=True).copy()
    for i, col in enumerate(pred_cols):
        oof[col] = oof_pred[:, i]
    save_table(oof, output_dir / "oof" / f"{task}_oof.csv")
    save_table(pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])]), output_dir / "embeddings" / f"{task}_embeddings.parquet")
    save_json(
        {
            "folds": [
                {
                    "fold": r.fold,
                    "best_score": r.best_score,
                    "metrics": r.metrics,
                    "model_path": str(r.model_path),
                }
                for r in results
            ]
        },
        output_dir / "metrics" / f"{task}_metrics.json",
    )
    return results


def load_model_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[MultiViewCNN2D, dict[str, object]]:
    ckpt = _torch_load(path, map_location=map_location)
    model = MultiViewCNN2D(num_outputs=int(ckpt["num_outputs"]), task_type=str(ckpt["task_type"]))
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt
