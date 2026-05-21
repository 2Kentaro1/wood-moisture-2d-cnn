from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.config.settings import DataConfig
from src.data.load_data import build_multiview_input, load_train_test
from src.data.targets import add_wood_metadata
from src.training.cv import group_kfold_indices
from src.training.trainer import SpectraDataset, load_model_checkpoint, predict_loader, train_one_fold
from src.utils.io import save_json, save_table
from src.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)
EXPERIMENT_NAME = "T12_cnn_soft_routing"
TASKS = ["woodtype", "wood_structure"]
WOODTYPE_LABELS = ["hardwood", "softwood"]
WOOD_STRUCTURE_LABELS = ["softwood", "ring_porous", "diffuse_porous", "ring_porous_like"]
STRUCTURE_ALIAS = {
    "tracheid": "softwood",
    "ring_porous": "ring_porous",
    "diffuse_porous": "diffuse_porous",
    "ring_porous_like": "ring_porous_like",
}


@dataclass(frozen=True)
class SoftRoutingResult:
    task: str
    labels: list[str]
    train_probs: pd.DataFrame
    test_probs: pd.DataFrame
    train_embeddings: pd.DataFrame
    test_embeddings: pd.DataFrame


def default_output_dir() -> str:
    drive_dir = f"/content/drive/MyDrive/wood-moisture-2d-cnn-outputs/{EXPERIMENT_NAME}"
    if os.path.isdir("/content/drive/MyDrive"):
        return os.environ.get("T12_OUTPUT_DIR", drive_dir)
    return os.environ.get("T12_OUTPUT_DIR", f"outputs/{EXPERIMENT_NAME}")


def ensure_t12_dirs(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    for name in ["models", "oof", "test", "metrics", "figures", "features", "embeddings", "temp"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    return output_dir


def prepare_target(task: str, train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DataFrame, pd.DataFrame, LabelEncoder]:
    train_meta = add_wood_metadata(train_meta)
    # test may contain species that are intentionally absent from the train
    # label table. Keep test metadata as-is; only train needs hard labels.
    test_meta = test_meta.copy()
    if task == "woodtype":
        label_col = "wood_class"
        labels = WOODTYPE_LABELS
    elif task == "wood_structure":
        label_col = "wood_structure_soft"
        train_meta[label_col] = train_meta["wood_structure"].map(STRUCTURE_ALIAS)
        if "wood_structure" in test_meta.columns:
            test_meta[label_col] = test_meta["wood_structure"].map(STRUCTURE_ALIAS)
        labels = WOOD_STRUCTURE_LABELS
    else:
        raise ValueError(f"Unsupported soft routing task: {task}")

    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(labels, dtype=object)
    label_to_id = {label: i for i, label in enumerate(labels)}
    mapped = train_meta[label_col].astype(str).map(label_to_id)
    if mapped.isna().any():
        raise ValueError(f"Unknown labels found for {task}: {sorted(train_meta[label_col].astype(str).unique())}")
    y = mapped.to_numpy(dtype=np.int64)
    return y, labels, train_meta, test_meta, encoder


def standardize_with_checkpoint(x: np.ndarray, ckpt: dict[str, object]) -> np.ndarray:
    mean = np.asarray(ckpt["mean"])
    std = np.asarray(ckpt["std"])
    return ((x - mean) / (std + 1e-8)).astype(np.float32)


def predict_fold_test(model_path: Path, x_test: np.ndarray, batch_size: int, device: torch.device, num_workers: int) -> tuple[np.ndarray, np.ndarray]:
    model, ckpt = load_model_checkpoint(model_path, map_location=device)
    model = model.to(device)
    x_std = standardize_with_checkpoint(x_test, ckpt)
    loader = DataLoader(SpectraDataset(x_std), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return predict_loader(model, loader, device, task_type="classification")


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-8, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def sample_columns(meta: pd.DataFrame, config: DataConfig) -> list[str]:
    return [c for c in [config.sample_col, config.species_col, config.species_name_col] if c in meta.columns]


def probability_frame(
    meta: pd.DataFrame,
    probs: np.ndarray,
    labels: list[str],
    task: str,
    config: DataConfig,
    y_true_labels: np.ndarray | None = None,
) -> pd.DataFrame:
    out = meta[sample_columns(meta, config)].reset_index(drop=True).copy()
    if y_true_labels is not None:
        out[f"true_{task}"] = y_true_labels
    pred_idx = probs.argmax(axis=1)
    out[f"pred_{task}"] = np.asarray(labels)[pred_idx]
    for i, label in enumerate(labels):
        out[f"prob_{task}_{label}"] = probs[:, i]
    out[f"maxprob_{task}"] = probs.max(axis=1)
    out[f"entropy_{task}"] = entropy(probs)
    return out


def embedding_frame(meta: pd.DataFrame, embeddings: np.ndarray, config: DataConfig) -> pd.DataFrame:
    out = meta[sample_columns(meta, config)].reset_index(drop=True).copy()
    for i in range(embeddings.shape[1]):
        out[f"emb_{i}"] = embeddings[:, i]
    return out


def save_label_encoder(encoder: LabelEncoder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(encoder, f)


def classification_diagnostics(y_true: np.ndarray, probs: np.ndarray, labels: list[str], species: pd.Series) -> dict[str, object]:
    pred = probs.argmax(axis=1)
    report = classification_report(y_true, pred, target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=np.arange(len(labels)))
    species_acc = (
        pd.DataFrame({"species": species.astype(str).to_numpy(), "correct": pred == y_true})
        .groupby("species", as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        "logloss": float(log_loss(y_true, probs, labels=np.arange(len(labels)))),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "species_accuracy": species_acc.to_dict(orient="records"),
    }


def plot_confusion(y_true: np.ndarray, probs: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, probs.argmax(axis=1), labels=np.arange(len(labels)))
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), max(3.5, len(labels))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_probability_distribution(probs: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    df = pd.DataFrame(probs, columns=labels).melt(var_name="class", value_name="probability")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=df, x="probability", hue="class", bins=30, element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_species_probability_boxplot(frame: pd.DataFrame, task: str, labels: list[str], config: DataConfig, path: Path) -> None:
    if config.species_name_col not in frame.columns:
        return
    prob_cols = [f"prob_{task}_{label}" for label in labels]
    df = frame[[config.species_name_col, *prob_cols]].melt(id_vars=config.species_name_col, var_name="class", value_name="probability")
    df["class"] = df["class"].str.replace(f"prob_{task}_", "", regex=False)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    sns.boxplot(data=df, x=config.species_name_col, y="probability", hue="class", ax=ax)
    ax.set_title(f"{task}: predicted probability by species")
    ax.set_xlabel("Species")
    ax.set_ylabel("Probability")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_true_class_probability(frame: pd.DataFrame, task: str, labels: list[str], path: Path) -> None:
    true_col = f"true_{task}"
    prob_cols = [f"prob_{task}_{label}" for label in labels]
    df = frame[[true_col, *prob_cols]].melt(id_vars=true_col, var_name="class", value_name="probability")
    df["class"] = df["class"].str.replace(f"prob_{task}_", "", regex=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=df, x=true_col, y="probability", hue="class", cut=0, inner="box", ax=ax)
    ax.set_title(f"{task}: probability distribution by true class")
    ax.set_xlabel("True class")
    ax.set_ylabel("Probability")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_train_test_probability(train_probs: np.ndarray, test_probs: np.ndarray, labels: list[str], task: str, path: Path) -> None:
    train_df = pd.DataFrame(train_probs, columns=labels).assign(split="train_oof")
    test_df = pd.DataFrame(test_probs, columns=labels).assign(split="test")
    df = pd.concat([train_df, test_df], ignore_index=True).melt(id_vars="split", var_name="class", value_name="probability")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="class", y="probability", hue="split", ax=ax)
    ax.set_title(f"{task}: train/test probability distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Probability")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_uncertainty(probs: np.ndarray, task: str, split: str, path: Path) -> None:
    uncertainty = 1.0 - probs.max(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.histplot(uncertainty, bins=30, ax=ax)
    ax.set_title(f"{task}: uncertainty histogram ({split})")
    ax.set_xlabel("1 - max probability")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_scalar_distribution(values: np.ndarray, title: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.histplot(values, bins=30, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_embedding_pca(embeddings: np.ndarray, labels: np.ndarray, class_names: list[str], task: str, path: Path) -> None:
    if len(embeddings) < 3:
        return
    xy = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    df = pd.DataFrame({"PC1": xy[:, 0], "PC2": xy[:, 1], "class": np.asarray(class_names)[labels]})
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="class", s=28, alpha=0.85, ax=ax)
    ax.set_title(f"{task}: CNN embedding PCA")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_embedding_umap(embeddings: np.ndarray, labels: np.ndarray, class_names: list[str], task: str, path: Path) -> None:
    try:
        from umap import UMAP
    except ImportError:
        return
    if len(embeddings) < 5:
        return
    xy = UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    df = pd.DataFrame({"UMAP1": xy[:, 0], "UMAP2": xy[:, 1], "class": np.asarray(class_names)[labels]})
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="class", s=28, alpha=0.85, ax=ax)
    ax.set_title(f"{task}: CNN embedding UMAP")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_probability_summaries(train_frame: pd.DataFrame, test_frame: pd.DataFrame, task: str, labels: list[str], config: DataConfig, out_dir: Path) -> None:
    prob_cols = [f"prob_{task}_{label}" for label in labels]
    species_split_rows = []
    for split, frame in [("train_oof", train_frame), ("test", test_frame)]:
        if config.species_name_col not in frame.columns:
            continue
        by_species = frame.groupby(config.species_name_col)[prob_cols].agg(["mean", "std"])
        by_species.columns = ["_".join(col).strip() for col in by_species.columns.to_flat_index()]
        by_species = by_species.reset_index()
        by_species.insert(0, "split", split)
        species_split_rows.append(by_species)
    if species_split_rows:
        save_table(pd.concat(species_split_rows, ignore_index=True), out_dir / "metrics" / f"{task}_probability_by_species.csv")
    split_rows = []
    for split, frame in [("train_oof", train_frame), ("test", test_frame)]:
        stats = frame[prob_cols + [f"maxprob_{task}", f"entropy_{task}"]].agg(["mean", "std"]).T.reset_index()
        stats.columns = ["feature", "mean", "std"]
        stats.insert(0, "split", split)
        split_rows.append(stats)
    save_table(pd.concat(split_rows, ignore_index=True), out_dir / "metrics" / f"{task}_train_test_probability_stats.csv")


def run_task(
    task: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    n_splits: int,
    num_workers: int,
    device: str,
    overwrite: bool,
) -> SoftRoutingResult:
    config = DataConfig()
    train_out = output_dir / "oof" / f"train_{task}_oof_probs.csv"
    test_out = output_dir / "test" / f"test_{task}_probs.csv"
    train_emb_out = output_dir / "embeddings" / f"train_cnn_embedding_{task}.csv"
    test_emb_out = output_dir / "embeddings" / f"test_cnn_embedding_{task}.csv"
    if all(p.exists() for p in [train_out, test_out, train_emb_out, test_emb_out]) and not overwrite:
        LOGGER.info("skip completed T12 task: %s", task)
        labels_path = output_dir / "metrics" / f"{task}_labels.json"
        labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"] if labels_path.exists() else []
        return SoftRoutingResult(task, labels, pd.read_csv(train_out), pd.read_csv(test_out), pd.read_csv(train_emb_out), pd.read_csv(test_emb_out))

    y, labels, train_meta_labeled, test_meta_labeled, encoder = prepare_target(task, train_meta, test_meta)
    groups = train_meta_labeled[config.species_col].to_numpy()
    folds = group_kfold_indices(groups, n_splits)
    num_outputs = len(labels)
    oof_probs = np.zeros((len(x_train), num_outputs), dtype=np.float32)
    oof_embeddings = np.zeros((len(x_train), 16), dtype=np.float32)
    test_fold_probs = []
    test_fold_embeddings = []
    fold_payloads = []

    for fold, (trn_idx, val_idx) in enumerate(folds):
        result, pred, emb = train_one_fold(
            x=x_train,
            y=y,
            train_idx=trn_idx,
            valid_idx=val_idx,
            fold=fold,
            task=task,
            task_type="classification",
            num_outputs=num_outputs,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        oof_probs[val_idx] = pred.reshape(len(val_idx), -1)
        oof_embeddings[val_idx] = emb
        fold_probs, fold_embeddings = predict_fold_test(result.model_path, x_test, batch_size, torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"), num_workers)
        test_fold_probs.append(fold_probs)
        test_fold_embeddings.append(fold_embeddings)
        fold_payloads.append({"fold": result.fold, "best_score": result.best_score, "metrics": result.metrics, "model_path": str(result.model_path)})

    test_probs = np.mean(test_fold_probs, axis=0)
    test_embeddings = np.mean(test_fold_embeddings, axis=0)
    best_model = min(fold_payloads, key=lambda x: float(x["best_score"]))["model_path"]
    shutil.copyfile(best_model, output_dir / "models" / f"{task}_best.pt")

    true_labels = np.asarray(labels)[y]
    train_probs = probability_frame(train_meta_labeled, oof_probs, labels, task, config, true_labels)
    test_probs_frame = probability_frame(test_meta_labeled, test_probs, labels, task, config)
    train_embeddings = embedding_frame(train_meta_labeled, oof_embeddings, config)
    test_embeddings = embedding_frame(test_meta_labeled, test_embeddings, config)

    save_table(train_probs, train_out)
    save_table(test_probs_frame, test_out)
    save_table(train_embeddings, train_emb_out)
    save_table(test_embeddings, test_emb_out)
    save_label_encoder(encoder, output_dir / "metrics" / f"{task}_label_encoder.pkl")
    save_json({"labels": labels}, output_dir / "metrics" / f"{task}_labels.json")

    metrics = classification_diagnostics(y, oof_probs, labels, train_meta_labeled[config.species_col])
    metrics["folds"] = fold_payloads
    save_json(metrics, output_dir / "metrics" / f"{task}_metrics.json")
    save_table(pd.DataFrame(metrics["classification_report"]).T.reset_index().rename(columns={"index": "class"}), output_dir / "metrics" / f"{task}_classification_report.csv")
    save_table(pd.DataFrame(metrics["species_accuracy"]), output_dir / "metrics" / f"{task}_species_accuracy.csv")
    save_probability_summaries(train_probs, test_probs_frame, task, labels, config, output_dir)

    figures = output_dir / "figures"
    plot_confusion(y, oof_probs, labels, f"{task}: OOF confusion matrix", figures / f"{task}_confusion_matrix.png")
    plot_probability_distribution(oof_probs, labels, f"{task}: OOF class probability distribution", figures / f"{task}_probability_distribution_train.png")
    plot_probability_distribution(test_probs, labels, f"{task}: test class probability distribution", figures / f"{task}_probability_distribution_test.png")
    plot_species_probability_boxplot(train_probs, task, labels, config, figures / f"{task}_species_probability_boxplot.png")
    plot_true_class_probability(train_probs, task, labels, figures / f"{task}_true_class_probability_violin.png")
    plot_train_test_probability(oof_probs, test_probs, labels, task, figures / f"{task}_train_test_probability_comparison.png")
    plot_uncertainty(oof_probs, task, "train_oof", figures / f"{task}_uncertainty_train.png")
    plot_uncertainty(test_probs, task, "test", figures / f"{task}_uncertainty_test.png")
    plot_scalar_distribution(oof_probs.max(axis=1), f"{task}: max probability distribution (train OOF)", "max probability", figures / f"{task}_max_probability_train.png")
    plot_scalar_distribution(test_probs.max(axis=1), f"{task}: max probability distribution (test)", "max probability", figures / f"{task}_max_probability_test.png")
    plot_scalar_distribution(entropy(oof_probs), f"{task}: entropy distribution (train OOF)", "entropy", figures / f"{task}_entropy_train.png")
    plot_scalar_distribution(entropy(test_probs), f"{task}: entropy distribution (test)", "entropy", figures / f"{task}_entropy_test.png")
    plot_embedding_pca(oof_embeddings, y, labels, task, figures / f"{task}_embedding_pca.png")
    plot_embedding_umap(oof_embeddings, y, labels, task, figures / f"{task}_embedding_umap.png")

    return SoftRoutingResult(task, labels, train_probs, test_probs_frame, train_embeddings, test_embeddings)


def merge_feature_tables(meta: pd.DataFrame, result_frames: list[pd.DataFrame], output_path: Path) -> None:
    config = DataConfig()
    merged = meta.reset_index(drop=True).copy()
    if config.sample_col not in merged.columns:
        merged.insert(0, config.sample_col, np.arange(len(merged)))
    for frame in result_frames:
        keep = [config.sample_col, *[c for c in frame.columns if c.startswith("prob_") or c.startswith("entropy_") or c.startswith("maxprob_")]]
        merged = merged.merge(frame[keep], on=config.sample_col, how="left")
    save_table(merged, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=TASKS, choices=TASKS)
    parser.add_argument("--output-dir", default=default_output_dir())
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    seed_everything(42)
    output_dir = ensure_t12_dirs(args.output_dir)
    train_frame, test_frame = load_train_test(".")
    x_train = build_multiview_input(train_frame)
    x_test = build_multiview_input(test_frame)

    results = [
        run_task(task, x_train, x_test, train_frame.metadata, test_frame.metadata, output_dir, args.epochs, args.batch_size, args.n_splits, args.num_workers, args.device, args.overwrite)
        for task in args.tasks
    ]
    merge_feature_tables(train_frame.metadata, [r.train_probs for r in results], output_dir / "features" / "cnn_soft_routing_features_train.csv")
    merge_feature_tables(test_frame.metadata, [r.test_probs for r in results], output_dir / "features" / "cnn_soft_routing_features_test.csv")
    print(f"saved T12 outputs: {output_dir}")


if __name__ == "__main__":
    main()
