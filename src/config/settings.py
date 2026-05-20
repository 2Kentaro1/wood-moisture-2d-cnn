from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


VIEW_NAMES = ["raw", "snv", "raw_sg1", "snv_sg1", "raw_sg2", "snv_sg2"]
OCCLUSION_BANDS = [(1000, 1300), (1300, 1600), (1600, 1800), (1800, 2000), (2000, 2300), (2300, 2500)]
MOISTURE_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120), (120, 200), (200, float("inf"))]


@dataclass
class PathsConfig:
    root: Path = Path(".")
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    def resolve(self) -> "PathsConfig":
        self.root = self.root.resolve()
        self.data_dir = (self.root / self.data_dir).resolve()
        self.output_dir = (self.root / self.output_dir).resolve()
        return self


@dataclass
class DataConfig:
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    encoding: str = "cp932"
    sample_col: str = "sample number"
    species_col: str = "species number"
    species_name_col: str = "樹種"
    mc_col: str = "含水率"
    sg_window_length: int = 21
    sg_polyorder: int = 3


@dataclass
class ModelConfig:
    in_channels: int = 1
    views: int = 6
    base_channels: int = 32
    embedding_dim: int = 16
    dropout: float = 0.2
    task_type: str = "regression"
    num_outputs: int = 1


@dataclass
class TrainConfig:
    task: str = "mc"
    n_splits: int = 5
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    num_workers: int = 2
    seed: int = 42
    amp: bool = True
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
