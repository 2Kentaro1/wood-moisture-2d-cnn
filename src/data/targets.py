from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config.settings import DataConfig


@dataclass(frozen=True)
class WoodSpeciesInfo:
    species: str
    class_label: str
    structure_label: str
    note: str = ""


SPECIES_INFO: dict[str, WoodSpeciesInfo] = {
    "イチョウ": WoodSpeciesInfo("イチョウ", "softwood", "tracheid", "gymnosperm; no vessels"),
    "ウエンジ": WoodSpeciesInfo("ウエンジ", "hardwood", "ring_porous_like", "large vessels; high density"),
    "ウォールナット": WoodSpeciesInfo("ウォールナット", "hardwood", "diffuse_porous", "relatively uniform vessels"),
    "クリ": WoodSpeciesInfo("クリ", "hardwood", "ring_porous", "clear large vessels"),
    "スプルース": WoodSpeciesInfo("スプルース", "softwood", "tracheid", "uniform softwood"),
    "チェリー": WoodSpeciesInfo("チェリー", "hardwood", "diffuse_porous", "medium structure"),
    "トチ": WoodSpeciesInfo("トチ", "hardwood", "diffuse_porous", "uniform vessels"),
    "ナラ": WoodSpeciesInfo("ナラ", "hardwood", "ring_porous", "strong large vessels"),
    "ヒノキ": WoodSpeciesInfo("ヒノキ", "softwood", "tracheid", "Chamaecyparis"),
    "ベイスギ": WoodSpeciesInfo("ベイスギ", "softwood", "tracheid", "tracheids plus strong extractives"),
    "ベイマツ": WoodSpeciesInfo("ベイマツ", "softwood", "tracheid", "density and resin effects"),
    "ホワイトオーク": WoodSpeciesInfo("ホワイトオーク", "hardwood", "ring_porous", "oak group; strong vessels"),
    "米ヒバ": WoodSpeciesInfo("米ヒバ", "softwood", "tracheid", "hiba group"),
}

WOODTYPE_LABELS = ["hardwood", "softwood"]
WOOD_STRUCTURE_LABELS = ["tracheid", "ring_porous", "diffuse_porous", "ring_porous_like"]


@dataclass
class TargetInfo:
    y: np.ndarray
    task_type: str
    num_outputs: int
    labels: list[str] | None = None


def _normalize_species_name(name: object) -> str:
    return str(name).strip().replace(" ", "").replace("　", "")


def _species_info(name: object) -> WoodSpeciesInfo:
    key = _normalize_species_name(name)
    if key not in SPECIES_INFO:
        known = ", ".join(SPECIES_INFO)
        raise ValueError(f"Unknown species '{key}'. Add it to SPECIES_INFO. Known species: {known}")
    return SPECIES_INFO[key]


def add_wood_metadata(meta: pd.DataFrame, config: DataConfig | None = None) -> pd.DataFrame:
    config = config or DataConfig()
    out = meta.copy()
    infos = out[config.species_name_col].map(_species_info)
    out["wood_class"] = infos.map(lambda x: x.class_label)
    out["wood_structure"] = infos.map(lambda x: x.structure_label)
    out["wood_structure_note"] = infos.map(lambda x: x.note)
    return out


def make_species_classification(meta: pd.DataFrame, config: DataConfig | None = None) -> TargetInfo:
    config = config or DataConfig()
    encoder = LabelEncoder()
    y = encoder.fit_transform(meta[config.species_col].astype(str))
    return TargetInfo(y=y.astype(np.int64), task_type="classification", num_outputs=len(encoder.classes_), labels=encoder.classes_.tolist())


def make_mc_regression(meta: pd.DataFrame, config: DataConfig | None = None) -> TargetInfo:
    config = config or DataConfig()
    return TargetInfo(y=meta[config.mc_col].to_numpy(dtype=np.float32), task_type="regression", num_outputs=1)


def infer_softwood(name: object) -> int:
    return int(_species_info(name).class_label == "softwood")


def make_softwood_hardwood(meta: pd.DataFrame, config: DataConfig | None = None) -> TargetInfo:
    config = config or DataConfig()
    y = meta[config.species_name_col].map(infer_softwood).to_numpy(dtype=np.int64)
    return TargetInfo(y=y, task_type="classification", num_outputs=2, labels=WOODTYPE_LABELS)


def infer_wood_structure(name: object) -> int:
    label = _species_info(name).structure_label
    return WOOD_STRUCTURE_LABELS.index(label)


def make_wood_structure(meta: pd.DataFrame, config: DataConfig | None = None) -> TargetInfo:
    config = config or DataConfig()
    y = meta[config.species_name_col].map(infer_wood_structure).to_numpy(dtype=np.int64)
    return TargetInfo(y=y, task_type="classification", num_outputs=len(WOOD_STRUCTURE_LABELS), labels=WOOD_STRUCTURE_LABELS)


def make_species_index_norm(meta: pd.DataFrame, config: DataConfig | None = None) -> TargetInfo:
    config = config or DataConfig()
    values = np.zeros(len(meta), dtype=np.float32)
    for _, idx in meta.groupby(config.species_col).groups.items():
        order = np.arange(len(idx), dtype=np.float32)
        denom = max(len(idx) - 1, 1)
        values[np.asarray(idx)] = order / denom
    return TargetInfo(y=values, task_type="regression", num_outputs=1)


def make_species_mc_norm(meta: pd.DataFrame, config: DataConfig | None = None) -> TargetInfo:
    config = config or DataConfig()
    mc = meta[config.mc_col].to_numpy(dtype=np.float32)
    values = np.zeros(len(meta), dtype=np.float32)
    for _, idx in meta.groupby(config.species_col).groups.items():
        ids = np.asarray(idx)
        mc_start = float(mc[ids].max())
        mc_end = float(mc[ids].min())
        denom = mc_start - mc_end
        values[ids] = 0.0 if abs(denom) < 1e-8 else (mc[ids] - mc_end) / denom
    return TargetInfo(y=values, task_type="regression", num_outputs=1)


def build_target(task: str, meta: pd.DataFrame, config: DataConfig | None = None) -> TargetInfo:
    makers = {
        "mc": make_mc_regression,
        "species": make_species_classification,
        "woodtype": make_softwood_hardwood,
        "wood_structure": make_wood_structure,
        "index_norm": make_species_index_norm,
        "mc_norm": make_species_mc_norm,
    }
    if task not in makers:
        raise ValueError(f"Unknown task: {task}. Available: {sorted(makers)}")
    return makers[task](meta, config)
