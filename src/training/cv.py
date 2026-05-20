from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def group_kfold_indices(groups: np.ndarray, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("GroupKFold requires at least two unique groups.")
    splitter = GroupKFold(n_splits=n_splits)
    dummy_x = np.zeros(len(groups))
    return [(trn, val) for trn, val in splitter.split(dummy_x, groups=groups)]

