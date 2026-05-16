"""Stratified split generation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass(frozen=True)
class StratifiedSplitSpec:
    """Metadata and indexes for one stratified split."""

    repeat_id: int
    split_id: str
    split_seed: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def generate_stratified_repeated_splits(
    y: np.ndarray,
    n_repeats: int = 5,
    test_size: float = 0.5,
    random_seed: int = 0,
) -> list[StratifiedSplitSpec]:
    """Create repeated stratified train/test splits with deterministic seeds."""

    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")

    y_arr = np.asarray(y)
    features = np.zeros((y_arr.size, 1), dtype=float)
    specs: list[StratifiedSplitSpec] = []
    for repeat_id in range(n_repeats):
        split_seed = int(random_seed + repeat_id)
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=split_seed,
        )
        train_idx, test_idx = next(splitter.split(features, y_arr))
        specs.append(
            StratifiedSplitSpec(
                repeat_id=repeat_id,
                split_id=f"repeat_{repeat_id}",
                split_seed=split_seed,
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )
    return specs
