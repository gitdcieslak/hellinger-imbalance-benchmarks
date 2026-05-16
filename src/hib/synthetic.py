"""Synthetic Gaussian skew dataset generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SyntheticSkewConfig:
    """Configuration for a binary Gaussian skew dataset."""

    skew_ratio: int
    seed: int
    separation: float = 2.0
    minority_count: int = 20
    n_features: int = 6
    noise: float = 1.0
    test_size: float = 0.3

    @property
    def majority_count(self) -> int:
        return self.minority_count * self.skew_ratio

    @property
    def dataset_id(self) -> str:
        return f"synthetic-gaussian-skew-{self.skew_ratio}-to-1"


def make_gaussian_skew_dataset(config: SyntheticSkewConfig) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic binary Gaussian dataset for a fixed config."""

    if config.skew_ratio < 1:
        raise ValueError("skew_ratio must be at least 1")
    if config.minority_count < 1:
        raise ValueError("minority_count must be at least 1")
    if config.n_features < 1:
        raise ValueError("n_features must be at least 1")
    if config.noise <= 0:
        raise ValueError("noise must be positive")

    rng = np.random.default_rng(config.seed)
    majority_center = np.zeros(config.n_features)
    minority_center = np.zeros(config.n_features)
    minority_center[0] = config.separation

    majority = rng.normal(
        loc=majority_center,
        scale=config.noise,
        size=(config.majority_count, config.n_features),
    )
    minority = rng.normal(
        loc=minority_center,
        scale=config.noise,
        size=(config.minority_count, config.n_features),
    )

    X = np.vstack([majority, minority])
    y = np.concatenate(
        [np.zeros(config.majority_count, dtype=int), np.ones(config.minority_count, dtype=int)]
    )

    permutation = rng.permutation(y.size)
    return X[permutation], y[permutation]


def make_train_test_split(
    config: SyntheticSkewConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic stratified train/test split."""

    X, y = make_gaussian_skew_dataset(config)
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y,
    )
