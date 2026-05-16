"""Threshold sweep utilities for binary classifiers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score


DEFAULT_THRESHOLDS = [0.01, 0.05, 0.10, 0.25, 0.50]
THRESHOLD_METRICS = ["precision", "recall", "f1", "balanced_accuracy"]


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute binary classification metrics at a probability threshold."""

    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict[str, float]]:
    """Compute threshold-aware metrics for each configured threshold."""

    threshold_values = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    return [
        {
            "threshold": float(threshold),
            "metrics": compute_threshold_metrics(y_true, y_score, float(threshold)),
        }
        for threshold in threshold_values
    ]
