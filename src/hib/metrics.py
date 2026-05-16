"""Metric computation for binary imbalance benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_class_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Return positive-class scores from a fitted binary classifier."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.shape[1] == 1:
            return probabilities[:, 0]
        return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Compute the core binary imbalance metrics."""

    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0))),
    }


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute core metrics for a fitted model."""

    y_pred = model.predict(X)
    y_score = positive_class_scores(model, X)
    return compute_binary_metrics(y, y_pred, y_score)
