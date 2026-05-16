"""Score distribution analysis utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


HISTOGRAM_BINS = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
HISTOGRAM_LABELS = [
    "[0.0,0.01)",
    "[0.01,0.05)",
    "[0.05,0.10)",
    "[0.10,0.25)",
    "[0.25,0.50)",
    "[0.50,1.00]",
]
QUANTILES = {
    "p01": 0.01,
    "p05": 0.05,
    "p10": 0.10,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
}


def score_summary(scores: np.ndarray) -> dict[str, Any]:
    """Compute summary statistics for predicted probabilities."""

    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError("scores must not be empty")

    stats: dict[str, Any] = {
        "mean_probability": float(np.mean(arr)),
        "std_probability": float(np.std(arr, ddof=0)),
        "min_probability": float(np.min(arr)),
        "max_probability": float(np.max(arr)),
    }
    for name, q in QUANTILES.items():
        stats[name] = float(np.quantile(arr, q))
    stats["histogram"] = histogram_counts(arr)
    return stats


def histogram_counts(scores: np.ndarray) -> dict[str, int]:
    """Count probabilities in fixed histogram bins."""

    arr = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    counts, _ = np.histogram(arr, bins=HISTOGRAM_BINS)
    return {label: int(count) for label, count in zip(HISTOGRAM_LABELS, counts)}


def class_conditional_score_summaries(y_true: np.ndarray, y_score: np.ndarray) -> list[dict[str, Any]]:
    """Compute score summaries separately for negative and positive classes."""

    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)
    output: list[dict[str, Any]] = []
    for class_label in (0, 1):
        class_scores = y_score_arr[y_true_arr == class_label]
        if class_scores.size == 0:
            continue
        summary = score_summary(class_scores)
        summary["class_label"] = int(class_label)
        output.append(summary)
    return output
