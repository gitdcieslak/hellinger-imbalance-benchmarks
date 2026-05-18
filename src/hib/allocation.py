"""Allocation concentration metrics for predicted scores."""

from __future__ import annotations

import numpy as np

from hib.scores import HISTOGRAM_BINS


def gini_coefficient(scores: np.ndarray) -> float:
    """Compute Gini concentration over non-negative scores."""

    x = np.asarray(scores, dtype=float)
    if x.size == 0:
        raise ValueError("scores must not be empty")
    x = np.clip(x, 0.0, None)
    total = float(np.sum(x))
    if total == 0.0:
        return 0.0
    sorted_x = np.sort(x)
    n = sorted_x.size
    index = np.arange(1, n + 1, dtype=float)
    gini = (2.0 * np.sum(index * sorted_x) / (n * total)) - (n + 1) / n
    return float(max(0.0, min(1.0, gini)))


def allocation_concentration_metrics(scores: np.ndarray) -> dict[str, float]:
    """Compute concentration metrics over positive-class scores."""

    arr = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    if arr.size == 0:
        raise ValueError("scores must not be empty")

    hist_counts, _ = np.histogram(arr, bins=HISTOGRAM_BINS)
    mass = hist_counts.astype(float) / float(hist_counts.sum())
    positive_mass = mass[mass > 0.0]
    entropy = float(-np.sum(positive_mass * np.log(positive_mass))) if positive_mass.size else 0.0
    support = float(np.exp(entropy))

    return {
        "score_mean": float(np.mean(arr)),
        "score_std": float(np.std(arr, ddof=0)),
        "score_min": float(np.min(arr)),
        "score_max": float(np.max(arr)),
        "fraction_scores_below_0_001": float(np.mean(arr < 0.001)),
        "fraction_scores_below_0_005": float(np.mean(arr < 0.005)),
        "fraction_scores_below_0_01": float(np.mean(arr < 0.01)),
        "fraction_scores_below_0_05": float(np.mean(arr < 0.05)),
        "fraction_scores_above_0_50": float(np.mean(arr > 0.50)),
        "histogram_entropy": entropy,
        "effective_support_size": support,
        "gini_coefficient": gini_coefficient(arr),
    }
