"""Shape diagnostics for positive-score allocation."""

from __future__ import annotations

import numpy as np

from hib.allocation import gini_coefficient
from hib.scores import HISTOGRAM_BINS


def positive_score_allocation_shape_metrics(scores: np.ndarray) -> dict[str, float]:
    """Compute allocation-shape metrics over positive-class scores only."""

    arr = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    if arr.size == 0:
        raise ValueError("positive scores must not be empty")

    rounded = np.round(arr, 6)
    unique_ratio = float(np.unique(rounded).size / float(arr.size))
    counts, _ = np.histogram(arr, bins=HISTOGRAM_BINS)
    mass = counts.astype(float) / float(max(1, counts.sum()))
    positive_mass = mass[mass > 0.0]
    entropy = float(-np.sum(positive_mass * np.log(positive_mass))) if positive_mass.size else 0.0
    q10, q25, q75, q90 = np.quantile(arr, [0.10, 0.25, 0.75, 0.90])

    return {
        "positive_unique_score_ratio": unique_ratio,
        "positive_quantization_score": float(1.0 - unique_ratio),
        "positive_histogram_entropy": entropy,
        "positive_effective_score_bins": float(np.exp(entropy)),
        "positive_max_bin_mass": float(np.max(mass)) if mass.size else 0.0,
        "positive_top_bin_mass": float(mass[-1]) if mass.size else 0.0,
        "positive_score_iqr": float(q75 - q25),
        "positive_score_q10_q90_width": float(q90 - q10),
        "positive_score_gini_or_concentration_index": gini_coefficient(arr),
    }
