"""Prediction-space occupancy metrics and artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np

from hib.scores import HISTOGRAM_BINS, HISTOGRAM_LABELS


def _entropy_from_counts(counts: np.ndarray) -> float:
    mass = counts.astype(float)
    total = float(mass.sum())
    if total <= 0:
        return 0.0
    probs = mass / total
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log(probs)))


def _ecdf_points(values: np.ndarray, max_points: int = 200) -> list[dict[str, float]]:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return []
    y = np.arange(1, arr.size + 1, dtype=float) / float(arr.size)
    if arr.size > max_points:
        idx = np.linspace(0, arr.size - 1, num=max_points, dtype=int)
        arr = arr[idx]
        y = y[idx]
    return [{"x": float(x), "y": float(v)} for x, v in zip(arr, y, strict=False)]


def compute_occupancy_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    """Compute occupancy metrics and compact score distribution artifacts."""

    y = np.asarray(y_true, dtype=int)
    s = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
    if y.size == 0 or s.size == 0 or y.size != s.size:
        raise ValueError("y_true and y_score must be non-empty and aligned")

    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("both positive and negative class scores are required")

    all_counts, _ = np.histogram(s, bins=HISTOGRAM_BINS)
    pos_counts, _ = np.histogram(pos, bins=HISTOGRAM_BINS)
    neg_counts, _ = np.histogram(neg, bins=HISTOGRAM_BINS)
    n_bins = len(HISTOGRAM_BINS) - 1

    all_entropy = _entropy_from_counts(all_counts)
    pos_entropy = _entropy_from_counts(pos_counts)
    neg_entropy = _entropy_from_counts(neg_counts)
    occupied_bins = int(np.sum(all_counts > 0))
    posterior_sparsity_index = float(1.0 - (occupied_bins / float(n_bins)))

    q10_pos, q90_pos = np.quantile(pos, [0.1, 0.9])
    q10_neg, q90_neg = np.quantile(neg, [0.1, 0.9])
    pos_width = float(max(1e-9, q90_pos - q10_pos))
    neg_width = float(max(1e-9, q90_neg - q10_neg))
    compression_ratio = float(pos_width / neg_width)

    unique_ratio = float(np.unique(np.round(s, 6)).size / float(s.size))
    quantization_score = float(max(0.0, min(1.0, 1.0 - unique_ratio)))

    occupancy_traj: list[dict[str, float]] = []
    pos_survival: list[float] = []
    for threshold in thresholds:
        th = float(threshold)
        pos_above = float(np.mean(pos >= th))
        neg_above = float(np.mean(neg >= th))
        occupancy_traj.append(
            {
                "threshold": th,
                "positive_occupancy": pos_above,
                "negative_occupancy": neg_above,
            }
        )
        pos_survival.append(pos_above)

    threshold_occupancy_persistence = float(np.mean(pos_survival))
    occupancy_density_ratio = float(pos_entropy / max(1e-9, neg_entropy))

    return {
        "score_mean": float(np.mean(s)),
        "score_std": float(np.std(s, ddof=0)),
        "score_min": float(np.min(s)),
        "score_max": float(np.max(s)),
        "occupancy_entropy": all_entropy,
        "minority_occupancy_entropy": pos_entropy,
        "majority_occupancy_entropy": neg_entropy,
        "occupied_bin_count": occupied_bins,
        "posterior_sparsity_index": posterior_sparsity_index,
        "occupancy_density_ratio": occupancy_density_ratio,
        "threshold_occupancy_persistence": threshold_occupancy_persistence,
        "minority_occupancy_compression_ratio": compression_ratio,
        "quantization_score": quantization_score,
        "histogram_counts": {
            "all": {label: int(value) for label, value in zip(HISTOGRAM_LABELS, all_counts, strict=False)},
            "positive": {label: int(value) for label, value in zip(HISTOGRAM_LABELS, pos_counts, strict=False)},
            "negative": {label: int(value) for label, value in zip(HISTOGRAM_LABELS, neg_counts, strict=False)},
        },
        "ecdf": {
            "positive": _ecdf_points(pos),
            "negative": _ecdf_points(neg),
        },
        "threshold_occupancy": occupancy_traj,
    }
