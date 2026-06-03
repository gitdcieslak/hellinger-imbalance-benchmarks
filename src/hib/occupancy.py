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

    # Backward-compatible persistence based on mean positive survival.
    # minority_survival_auc uses trapezoidal integration over sorted thresholds.
    sorted_threshold_order = np.argsort(np.asarray(thresholds, dtype=float))
    sorted_thresholds = np.asarray([thresholds[idx] for idx in sorted_threshold_order], dtype=float)
    sorted_survival = np.asarray([pos_survival[idx] for idx in sorted_threshold_order], dtype=float)
    if sorted_thresholds.size <= 1:
        minority_survival_auc = threshold_occupancy_persistence
    else:
        interval_width = float(sorted_thresholds[-1] - sorted_thresholds[0])
        if interval_width <= 0.0:
            minority_survival_auc = threshold_occupancy_persistence
        else:
            auc_raw = float(np.trapezoid(sorted_survival, sorted_thresholds))
            minority_survival_auc = auc_raw / interval_width

    if sorted_survival.size <= 1:
        drops = np.asarray([], dtype=float)
    else:
        drops = np.maximum(0.0, sorted_survival[:-1] - sorted_survival[1:])
    total_drop = float(np.sum(drops))
    max_drop = float(np.max(drops)) if drops.size else 0.0
    total_variation = (
        float(np.sum(np.abs(np.diff(sorted_survival)))) if sorted_survival.size > 1 else 0.0
    )

    if total_drop > 0.0:
        drop_probs = drops / total_drop
        drop_probs = drop_probs[drop_probs > 0.0]
        drop_entropy = float(-np.sum(drop_probs * np.log(drop_probs)))
        effective_drop_count = float(np.exp(drop_entropy))
        cliffiness = float(max_drop / total_drop)
    else:
        drop_entropy = 0.0
        effective_drop_count = 0.0
        cliffiness = 0.0

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
        "minority_survival_auc": float(minority_survival_auc),
        "minority_survival_max_drop": max_drop,
        "minority_survival_total_variation": total_variation,
        "minority_survival_drop_entropy": drop_entropy,
        "minority_survival_effective_drop_count": effective_drop_count,
        "minority_survival_cliffiness": cliffiness,
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
