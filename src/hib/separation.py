"""Score separation metrics for binary class probability outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from hib.scores import HISTOGRAM_BINS


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def _fraction_above_threshold(values: np.ndarray, threshold: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values > threshold))


def _normalized_histogram(values: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=HISTOGRAM_BINS)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / float(total)


def _hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2) / 2.0))


def score_separation_metrics(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> dict[str, Any]:
    """Compute compact class-conditional score separation metrics."""

    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("positive_scores and negative_scores must both be non-empty")

    positive_mean = float(np.mean(pos))
    negative_mean = float(np.mean(neg))
    positive_median = _quantile(pos, 0.5)
    negative_median = _quantile(neg, 0.5)
    positive_p10 = _quantile(pos, 0.10)
    positive_p50 = _quantile(pos, 0.50)
    positive_p90 = _quantile(pos, 0.90)
    negative_p90 = _quantile(neg, 0.90)
    negative_p95 = _quantile(neg, 0.95)
    negative_p99 = _quantile(neg, 0.99)

    pos_hist = _normalized_histogram(pos)
    neg_hist = _normalized_histogram(neg)
    ks_stat = float(ks_2samp(pos, neg, alternative="two-sided", method="auto").statistic)

    return {
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "mean_gap": positive_mean - negative_mean,
        "positive_median": positive_median,
        "negative_median": negative_median,
        "median_gap": positive_median - negative_median,
        "positive_p10": positive_p10,
        "positive_p50": positive_p50,
        "positive_p90": positive_p90,
        "negative_p90": negative_p90,
        "negative_p95": negative_p95,
        "negative_p99": negative_p99,
        "p50_vs_neg_p95_gap": positive_p50 - negative_p95,
        "p10_vs_neg_p90_gap": positive_p10 - negative_p90,
        "fraction_positives_above_negative_p95": _fraction_above_threshold(pos, negative_p95),
        "fraction_positives_above_negative_p99": _fraction_above_threshold(pos, negative_p99),
        "ks_statistic": ks_stat,
        "histogram_hellinger_distance": _hellinger_distance(pos_hist, neg_hist),
    }
