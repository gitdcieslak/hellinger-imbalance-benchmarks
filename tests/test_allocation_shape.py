import numpy as np

from hib.allocation_shape import positive_score_allocation_shape_metrics


def test_positive_score_allocation_shape_concentrated_scores():
    metrics = positive_score_allocation_shape_metrics(np.array([0.9, 0.9, 0.9, 0.9]))

    assert metrics["positive_unique_score_ratio"] == 0.25
    assert metrics["positive_quantization_score"] == 0.75
    assert metrics["positive_effective_score_bins"] == 1.0
    assert metrics["positive_max_bin_mass"] == 1.0
    assert metrics["positive_top_bin_mass"] == 1.0
    assert metrics["positive_score_iqr"] == 0.0


def test_positive_score_allocation_shape_spread_scores():
    metrics = positive_score_allocation_shape_metrics(np.array([0.01, 0.1, 0.3, 0.8]))

    assert metrics["positive_unique_score_ratio"] == 1.0
    assert metrics["positive_quantization_score"] == 0.0
    assert metrics["positive_effective_score_bins"] > 1.0
    assert metrics["positive_score_q10_q90_width"] > metrics["positive_score_iqr"]
