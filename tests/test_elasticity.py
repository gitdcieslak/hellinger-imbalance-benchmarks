import pandas as pd
import pytest

from hib.elasticity import (
    compute_threshold_elasticity_intervals,
    summarize_threshold_elasticity,
    threshold_order,
)
from hib.reporting import threshold_elasticity_summary_to_markdown


def _summary_frame() -> pd.DataFrame:
    thresholds = [0.5, 0.25, 0.1, 0.05, 0.01]
    rows = []
    recall = [0.1, 0.2, 0.4, 0.55, 0.8]
    precision = [0.6, 0.5, 0.4, 0.3, 0.2]
    f1 = [0.17, 0.29, 0.4, 0.39, 0.32]
    ba = [0.52, 0.56, 0.62, 0.65, 0.7]
    for idx, threshold in enumerate(thresholds):
        rows.append(
            {
                "dataset_id": "boundary",
                "model_id": "hddt",
                "threshold": threshold,
                "recall_mean": recall[idx],
                "precision_mean": precision[idx],
                "f1_mean": f1[idx],
                "balanced_accuracy_mean": ba[idx],
            }
        )
    return pd.DataFrame(rows)


def test_threshold_ordering_is_correct():
    assert threshold_order() == [0.5, 0.25, 0.1, 0.05, 0.01]


def test_interval_deltas_and_signs():
    intervals = compute_threshold_elasticity_intervals(_summary_frame())
    first = intervals.iloc[0]
    assert first["threshold_start"] == 0.5
    assert first["threshold_end"] == 0.25
    assert first["threshold_delta"] == -0.25
    assert first["recall_delta"] > 0
    assert first["precision_delta"] < 0
    assert first["recall_elasticity"] > 0
    assert first["precision_elasticity"] < 0


def test_max_jump_and_precision_drop_identified():
    intervals = compute_threshold_elasticity_intervals(_summary_frame())
    summary = summarize_threshold_elasticity(intervals)
    row = summary.iloc[0]
    assert row["max_recall_jump"] > 0
    assert row["max_precision_drop"] < 0
    assert row["threshold_of_max_recall_jump"]


def test_missing_thresholds_raise_clear_error():
    frame = _summary_frame()
    frame = frame[frame["threshold"] != 0.05]
    with pytest.raises(ValueError, match="missing thresholds"):
        compute_threshold_elasticity_intervals(frame)


def test_markdown_generation_works():
    intervals = compute_threshold_elasticity_intervals(_summary_frame())
    summary = summarize_threshold_elasticity(intervals)
    text = threshold_elasticity_summary_to_markdown(summary)
    assert "Threshold Elasticity Summary" in text
    assert "boundary" in text
