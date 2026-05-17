import pandas as pd
import pytest

from hib.threshold_plots import (
    load_threshold_summary_csv,
    ordered_model_ids,
    plot_metric_vs_threshold,
    plot_threshold_response,
)


def _sample_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "boundary",
                "source_group": "imbalanced",
                "model_id": "hddt",
                "threshold": 0.5,
                "recall_mean": 0.1,
                "recall_std": 0.01,
                "f1_mean": 0.11,
                "f1_std": 0.01,
                "precision_mean": 0.2,
                "precision_std": 0.01,
                "balanced_accuracy_mean": 0.55,
                "balanced_accuracy_std": 0.01,
                "n_runs": 5,
            },
            {
                "dataset_id": "boundary",
                "source_group": "imbalanced",
                "model_id": "cart",
                "threshold": 0.5,
                "recall_mean": 0.2,
                "recall_std": 0.02,
                "f1_mean": 0.12,
                "f1_std": 0.01,
                "precision_mean": 0.1,
                "precision_std": 0.01,
                "balanced_accuracy_mean": 0.51,
                "balanced_accuracy_std": 0.01,
                "n_runs": 5,
            },
            {
                "dataset_id": "boundary",
                "source_group": "imbalanced",
                "model_id": "hddt",
                "threshold": 0.01,
                "recall_mean": 0.8,
                "recall_std": 0.05,
                "f1_mean": 0.25,
                "f1_std": 0.03,
                "precision_mean": 0.12,
                "precision_std": 0.02,
                "balanced_accuracy_mean": 0.7,
                "balanced_accuracy_std": 0.04,
                "n_runs": 5,
            },
            {
                "dataset_id": "boundary",
                "source_group": "imbalanced",
                "model_id": "cart",
                "threshold": 0.01,
                "recall_mean": 0.2,
                "recall_std": 0.02,
                "f1_mean": 0.12,
                "f1_std": 0.01,
                "precision_mean": 0.1,
                "precision_std": 0.01,
                "balanced_accuracy_mean": 0.51,
                "balanced_accuracy_std": 0.01,
                "n_runs": 5,
            },
            {
                "dataset_id": "cam",
                "source_group": "imbalanced",
                "model_id": "xgboost",
                "threshold": 0.5,
                "recall_mean": 0.03,
                "recall_std": 0.01,
                "f1_mean": 0.05,
                "f1_std": 0.01,
                "precision_mean": 0.2,
                "precision_std": 0.05,
                "balanced_accuracy_mean": 0.5,
                "balanced_accuracy_std": 0.01,
                "n_runs": 5,
            },
        ]
    )


def test_load_threshold_summary_csv(tmp_path):
    frame = _sample_summary()
    path = tmp_path / "summary.csv"
    frame.to_csv(path, index=False)
    loaded = load_threshold_summary_csv(path)
    assert not loaded.empty
    assert "dataset_id" in loaded.columns


def test_model_ordering_is_stable():
    ordered = ordered_model_ids(["xgboost", "custom", "cart", "hddt"])
    assert ordered == ["cart", "hddt", "xgboost", "custom"]


def test_plot_one_metric_for_one_dataset(tmp_path):
    frame = _sample_summary()
    dataset_frame = frame[frame["dataset_id"] == "boundary"]
    created = plot_metric_vs_threshold(
        dataset_frame=dataset_frame,
        metric="recall",
        dataset_id="boundary",
        source="legacy",
        output_dir=tmp_path,
        error_bands=True,
    )
    assert len(created) == 2
    assert all(path.exists() for path in created)
    assert created[0].name == "legacy_boundary_recall_vs_threshold.png"


def test_plot_multiple_metrics_and_datasets(tmp_path):
    frame = _sample_summary()
    created = plot_threshold_response(
        summary=frame,
        source="legacy",
        datasets=["boundary"],
        metrics=["recall", "f1"],
        output_dir=tmp_path,
    )
    assert len(created) == 4
    assert all(path.exists() for path in created)


def test_missing_metric_raises_clear_error(tmp_path):
    frame = _sample_summary().drop(columns=["precision_mean"])
    with pytest.raises(ValueError, match="metric 'precision' not found"):
        plot_threshold_response(
            summary=frame,
            source="legacy",
            datasets=["boundary"],
            metrics=["precision"],
            output_dir=tmp_path,
        )
