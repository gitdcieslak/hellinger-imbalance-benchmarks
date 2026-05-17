import pandas as pd
import pytest

from hib.threshold_plots import (
    load_threshold_summary_csv,
    plot_threshold_trajectories,
    threshold_descending_values,
    trajectory_plot_paths,
)


def _summary_frame() -> pd.DataFrame:
    thresholds = [0.5, 0.25, 0.1, 0.05, 0.01]
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        rows.append(
            {
                "dataset_id": "boundary",
                "source_group": "imbalanced",
                "model_id": "hddt",
                "threshold": threshold,
                "precision_mean": 0.2,
                "precision_std": 0.01,
                "recall_mean": 0.8 - threshold,
                "recall_std": 0.01,
            }
        )
        rows.append(
            {
                "dataset_id": "boundary",
                "source_group": "imbalanced",
                "model_id": "cart",
                "threshold": threshold,
                "precision_mean": 0.1,
                "precision_std": 0.01,
                "recall_mean": 0.2,
                "recall_std": 0.01,
            }
        )
    rows.append(
        {
            "dataset_id": "cam",
            "source_group": "imbalanced",
            "model_id": "xgboost",
            "threshold": 0.5,
            "precision_mean": 0.3,
            "precision_std": 0.02,
            "recall_mean": 0.1,
            "recall_std": 0.02,
        }
    )
    return pd.DataFrame(rows)


def test_load_and_filter_trajectory_data(tmp_path):
    frame = _summary_frame()
    path = tmp_path / "summary.csv"
    frame.to_csv(path, index=False)
    loaded = load_threshold_summary_csv(path)
    assert sorted(loaded["dataset_id"].unique().tolist()) == ["boundary", "cam"]


def test_trajectory_filename_is_deterministic(tmp_path):
    png, svg = trajectory_plot_paths(tmp_path, "legacy", "boundary")
    assert png.name == "legacy_boundary_precision_recall_trajectory.png"
    assert svg.name == "legacy_boundary_precision_recall_trajectory.svg"


def test_threshold_order_descending():
    frame = _summary_frame()
    ordered = threshold_descending_values(frame)
    assert ordered[:5] == [0.5, 0.25, 0.1, 0.05, 0.01]


def test_plot_one_dataset_outputs_png_svg(tmp_path):
    frame = _summary_frame()
    created = plot_threshold_trajectories(
        summary=frame,
        source="legacy",
        datasets=["boundary"],
        output_dir=tmp_path,
        annotate_thresholds=True,
    )
    assert len(created) == 2
    assert all(path.exists() for path in created)


def test_synthetic_naming_works(tmp_path):
    frame = _summary_frame().rename(columns={"dataset_id": "skew_ratio"})
    created = plot_threshold_trajectories(
        summary=frame,
        source="synthetic",
        datasets=["boundary"],
        output_dir=tmp_path,
        annotate_thresholds=False,
    )
    assert created[0].name == "synthetic_skew_boundary_precision_recall_trajectory.png"


def test_missing_required_columns_raise_clear_error(tmp_path):
    frame = _summary_frame().drop(columns=["precision_mean"])
    with pytest.raises(ValueError, match="missing required trajectory columns"):
        plot_threshold_trajectories(summary=frame, output_dir=tmp_path)
