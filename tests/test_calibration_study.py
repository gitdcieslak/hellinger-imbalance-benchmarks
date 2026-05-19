import numpy as np

from hib.calibration_study import (
    build_calibration_summary_tables,
    calibration_slope_intercept,
    expected_calibration_error,
    reliability_curve_points,
    run_calibration_interaction_legacy,
    write_calibration_study_artifacts,
)


def test_calibration_metrics_shapes_and_bounds():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    ece = expected_calibration_error(y, p, n_bins=4)
    assert 0.0 <= ece <= 1.0
    slope, intercept = calibration_slope_intercept(y, p)
    assert np.isfinite(slope)
    assert np.isfinite(intercept)
    points = reliability_curve_points(y, p, n_bins=4)
    assert len(points) > 0


def test_calibration_study_records_and_artifacts(monkeypatch, tmp_path):
    extracted = tmp_path / "extracted"
    path = extracted / "tiny.data"
    path.parent.mkdir(parents=True)
    path.write_text("1,a,yes\n2,b,no\n3,a,yes\n4,b,no\n5,a,yes\n6,b,no\n", encoding="utf-8")

    tiny_registry = {
        "tiny": {
            "dataset_id": "tiny",
            "relative_path": "tiny.data",
            "target_column": "2",
            "positive_class": "yes",
            "task_type": "binary",
            "source_group": "balanced",
            "n_rows": 6,
            "n_columns": 3,
            "imbalance_ratio": 1.0,
            "notes": "",
        }
    }
    monkeypatch.setattr("hib.calibration_study.LEGACY_HDDT_DATASET_REGISTRY", tiny_registry)

    records = run_calibration_interaction_legacy(
        dataset_ids=["tiny"],
        model_ids=["cart"],
        extracted_dir=extracted,
        n_repeats=1,
        test_size=0.5,
        split_seed=1,
        seed=1,
    )
    assert {row["calibration_method"] for row in records} == {"raw", "platt", "isotonic"}
    assert all("threshold_metrics" in row and "allocation_metrics" in row and "calibration_metrics" in row for row in records)

    summary, persistence, merged = build_calibration_summary_tables(records)
    assert not summary.empty
    assert not persistence.empty
    assert not merged.empty

    outputs = write_calibration_study_artifacts(records, tmp_path / "out")
    assert all(path.exists() for path in outputs.values())
