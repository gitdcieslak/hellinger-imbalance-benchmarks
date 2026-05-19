import numpy as np

from hib.occupancy import compute_occupancy_metrics
from hib.occupancy_plots import plot_occupancy_artifacts
from hib.reporting import summarize_occupancy_records
from hib.runner import run_prediction_space_occupancy_legacy


def test_occupancy_metrics_basic_properties():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.01, 0.02, 0.03, 0.2, 0.6, 0.8])
    metrics = compute_occupancy_metrics(y, s, thresholds=[0.5, 0.25, 0.1, 0.05, 0.01])
    assert 0.0 <= metrics["posterior_sparsity_index"] <= 1.0
    assert 0.0 <= metrics["quantization_score"] <= 1.0
    assert len(metrics["threshold_occupancy"]) == 5


def test_occupancy_runner_and_summary_and_plots(monkeypatch, tmp_path):
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
    monkeypatch.setattr("hib.runner.CURATED_LEGACY_HDDT_DATASET_REGISTRY", tiny_registry)

    records = run_prediction_space_occupancy_legacy(
        dataset_ids=["tiny"],
        model_ids=["cart"],
        extracted_dir=extracted,
        n_repeats=1,
        test_size=0.5,
        split_seed=1,
    )
    assert len(records) == 1
    assert "threshold_occupancy" in records[0]["metrics"]
    summary = summarize_occupancy_records(records)
    assert int(summary.loc[0, "n_runs"]) == 1

    plots = plot_occupancy_artifacts(records, tmp_path / "plots")
    assert len(plots) >= 8
    assert all(path.exists() for path in plots)
