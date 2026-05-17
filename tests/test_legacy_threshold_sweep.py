from pathlib import Path

from hib.reporting import legacy_threshold_summary_to_markdown, summarize_legacy_threshold_records
from hib.runner import run_legacy_threshold_sweep


def test_legacy_threshold_sweep_tiny_fixture(monkeypatch, tmp_path):
    extracted = tmp_path / "extracted"
    path = extracted / "tiny.data"
    path.parent.mkdir(parents=True)
    path.write_text("1,a,yes\n2,b,no\n3,a,yes\n4,b,no\n", encoding="utf-8")

    tiny_registry = {
        "tiny": {
            "dataset_id": "tiny",
            "relative_path": "tiny.data",
            "target_column": "2",
            "positive_class": "yes",
            "task_type": "binary",
            "source_group": "balanced",
            "n_rows": 4,
            "n_columns": 3,
            "imbalance_ratio": 1.0,
            "notes": "",
        }
    }
    monkeypatch.setattr("hib.runner.CURATED_LEGACY_HDDT_DATASET_REGISTRY", tiny_registry)

    records = run_legacy_threshold_sweep(
        dataset_ids=["tiny"],
        model_ids=["cart"],
        extracted_dir=extracted,
        n_repeats=1,
        test_size=0.5,
        split_seed=2,
        thresholds=[0.01, 0.05, 0.10, 0.25, 0.50],
    )

    assert len(records) == 5
    assert {record["threshold"] for record in records} == {0.01, 0.05, 0.1, 0.25, 0.5}
    assert all(record["dataset_id"] == "tiny" for record in records)
    assert all(record["source_group"] == "balanced" for record in records)
    assert all("repeat_id" in record and "split_id" in record for record in records)
    assert all(set(record["metrics"]).issuperset({"precision", "recall", "f1", "balanced_accuracy"}) for record in records)


def test_legacy_threshold_summary_includes_n_runs_and_markdown(tmp_path):
    records = [
        {
            "dataset_id": "tiny",
            "source_group": "balanced",
            "model_id": "cart",
            "threshold": 0.5,
            "metrics": {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "balanced_accuracy": 0.75},
        },
        {
            "dataset_id": "tiny",
            "source_group": "balanced",
            "model_id": "cart",
            "threshold": 0.5,
            "metrics": {"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "balanced_accuracy": 0.75},
        },
    ]
    summary = summarize_legacy_threshold_records(records)
    assert int(summary.loc[0, "n_runs"]) == 2

    markdown = legacy_threshold_summary_to_markdown(summary)
    assert "Legacy HDDT Threshold Sweep Summary" in markdown
    assert "Dataset tiny" in markdown

    md_path = tmp_path / "legacy_threshold.md"
    md_path.write_text(markdown, encoding="utf-8")
    assert md_path.exists()
