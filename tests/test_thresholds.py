import json

from hib.reporting import summarize_threshold_records, summarize_threshold_jsonl
from hib.runner import run_threshold_sweep_suite, write_jsonl
from hib.thresholds import compute_threshold_metrics


def test_threshold_metrics_computed_correctly():
    y_true = [0, 0, 1, 1]
    y_score = [0.01, 0.40, 0.20, 0.90]

    metrics = compute_threshold_metrics(y_true, y_score, threshold=0.25)

    assert metrics == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "balanced_accuracy": 0.5,
    }


def test_threshold_sweep_output_generated(tmp_path):
    config = {
        "experiment_id": "test_threshold_sweep",
        "skew_ratios": [1],
        "seeds": [0],
        "separation": 1.5,
        "minority_count": 8,
        "n_features": 4,
        "noise": 1.0,
        "test_size": 0.25,
        "n_repeats": 5,
        "split_seed": 2,
        "models": ["cart"],
        "thresholds": [0.1, 0.5],
    }
    output_path = write_jsonl(
        run_threshold_sweep_suite(config),
        tmp_path / "threshold_sweep.jsonl",
    )
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 10
    assert {record["threshold"] for record in records} == {0.1, 0.5}
    assert all(record["model_id"] == "cart" for record in records)
    assert {record["repeat_id"] for record in records} == {0, 1, 2, 3, 4}


def test_threshold_summaries_aggregate_correctly(tmp_path):
    records = [
        {
            "skew_ratio": "10:1",
            "model_id": "cart",
            "threshold": 0.1,
            "metrics": {
                "precision": 0.25,
                "recall": 1.0,
                "f1": 0.4,
                "balanced_accuracy": 0.75,
            },
        },
        {
            "skew_ratio": "10:1",
            "model_id": "cart",
            "threshold": 0.1,
            "metrics": {
                "precision": 0.75,
                "recall": 0.5,
                "f1": 0.6,
                "balanced_accuracy": 0.70,
            },
        },
    ]

    summary = summarize_threshold_records(records)
    input_path = write_jsonl(records, tmp_path / "threshold_sweep.jsonl")
    csv_path, markdown_path = summarize_threshold_jsonl(
        input_path,
        tmp_path / "threshold_summary.csv",
        tmp_path / "threshold_summary.md",
    )

    assert list(summary.columns) == [
        "skew_ratio",
        "model_id",
        "threshold",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "n_runs",
    ]
    assert summary.loc[0, "precision_mean"] == 0.5
    assert summary.loc[0, "n_runs"] == 2
    assert csv_path.exists()
    assert markdown_path.exists()
    assert "## Skew Ratio 10:1" in markdown_path.read_text(encoding="utf-8")
