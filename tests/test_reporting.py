from hib.reporting import (
    read_jsonl_records,
    summarize_records,
    write_summary_csv,
    write_summary_markdown,
)
from hib.runner import run_synthetic_suite, write_jsonl


def _records():
    config = {
        "experiment_id": "test_synthetic_smoke",
        "skew_ratios": [1],
        "seeds": [0, 1],
        "separation": 1.5,
        "minority_count": 8,
        "n_features": 4,
        "noise": 1.0,
        "test_size": 0.25,
        "models": ["cart", "random_forest"],
    }
    return run_synthetic_suite(config)


def test_reporting_reads_jsonl_records(tmp_path):
    input_path = write_jsonl(_records(), tmp_path / "synthetic_smoke.jsonl")

    records = read_jsonl_records(input_path)

    assert len(records) == 20
    assert records[0]["skew_ratio"] == "1:1"
    assert "metrics" in records[0]


def test_aggregation_produces_expected_columns():
    summary = summarize_records(_records())

    expected_columns = {
        "skew_ratio",
        "model_id",
        "auroc_mean",
        "auroc_std",
        "average_precision_mean",
        "average_precision_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "brier_score_mean",
        "brier_score_std",
        "n_runs",
    }

    assert set(summary.columns) == expected_columns
    assert len(summary) == 2
    assert set(summary["n_runs"]) == {10}


def test_markdown_report_is_created(tmp_path):
    summary = summarize_records(_records())
    output_path = write_summary_markdown(summary, tmp_path / "summary.md")

    content = output_path.read_text(encoding="utf-8")

    assert output_path.exists()
    assert "## Skew Ratio 1:1" in content
    assert "| model_id" in content


def test_csv_report_is_created(tmp_path):
    summary = summarize_records(_records())
    output_path = write_summary_csv(summary, tmp_path / "summary.csv")

    content = output_path.read_text(encoding="utf-8")

    assert output_path.exists()
    assert "skew_ratio,model_id" in content
