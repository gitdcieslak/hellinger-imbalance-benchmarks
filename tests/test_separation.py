import numpy as np

from hib.reporting import summarize_separation_jsonl, summarize_separation_records
from hib.runner import run_score_separation_suite, write_jsonl
from hib.separation import score_separation_metrics


def test_separation_metrics_on_clearly_separated_distributions():
    pos = np.array([0.8, 0.85, 0.9, 0.95])
    neg = np.array([0.02, 0.05, 0.08, 0.10])

    metrics = score_separation_metrics(pos, neg)

    assert metrics["mean_gap"] > 0.7
    assert metrics["median_gap"] > 0.7
    assert metrics["ks_statistic"] > 0.9
    assert metrics["fraction_positives_above_negative_p95"] == 1.0


def test_separation_metrics_on_overlapping_distributions():
    pos = np.array([0.3, 0.4, 0.5, 0.6])
    neg = np.array([0.2, 0.4, 0.5, 0.7])

    metrics = score_separation_metrics(pos, neg)

    assert abs(metrics["mean_gap"]) < 0.2
    assert metrics["ks_statistic"] < 0.6


def test_hellinger_histogram_distance_bounded():
    pos = np.array([0.0, 0.1, 0.9, 1.0])
    neg = np.array([0.2, 0.3, 0.4, 0.5])

    metrics = score_separation_metrics(pos, neg)

    assert 0.0 <= metrics["histogram_hellinger_distance"] <= 1.0


def test_fraction_above_negative_quantiles_are_correct():
    pos = np.array([0.50, 0.60, 0.70, 0.80])
    neg = np.array([0.10, 0.20, 0.30, 0.40])

    metrics = score_separation_metrics(pos, neg)

    assert metrics["fraction_positives_above_negative_p95"] == 1.0
    assert metrics["fraction_positives_above_negative_p99"] == 1.0


def test_separation_jsonl_and_reports_generation(tmp_path):
    config = {
        "experiment_id": "test_score_separation",
        "skew_ratios": [1],
        "seeds": [0],
        "separation": 1.5,
        "minority_count": 10,
        "n_features": 4,
        "noise": 1.0,
        "test_size": 0.5,
        "n_repeats": 2,
        "split_seed": 0,
        "models": ["cart"],
    }
    records = run_score_separation_suite(config)
    assert len(records) == 2
    assert all("separation_metrics" in record for record in records)

    summary = summarize_separation_records(records)
    assert "n_runs" in summary.columns
    assert set(summary["n_runs"]) == {2}

    input_path = write_jsonl(records, tmp_path / "separation.jsonl")
    csv_path, markdown_path = summarize_separation_jsonl(
        input_path,
        tmp_path / "separation_summary.csv",
        tmp_path / "separation_summary.md",
    )
    assert csv_path.exists()
    assert markdown_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Synthetic Score Separation Summary" in markdown
    assert "| model_id" in markdown
