import numpy as np

from hib.reporting import summarize_score_jsonl, summarize_score_records
from hib.runner import run_score_analysis_suite, write_jsonl
from hib.scores import class_conditional_score_summaries, histogram_counts, score_summary


def test_score_aggregation_and_quantiles():
    scores = np.array([0.0, 0.1, 0.2, 0.9, 1.0])

    summary = score_summary(scores)

    assert summary["mean_probability"] == float(np.mean(scores))
    assert summary["p50"] == float(np.quantile(scores, 0.5))
    assert summary["p99"] == float(np.quantile(scores, 0.99))


def test_histogram_bucket_assignment():
    scores = np.array([0.0, 0.009, 0.01, 0.049, 0.1, 0.5, 1.0])

    hist = histogram_counts(scores)

    assert hist["[0.0,0.01)"] == 2
    assert hist["[0.01,0.05)"] == 2
    assert hist["[0.10,0.25)"] == 1
    assert hist["[0.50,1.00]"] == 2


def test_summary_generation_and_markdown(tmp_path):
    config = {
        "experiment_id": "test_score_analysis",
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
    records = run_score_analysis_suite(config)
    assert len(records) == 4
    assert {record["class_label"] for record in records} == {0, 1}

    summary = summarize_score_records(records)
    assert "n_runs" in summary.columns
    assert set(summary["n_runs"]) == {2}

    input_path = write_jsonl(records, tmp_path / "score.jsonl")
    csv_path, markdown_path = summarize_score_jsonl(
        input_path,
        tmp_path / "score_summary.csv",
        tmp_path / "score_summary.md",
    )
    assert csv_path.exists()
    assert markdown_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Synthetic Score Distribution Summary" in markdown
    assert "| model_id" in markdown


def test_class_conditional_summaries_exist_for_both_classes():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

    output = class_conditional_score_summaries(y_true, y_score)

    assert len(output) == 2
    assert {record["class_label"] for record in output} == {0, 1}
