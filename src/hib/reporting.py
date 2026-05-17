"""Reporting utilities for benchmark JSONL outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_NAMES = [
    "auroc",
    "average_precision",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
    "brier_score",
]
THRESHOLD_METRIC_NAMES = ["precision", "recall", "f1", "balanced_accuracy"]
SCORE_SUMMARY_METRICS = [
    "mean_probability",
    "std_probability",
    "min_probability",
    "max_probability",
    "p01",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
]
SEPARATION_METRICS = [
    "positive_mean",
    "negative_mean",
    "mean_gap",
    "positive_median",
    "negative_median",
    "median_gap",
    "positive_p10",
    "positive_p50",
    "positive_p90",
    "negative_p90",
    "negative_p95",
    "negative_p99",
    "p50_vs_neg_p95_gap",
    "p10_vs_neg_p90_gap",
    "fraction_positives_above_negative_p95",
    "fraction_positives_above_negative_p99",
    "ks_statistic",
    "histogram_hellinger_distance",
]


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Read benchmark result records from JSONL."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert nested JSONL records to a flat metric dataframe."""

    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "skew_ratio": record["skew_ratio"],
            "model_id": record["model_id"],
        }
        row.update({metric: record["metrics"][metric] for metric in METRIC_NAMES})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate result records by skew ratio and model id."""

    frame = records_to_frame(records)
    grouped = frame.groupby(["skew_ratio", "model_id"], as_index=False)[METRIC_NAMES]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    std_columns = [f"{metric}_std" for metric in METRIC_NAMES]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    run_counts = frame.groupby(["skew_ratio", "model_id"]).size().reset_index(name="n_runs")
    summary = summary.merge(run_counts, on=["skew_ratio", "model_id"], how="left")
    return summary.sort_values(["skew_ratio", "model_id"]).reset_index(drop=True)


def write_summary_csv(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Write an aggregate summary CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path


def summary_to_markdown(summary: pd.DataFrame) -> str:
    """Render simple markdown tables grouped by skew ratio."""

    sections = ["# Synthetic Smoke Summary", ""]
    for skew_ratio, group in summary.groupby("skew_ratio", sort=True):
        table = _frame_to_markdown(group.drop(columns=["skew_ratio"]))
        sections.extend(
            [
                f"## Skew Ratio {skew_ratio}",
                "",
                table,
                "",
            ]
        )
    return "\n".join(sections).rstrip() + "\n"


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small dataframe as a GitHub-flavored markdown table."""

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [_format_markdown_cell(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_summary_markdown(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a markdown summary report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary_to_markdown(summary), encoding="utf-8")
    return path


def summarize_jsonl(
    input_path: str | Path,
    csv_output_path: str | Path,
    markdown_output_path: str | Path,
) -> tuple[Path, Path]:
    """Read JSONL results and write CSV and markdown summaries."""

    records = read_jsonl_records(input_path)
    summary = summarize_records(records)
    csv_path = write_summary_csv(summary, csv_output_path)
    markdown_path = write_summary_markdown(summary, markdown_output_path)
    return csv_path, markdown_path


def threshold_records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert threshold JSONL records to a flat metric dataframe."""

    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "skew_ratio": record["skew_ratio"],
            "model_id": record["model_id"],
            "threshold": float(record["threshold"]),
        }
        row.update({metric: record["metrics"][metric] for metric in THRESHOLD_METRIC_NAMES})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_threshold_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate threshold records by skew ratio, model id, and threshold."""

    frame = threshold_records_to_frame(records)
    grouped = frame.groupby(["skew_ratio", "model_id", "threshold"], as_index=False)[
        THRESHOLD_METRIC_NAMES
    ]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    std_columns = [f"{metric}_std" for metric in THRESHOLD_METRIC_NAMES]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    run_counts = (
        frame.groupby(["skew_ratio", "model_id", "threshold"])
        .size()
        .reset_index(name="n_runs")
    )
    summary = summary.merge(run_counts, on=["skew_ratio", "model_id", "threshold"], how="left")
    return summary.sort_values(["skew_ratio", "model_id", "threshold"]).reset_index(
        drop=True
    )


def threshold_summary_to_markdown(summary: pd.DataFrame) -> str:
    """Render threshold summaries grouped by skew ratio."""

    sections = ["# Synthetic Threshold Sweep Summary", ""]
    for skew_ratio, group in summary.groupby("skew_ratio", sort=True):
        table = _frame_to_markdown(group.drop(columns=["skew_ratio"]))
        sections.extend([f"## Skew Ratio {skew_ratio}", "", table, ""])
    return "\n".join(sections).rstrip() + "\n"


def write_threshold_summary_markdown(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a threshold sweep markdown summary."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(threshold_summary_to_markdown(summary), encoding="utf-8")
    return path


def summarize_threshold_jsonl(
    input_path: str | Path,
    csv_output_path: str | Path,
    markdown_output_path: str | Path,
) -> tuple[Path, Path]:
    """Read threshold JSONL results and write CSV and markdown summaries."""

    records = read_jsonl_records(input_path)
    summary = summarize_threshold_records(records)
    csv_path = write_summary_csv(summary, csv_output_path)
    markdown_path = write_threshold_summary_markdown(summary, markdown_output_path)
    return csv_path, markdown_path


def summarize_legacy_threshold_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate legacy threshold records by dataset, model, and threshold."""

    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "dataset_id": record["dataset_id"],
            "source_group": record.get("source_group", "unknown"),
            "model_id": record["model_id"],
            "threshold": float(record["threshold"]),
        }
        row.update({metric: record["metrics"][metric] for metric in THRESHOLD_METRIC_NAMES})
        rows.append(row)

    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["dataset_id", "source_group", "model_id", "threshold"], as_index=False)[
        THRESHOLD_METRIC_NAMES
    ]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    std_columns = [f"{metric}_std" for metric in THRESHOLD_METRIC_NAMES]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    run_counts = (
        frame.groupby(["dataset_id", "source_group", "model_id", "threshold"])
        .size()
        .reset_index(name="n_runs")
    )
    summary = summary.merge(
        run_counts,
        on=["dataset_id", "source_group", "model_id", "threshold"],
        how="left",
    )
    return summary.sort_values(["dataset_id", "model_id", "threshold"]).reset_index(drop=True)


def legacy_threshold_summary_to_markdown(summary: pd.DataFrame) -> str:
    """Render legacy threshold sweep summary grouped by dataset."""

    sections = ["# Legacy HDDT Threshold Sweep Summary", ""]
    for dataset_id, group in summary.groupby("dataset_id", sort=True):
        table = _frame_to_markdown(group.drop(columns=["dataset_id"]))
        sections.extend([f"## Dataset {dataset_id}", "", table, ""])
    return "\n".join(sections).rstrip() + "\n"


def write_legacy_threshold_summary_markdown(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a markdown report for legacy threshold sweep summaries."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(legacy_threshold_summary_to_markdown(summary), encoding="utf-8")
    return path


def score_records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert score-analysis JSONL records to a flat dataframe."""

    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "skew_ratio": record["skew_ratio"],
            "model_id": record["model_id"],
            "class_label": int(record["class_label"]),
        }
        row.update(record["score_summary"])
        row.update({f"hist_{k}": v for k, v in record["histogram"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_score_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate score-analysis records by skew ratio, model id, and class."""

    frame = score_records_to_frame(records)
    histogram_columns = [column for column in frame.columns if column.startswith("hist_")]
    group_keys = ["skew_ratio", "model_id", "class_label"]
    grouped = frame.groupby(group_keys, as_index=False)

    metric_summary = grouped[SCORE_SUMMARY_METRICS].agg(["mean", "std"])
    metric_summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in metric_summary.columns
    ]
    std_columns = [f"{metric}_std" for metric in SCORE_SUMMARY_METRICS]
    metric_summary[std_columns] = metric_summary[std_columns].fillna(0.0)

    histogram_summary = grouped[histogram_columns].sum()
    run_counts = frame.groupby(group_keys).size().reset_index(name="n_runs")

    summary = metric_summary.merge(
        histogram_summary,
        on=["skew_ratio", "model_id", "class_label"],
        how="left",
    )
    summary = summary.merge(run_counts, on=["skew_ratio", "model_id", "class_label"], how="left")
    return summary.sort_values(["skew_ratio", "model_id", "class_label"]).reset_index(drop=True)


def score_summary_to_markdown(summary: pd.DataFrame) -> str:
    """Render score analysis summary grouped by skew ratio."""

    sections = ["# Synthetic Score Distribution Summary", ""]
    for skew_ratio, group in summary.groupby("skew_ratio", sort=True):
        table = _frame_to_markdown(group.drop(columns=["skew_ratio"]))
        sections.extend([f"## Skew Ratio {skew_ratio}", "", table, ""])
    return "\n".join(sections).rstrip() + "\n"


def write_score_summary_markdown(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a score-analysis markdown summary."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(score_summary_to_markdown(summary), encoding="utf-8")
    return path


def summarize_score_jsonl(
    input_path: str | Path,
    csv_output_path: str | Path,
    markdown_output_path: str | Path,
) -> tuple[Path, Path]:
    """Read score-analysis JSONL results and write CSV/markdown summaries."""

    records = read_jsonl_records(input_path)
    summary = summarize_score_records(records)
    csv_path = write_summary_csv(summary, csv_output_path)
    markdown_path = write_score_summary_markdown(summary, markdown_output_path)
    return csv_path, markdown_path


def separation_records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert score-separation JSONL records to a flat dataframe."""

    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "skew_ratio": record["skew_ratio"],
            "model_id": record["model_id"],
        }
        row.update(record["separation_metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_separation_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate score-separation records by skew ratio and model id."""

    frame = separation_records_to_frame(records)
    grouped = frame.groupby(["skew_ratio", "model_id"], as_index=False)[SEPARATION_METRICS]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    std_columns = [f"{metric}_std" for metric in SEPARATION_METRICS]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    run_counts = frame.groupby(["skew_ratio", "model_id"]).size().reset_index(name="n_runs")
    summary = summary.merge(run_counts, on=["skew_ratio", "model_id"], how="left")
    return summary.sort_values(["skew_ratio", "model_id"]).reset_index(drop=True)


def separation_summary_to_markdown(summary: pd.DataFrame) -> str:
    """Render score-separation summary grouped by skew ratio."""

    sections = ["# Synthetic Score Separation Summary", ""]
    for skew_ratio, group in summary.groupby("skew_ratio", sort=True):
        table = _frame_to_markdown(group.drop(columns=["skew_ratio"]))
        sections.extend([f"## Skew Ratio {skew_ratio}", "", table, ""])
    return "\n".join(sections).rstrip() + "\n"


def write_separation_summary_markdown(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a score-separation markdown summary."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(separation_summary_to_markdown(summary), encoding="utf-8")
    return path


def summarize_separation_jsonl(
    input_path: str | Path,
    csv_output_path: str | Path,
    markdown_output_path: str | Path,
) -> tuple[Path, Path]:
    """Read score-separation JSONL results and write CSV/markdown summaries."""

    records = read_jsonl_records(input_path)
    summary = summarize_separation_records(records)
    csv_path = write_summary_csv(summary, csv_output_path)
    markdown_path = write_separation_summary_markdown(summary, markdown_output_path)
    return csv_path, markdown_path
