"""Threshold elasticity analysis from threshold sweep summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


THRESHOLD_ORDER = [0.50, 0.25, 0.10, 0.05, 0.01]
METRICS = ["recall", "precision", "f1", "balanced_accuracy"]


def threshold_order() -> list[float]:
    """Return canonical threshold relaxation order."""

    return list(THRESHOLD_ORDER)


def load_threshold_summary(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"model_id", "threshold", "recall_mean", "precision_mean", "f1_mean", "balanced_accuracy_mean"}
    if "dataset_id" not in frame.columns and "skew_ratio" not in frame.columns:
        raise ValueError("summary csv must include dataset_id or skew_ratio")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required summary columns: {', '.join(missing)}")
    return frame


def _group_key(frame: pd.DataFrame) -> str:
    return "dataset_id" if "dataset_id" in frame.columns else "skew_ratio"


def compute_threshold_elasticity_intervals(summary: pd.DataFrame, datasets: list[str] | None = None) -> pd.DataFrame:
    """Compute interval-level deltas and elasticities for each dataset/model."""

    group_key = _group_key(summary)
    working = summary.copy()
    if datasets is not None:
        working = working[working[group_key].astype(str).isin([str(item) for item in datasets])]

    rows: list[dict[str, object]] = []
    for dataset_id, dataset_frame in working.groupby(group_key, sort=True):
        for model_id, model_frame in dataset_frame.groupby("model_id", sort=True):
            mapped = {float(row["threshold"]): row for _, row in model_frame.iterrows()}
            missing_thresholds = [thr for thr in THRESHOLD_ORDER if thr not in mapped]
            if missing_thresholds:
                raise ValueError(
                    f"missing thresholds for {group_key}={dataset_id}, model={model_id}: {missing_thresholds}"
                )

            for start, end in zip(THRESHOLD_ORDER[:-1], THRESHOLD_ORDER[1:], strict=False):
                start_row = mapped[start]
                end_row = mapped[end]
                threshold_delta = float(end - start)
                interval: dict[str, object] = {
                    group_key: dataset_id,
                    "model_id": model_id,
                    "threshold_start": float(start),
                    "threshold_end": float(end),
                    "threshold_delta": threshold_delta,
                }
                for metric in METRICS:
                    start_value = float(start_row[f"{metric}_mean"])
                    end_value = float(end_row[f"{metric}_mean"])
                    delta = float(end_value - start_value)
                    interval[f"{metric}_start"] = start_value
                    interval[f"{metric}_end"] = end_value
                    interval[f"{metric}_delta"] = delta
                    interval[f"{metric}_elasticity"] = float(delta / abs(threshold_delta))
                rows.append(interval)
    return pd.DataFrame(rows)


def summarize_threshold_elasticity(intervals: pd.DataFrame) -> pd.DataFrame:
    """Aggregate elasticity intervals into dataset/model summary metrics."""

    group_key = _group_key(intervals)
    rows: list[dict[str, object]] = []
    for (dataset_id, model_id), group in intervals.groupby([group_key, "model_id"], sort=True):
        recall_deltas = group["recall_delta"].astype(float)
        precision_deltas = group["precision_delta"].astype(float)
        f1_deltas = group["f1_delta"].astype(float)

        idx_max_recall = recall_deltas.idxmax()
        row = {
            group_key: dataset_id,
            "model_id": model_id,
            "max_recall_jump": float(recall_deltas.max()),
            "max_precision_drop": float(precision_deltas.min()),
            "max_f1_jump": float(f1_deltas.max()),
            "mean_abs_recall_elasticity": float(group["recall_elasticity"].abs().mean()),
            "mean_abs_precision_elasticity": float(group["precision_elasticity"].abs().mean()),
            "mean_abs_f1_elasticity": float(group["f1_elasticity"].abs().mean()),
            "threshold_of_max_recall_jump": f"{group.loc[idx_max_recall, 'threshold_start']:.2f}->{group.loc[idx_max_recall, 'threshold_end']:.2f}",
        }
        denom = row["mean_abs_recall_elasticity"] + row["mean_abs_precision_elasticity"]
        row["operational_smoothness_index"] = float(1.0 / (1.0 + denom))
        rows.append(row)
    return pd.DataFrame(rows).sort_values([group_key, "model_id"]).reset_index(drop=True)
