"""Threshold-response plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


KNOWN_MODEL_ORDER = ["cart", "hddt", "bagged_hddt", "random_forest", "xgboost", "lightgbm"]
DEFAULT_METRICS = ["recall", "f1", "precision", "balanced_accuracy"]


def load_threshold_summary_csv(path: str | Path) -> pd.DataFrame:
    """Load threshold summary CSV for legacy or synthetic threshold sweeps."""

    frame = pd.read_csv(path)
    required = {"model_id", "threshold"}
    if "dataset_id" not in frame.columns and "skew_ratio" not in frame.columns:
        raise ValueError("threshold summary must include either dataset_id or skew_ratio")
    if not required.issubset(set(frame.columns)):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"missing required threshold summary columns: {', '.join(missing)}")
    return frame


def ordered_model_ids(model_ids: list[str]) -> list[str]:
    """Apply stable model ordering with unknown models appended alphabetically."""

    observed = list(dict.fromkeys(model_ids))
    known = [model for model in KNOWN_MODEL_ORDER if model in observed]
    extras = sorted([model for model in observed if model not in KNOWN_MODEL_ORDER])
    return known + extras


def _source_group_column(frame: pd.DataFrame) -> str:
    if "dataset_id" in frame.columns:
        return "dataset_id"
    return "skew_ratio"


def _plot_paths(output_dir: Path, source: str, dataset_id: str, metric: str) -> tuple[Path, Path]:
    stem = f"{source}_{dataset_id}_{metric}_vs_threshold"
    return output_dir / f"{stem}.png", output_dir / f"{stem}.svg"


def plot_metric_vs_threshold(
    dataset_frame: pd.DataFrame,
    metric: str,
    dataset_id: str,
    source: str,
    output_dir: str | Path,
    error_bands: bool = True,
) -> list[Path]:
    """Plot a single metric-versus-threshold chart for one dataset."""

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in dataset_frame.columns:
        raise ValueError(f"metric {metric!r} not found in summary columns")

    thresholds = sorted([float(value) for value in dataset_frame["threshold"].unique()], reverse=True)
    model_order = ordered_model_ids(dataset_frame["model_id"].astype(str).tolist())
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_id in model_order:
        subset = dataset_frame[dataset_frame["model_id"] == model_id].copy()
        if subset.empty:
            continue
        subset["threshold"] = subset["threshold"].astype(float)
        subset = subset.sort_values("threshold", ascending=False)
        x_vals = subset["threshold"].to_numpy(dtype=float)
        y_vals = subset[mean_col].to_numpy(dtype=float)
        ax.plot(x_vals, y_vals, marker="o", linewidth=1.8, label=model_id)

        if error_bands and std_col in subset.columns:
            std_vals = subset[std_col].to_numpy(dtype=float)
            lower = (y_vals - std_vals).clip(0.0, 1.0)
            upper = (y_vals + std_vals).clip(0.0, 1.0)
            ax.fill_between(x_vals, lower, upper, alpha=0.15)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(max(thresholds), min(thresholds))
    ax.set_xticks(thresholds)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{dataset_id}: {metric.replace('_', ' ').title()} vs Threshold ({source})")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()

    png_path, svg_path = _plot_paths(out_dir, source, dataset_id, metric)
    fig.savefig(png_path, dpi=120)
    fig.savefig(svg_path)
    plt.close(fig)
    return [png_path, svg_path]


def plot_threshold_response_panel(
    summary: pd.DataFrame,
    dataset_id: str,
    source: str,
    metrics: list[str] | None = None,
    output_dir: str | Path = "reports/plots/threshold_response",
    error_bands: bool = True,
) -> list[Path]:
    """Plot multiple threshold-response metrics for one dataset."""

    group_col = _source_group_column(summary)
    dataset_frame = summary[summary[group_col].astype(str) == str(dataset_id)].copy()
    if dataset_frame.empty:
        raise ValueError(f"dataset {dataset_id!r} not found in threshold summary")
    chosen_metrics = DEFAULT_METRICS if metrics is None else metrics

    created: list[Path] = []
    for metric in chosen_metrics:
        created.extend(
            plot_metric_vs_threshold(
                dataset_frame=dataset_frame,
                metric=metric,
                dataset_id=str(dataset_id),
                source=source,
                output_dir=output_dir,
                error_bands=error_bands,
            )
        )
    return created


def plot_threshold_response(
    summary: pd.DataFrame,
    source: str = "legacy",
    datasets: list[str] | None = None,
    metrics: list[str] | None = None,
    output_dir: str | Path = "reports/plots/threshold_response",
    error_bands: bool = True,
) -> list[Path]:
    """Plot threshold response charts for all selected datasets."""

    group_col = _source_group_column(summary)
    dataset_ids = (
        sorted(summary[group_col].astype(str).unique().tolist()) if datasets is None else [str(item) for item in datasets]
    )
    created: list[Path] = []
    for dataset_id in dataset_ids:
        created.extend(
            plot_threshold_response_panel(
                summary=summary,
                dataset_id=dataset_id,
                source=source,
                metrics=metrics,
                output_dir=output_dir,
                error_bands=error_bands,
            )
        )
    return created
