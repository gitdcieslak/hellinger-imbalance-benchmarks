"""Operational allocation regime synthesis utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_allocation_summary(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "dataset_id",
        "model_id",
        "histogram_entropy_mean",
        "effective_support_size_mean",
        "gini_coefficient_mean",
        "fraction_scores_below_0_01_mean",
        "fraction_scores_below_0_05_mean",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"allocation summary missing columns: {', '.join(missing)}")
    return frame


def load_elasticity_summary(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "dataset_id",
        "model_id",
        "mean_abs_recall_elasticity",
        "mean_abs_precision_elasticity",
        "operational_smoothness_index",
        "max_recall_jump",
        "max_precision_drop",
        "threshold_of_max_recall_jump",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"elasticity summary missing columns: {', '.join(missing)}")
    return frame


def _threshold_jump_fractions(series: pd.Series) -> dict[str, float]:
    total = max(1, int(series.shape[0]))
    labels = ["0.50->0.25", "0.25->0.10", "0.10->0.05", "0.05->0.01"]
    return {
        f"fraction_jump_{label.replace('->', '_to_')}": float((series == label).sum()) / float(total)
        for label in labels
    }


def combine_regime_metrics(allocation: pd.DataFrame, elasticity: pd.DataFrame) -> pd.DataFrame:
    """Combine allocation and elasticity summaries at dataset/model level."""

    merged = allocation.merge(elasticity, on=["dataset_id", "model_id"], how="inner")
    return merged


def infer_regime_labels(summary: pd.DataFrame) -> pd.DataFrame:
    """Infer simple heuristic regime labels from combined summary metrics."""

    labels: list[str] = []
    for _, row in summary.iterrows():
        entropy = float(row["mean_histogram_entropy"])
        support = float(row["mean_effective_support_size"])
        elasticity = float(row["mean_recall_elasticity"])
        max_jump = float(row["mean_max_recall_jump"])
        below_001 = float(row["mean_fraction_below_0_01"])
        smoothness = float(row["mean_operational_smoothness_index"])

        if entropy < 0.3 and support < 1.4 and elasticity < 0.2:
            label = "quantized_allocator"
        elif max_jump > 0.35 or elasticity > 3.0:
            label = "cliff_allocator"
        elif smoothness > 0.5 and support > 2.0:
            label = "smooth_allocator"
        elif support > 3.0 and entropy > 1.0:
            label = "broad_allocator"
        elif below_001 > 0.5:
            label = "conservative_allocator"
        else:
            label = "conservative_allocator"
        labels.append(label)
    out = summary.copy()
    out["inferred_regime"] = labels
    return out


def summarize_operational_regimes(combined: pd.DataFrame) -> pd.DataFrame:
    """Aggregate combined metrics to model-level regime summary."""

    rows: list[dict[str, object]] = []
    for model_id, group in combined.groupby("model_id", sort=True):
        jump_fractions = _threshold_jump_fractions(group["threshold_of_max_recall_jump"])
        row = {
            "model_id": model_id,
            "mean_histogram_entropy": float(group["histogram_entropy_mean"].mean()),
            "mean_effective_support_size": float(group["effective_support_size_mean"].mean()),
            "mean_gini_coefficient": float(group["gini_coefficient_mean"].mean()),
            "mean_fraction_below_0_01": float(group["fraction_scores_below_0_01_mean"].mean()),
            "mean_fraction_below_0_05": float(group["fraction_scores_below_0_05_mean"].mean()),
            "mean_recall_elasticity": float(group["mean_abs_recall_elasticity"].mean()),
            "mean_precision_elasticity": float(group["mean_abs_precision_elasticity"].mean()),
            "mean_operational_smoothness_index": float(group["operational_smoothness_index"].mean()),
            "mean_max_recall_jump": float(group["max_recall_jump"].mean()),
            "mean_max_precision_drop": float(group["max_precision_drop"].mean()),
            "mean_threshold_of_max_recall_jump": str(group["threshold_of_max_recall_jump"].mode().iat[0]),
            "n_datasets": int(group.shape[0]),
        }
        row.update(jump_fractions)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("model_id").reset_index(drop=True)
    return infer_regime_labels(summary)


def allocation_regime_summary_to_markdown(summary: pd.DataFrame) -> str:
    """Render markdown narrative summary for model-level operational regimes."""

    columns = [
        "model_id",
        "inferred_regime",
        "mean_histogram_entropy",
        "mean_effective_support_size",
        "mean_gini_coefficient",
        "mean_recall_elasticity",
        "mean_max_recall_jump",
        "mean_operational_smoothness_index",
    ]
    lines = [
        "# Allocation Regime Summary",
        "",
        "## Model-Level Regime Synthesis",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in summary[columns].itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Operational Regime Notes", ""])
    for item in summary.itertuples(index=False):
        lines.append(f"### {item.model_id}")
        lines.append(f"- inferred_regime: {item.inferred_regime}")
        lines.append(f"- mean_entropy: {item.mean_histogram_entropy:.4f}")
        lines.append(f"- mean_support: {item.mean_effective_support_size:.4f}")
        lines.append(f"- mean_recall_elasticity: {item.mean_recall_elasticity:.4f}")
        lines.append(f"- mean_max_recall_jump: {item.mean_max_recall_jump:.4f}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def plot_allocation_regime_scatter(summary: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Create regime scatter plot(s) for support vs elasticity."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 6))
    x = summary["mean_effective_support_size"].to_numpy(dtype=float)
    y = summary["mean_recall_elasticity"].to_numpy(dtype=float)
    ax.scatter(x, y)
    for _, row in summary.iterrows():
        ax.annotate(str(row["model_id"]), (float(row["mean_effective_support_size"]), float(row["mean_recall_elasticity"])), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Mean Effective Support Size")
    ax.set_ylabel("Mean Recall Elasticity")
    ax.set_title("Allocation Regime Scatter: Support vs Recall Elasticity")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    png = out_dir / "allocation_regime_scatter.png"
    svg = out_dir / "allocation_regime_scatter.svg"
    fig.savefig(png, dpi=120)
    fig.savefig(svg)
    plt.close(fig)
    paths.extend([png, svg])

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    x2 = summary["mean_histogram_entropy"].to_numpy(dtype=float)
    y2 = summary["mean_operational_smoothness_index"].to_numpy(dtype=float)
    ax2.scatter(x2, y2)
    for _, row in summary.iterrows():
        ax2.annotate(str(row["model_id"]), (float(row["mean_histogram_entropy"]), float(row["mean_operational_smoothness_index"])), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax2.set_xlabel("Mean Histogram Entropy")
    ax2.set_ylabel("Mean Operational Smoothness")
    ax2.set_title("Allocation Regime Scatter: Entropy vs Smoothness")
    ax2.grid(True, alpha=0.35)
    fig2.tight_layout()
    png2 = out_dir / "allocation_regime_entropy_smoothness_scatter.png"
    svg2 = out_dir / "allocation_regime_entropy_smoothness_scatter.svg"
    fig2.savefig(png2, dpi=120)
    fig2.savefig(svg2)
    plt.close(fig2)
    paths.extend([png2, svg2])

    return paths
