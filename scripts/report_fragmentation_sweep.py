"""Report minority fragmentation sweep under fixed density settings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    "auroc",
    "average_precision",
    "minority_survival_auc",
    "minority_survival_cliffiness",
    "minority_survival_max_drop",
    "minority_survival_effective_drop_count",
    "breadth",
    "effective_breadth",
    "elevation",
    "peak_concentration",
    "mean_positive_knn_distance",
    "positive_density_proxy",
    "local_positive_ratio_mean",
    "local_label_entropy_mean",
    "minority_cluster_count",
    "positives_per_cluster",
]


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join([str(c) for c in df.columns]) + " |"]
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            values.append(format(value, floatfmt) if isinstance(value, float) and pd.notna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def level_means(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("n_islands", as_index=False)[METRICS].mean().sort_values("n_islands")


def model_level_means(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["minority_survival_auc", "minority_survival_cliffiness", "breadth", "elevation", "auroc", "average_precision"]
    return df.groupby(["model_id", "n_islands"], as_index=False)[cols].mean().sort_values(["model_id", "n_islands"])


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictors = ["n_islands", "positives_per_cluster", "positive_density_proxy", "mean_positive_knn_distance"]
    outcomes = ["minority_survival_auc", "minority_survival_cliffiness", "breadth", "elevation"]
    for predictor in predictors:
        for outcome in outcomes:
            rows.append({"predictor": predictor, "outcome": outcome, "correlation": float(df[predictor].corr(df[outcome]))})
    return pd.DataFrame(rows)


def density_adjusted_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    controls = np.column_stack([np.ones(len(df)), df["positive_density_proxy"].to_numpy(dtype=float)])
    for predictor in ["n_islands", "positives_per_cluster"]:
        pred = df[predictor].to_numpy(dtype=float)
        pred_resid = pred - controls @ np.linalg.lstsq(controls, pred, rcond=None)[0]
        for outcome in ["minority_survival_auc", "minority_survival_cliffiness", "breadth", "elevation"]:
            y = df[outcome].to_numpy(dtype=float)
            y_resid = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
            corr = float(np.corrcoef(pred_resid, y_resid)[0, 1]) if np.std(pred_resid) > 0 and np.std(y_resid) > 0 else 0.0
            rows.append({"predictor": predictor, "outcome": outcome, "density_adjusted_correlation": corr})
    return pd.DataFrame(rows)


def dropout_delta(df: pd.DataFrame) -> pd.DataFrame:
    base = df[~df["model_id"].astype(str).str.contains("dropout")]
    drop = df[df["model_id"].astype(str).str.contains("dropout")]
    if base.empty or drop.empty:
        return pd.DataFrame()
    keys = ["n_islands", "seed"]
    metrics = ["minority_survival_auc", "minority_survival_cliffiness", "breadth", "elevation"]
    merged = base[keys + metrics].merge(drop[keys + metrics], on=keys, suffixes=("_base", "_dropout"))
    for metric in metrics:
        merged[f"delta_{metric}"] = merged[f"{metric}_dropout"] - merged[f"{metric}_base"]
    return merged.groupby("n_islands", as_index=False)[[f"delta_{m}" for m in metrics]].mean().sort_values("n_islands")


def threshold_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    means = level_means(df)
    rows = []
    for metric in ["minority_survival_auc", "minority_survival_cliffiness", "breadth", "elevation"]:
        diffs = np.diff(means[metric].to_numpy(dtype=float))
        if diffs.size == 0:
            continue
        idx = int(np.argmax(np.abs(diffs)))
        rows.append({
            "metric": metric,
            "largest_step_from": int(means.iloc[idx]["n_islands"]),
            "largest_step_to": int(means.iloc[idx + 1]["n_islands"]),
            "largest_step_delta": float(diffs[idx]),
        })
    return pd.DataFrame(rows)


def interpretation_lines(df: pd.DataFrame) -> list[str]:
    corr = correlations(df)
    adj = density_adjusted_correlations(df)
    drop = dropout_delta(df)
    n_surv = corr[(corr["predictor"] == "n_islands") & (corr["outcome"] == "minority_survival_auc")].iloc[0]["correlation"]
    n_cliff_adj = adj[(adj["predictor"] == "n_islands") & (adj["outcome"] == "minority_survival_cliffiness")].iloc[0]["density_adjusted_correlation"]
    ppc_surv = abs(corr[(corr["predictor"] == "positives_per_cluster") & (corr["outcome"] == "minority_survival_auc")].iloc[0]["correlation"])
    n_abs_surv = abs(n_surv)
    n_breadth = abs(corr[(corr["predictor"] == "n_islands") & (corr["outcome"] == "breadth")].iloc[0]["correlation"])
    n_elev = abs(corr[(corr["predictor"] == "n_islands") & (corr["outcome"] == "elevation")].iloc[0]["correlation"])
    if drop.empty:
        dropout_line = "- Does dropout help fragmented support or worsen survival? Not estimable; paired dropout/base rows are missing."
    else:
        high_frag = drop[drop["n_islands"] >= drop["n_islands"].quantile(0.75)]
        delta_surv = float(high_frag["delta_minority_survival_auc"].mean())
        dropout_line = f"- Does dropout help fragmented support or worsen survival? {'Helps' if delta_surv > 0 else 'Worsens'} survival in high-fragmentation levels; mean delta={delta_surv:.4f}."
    return [
        f"- Does fragmentation reduce survival AUC when density is approximately controlled? {'Yes' if n_surv < 0 else 'No'}; n_islands vs survival AUC correlation={n_surv:.4f}.",
        f"- Does fragmentation increase cliffiness independently of density? {'Yes' if n_cliff_adj > 0 else 'No'}; density-adjusted n_islands vs cliffiness correlation={n_cliff_adj:.4f}.",
        f"- Does positives_per_cluster predict accessibility better than n_islands? {'Yes' if ppc_surv > n_abs_surv else 'No'}; abs correlations survival AUC: positives_per_cluster={ppc_surv:.4f}, n_islands={n_abs_surv:.4f}.",
        f"- Does fragmentation mainly change breadth or elevation? {'breadth' if n_breadth >= n_elev else 'elevation'}; abs n_islands correlations breadth={n_breadth:.4f}, elevation={n_elev:.4f}.",
        dropout_line,
    ]


def build_fragmentation_report(df: pd.DataFrame) -> str:
    lines = [
        "# Minority Fragmentation Sweep",
        "",
        f"Rows: {len(df)}",
        "",
        "## Island Level Means",
        _markdown_table(level_means(df)),
        "",
        "## Model x Island Means",
        _markdown_table(model_level_means(df)),
        "",
        "## Correlations",
        _markdown_table(correlations(df)),
        "",
        "## Density-Adjusted Correlations",
        _markdown_table(density_adjusted_correlations(df)),
        "",
        "## Dropout Delta",
        _markdown_table(dropout_delta(df)),
        "",
        "## Fragmentation Threshold Diagnostics",
        _markdown_table(threshold_diagnostics(df)),
        "",
        "## Interpretation",
        *interpretation_lines(df),
    ]
    return "\n".join(lines) + "\n"


def _mean_ci(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for n_islands, group in df.groupby("n_islands"):
        arr = group[metric].to_numpy(dtype=float)
        ci = 1.96 * float(np.std(arr, ddof=1)) / np.sqrt(arr.size) if arr.size > 1 else 0.0
        rows.append({"n_islands": int(n_islands), "mean": float(np.mean(arr)), "ci": ci})
    return pd.DataFrame(rows).sort_values("n_islands")


def plot_metric(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_id, group in df.groupby("model_id"):
        stats = _mean_ci(group, metric)
        ax.plot(stats["n_islands"], stats["mean"], marker="o", label=model_id)
        ax.fill_between(stats["n_islands"], stats["mean"] - stats["ci"], stats["mean"] + stats["ci"], alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel("n_islands")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_breadth_elevation(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    means = model_level_means(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(means["breadth"], means["elevation"], c=means["n_islands"], s=80 + 180 * means["minority_survival_cliffiness"].clip(0, 1), cmap="viridis", edgecolors="black", linewidths=0.5)
    for row in means.itertuples(index=False):
        ax.annotate(f"{row.model_id}\n{row.n_islands}", (row.breadth, row.elevation), fontsize=7)
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Fragmentation Breadth/Elevation Space")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="n_islands")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_report(input_path: Path, output_md: Path, output_survival: Path, output_cliffiness: Path, output_morphology: Path) -> tuple[Path, Path, Path, Path]:
    df = pd.read_csv(input_path)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_fragmentation_report(df), encoding="utf-8")
    plot_metric(df, "minority_survival_auc", output_survival, "Survival AUC vs Fragmentation")
    plot_metric(df, "minority_survival_cliffiness", output_cliffiness, "Cliffiness vs Fragmentation")
    plot_breadth_elevation(df, output_morphology)
    return output_md, output_survival, output_cliffiness, output_morphology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "topology" / "fragmentation_sweep.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "topology" / "fragmentation_sweep_summary.md")
    parser.add_argument("--output-survival", type=Path, default=ROOT / "reports" / "topology" / "fragmentation_survival_auc.png")
    parser.add_argument("--output-cliffiness", type=Path, default=ROOT / "reports" / "topology" / "fragmentation_cliffiness.png")
    parser.add_argument("--output-morphology", type=Path, default=ROOT / "reports" / "topology" / "fragmentation_breadth_elevation.png")
    args = parser.parse_args()
    for path in write_report(args.input, args.output_md, args.output_survival, args.output_cliffiness, args.output_morphology):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
