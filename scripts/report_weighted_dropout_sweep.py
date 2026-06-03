"""Report weighted BCE feature-dropout sweep results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SUMMARY_METRICS = [
    "breadth",
    "elevation",
    "minority_survival_auc",
    "minority_survival_cliffiness",
    "auroc",
    "average_precision",
]


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(format(value, floatfmt) if pd.notna(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def dropout_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby(["skew_ratio", "dropout_rate"])[SUMMARY_METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def best_dropout_rates(df: pd.DataFrame) -> pd.DataFrame:
    means = df.groupby(["skew_ratio", "dropout_rate"], as_index=False)[SUMMARY_METRICS].mean()
    specs = [
        ("highest_mean_survival_auc", "minority_survival_auc", False),
        ("lowest_mean_cliffiness", "minority_survival_cliffiness", True),
        ("highest_mean_breadth", "breadth", False),
        ("highest_mean_elevation", "elevation", False),
        ("best_auroc", "auroc", False),
        ("best_ap", "average_precision", False),
    ]
    rows = []
    for skew_ratio, group in means.groupby("skew_ratio"):
        for criterion, metric, ascending in specs:
            row = group.sort_values(metric, ascending=ascending).iloc[0]
            rows.append(
                {
                    "skew_ratio": skew_ratio,
                    "criterion": criterion,
                    "dropout_rate": float(row["dropout_rate"]),
                    "metric": metric,
                    "mean_value": float(row[metric]),
                }
            )
    return pd.DataFrame(rows)


def tradeoff_correlations(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("dropout_rate", "breadth"),
        ("dropout_rate", "elevation"),
        ("dropout_rate", "minority_survival_auc"),
        ("dropout_rate", "minority_survival_cliffiness"),
        ("breadth", "minority_survival_cliffiness"),
        ("elevation", "minority_survival_auc"),
    ]
    rows = []
    for skew_ratio, group in df.groupby("skew_ratio"):
        for left, right in pairs:
            values = group[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
            corr = np.nan
            if len(values) >= 2 and values[left].nunique() > 1 and values[right].nunique() > 1:
                corr = float(values[left].corr(values[right]))
            rows.append({"skew_ratio": skew_ratio, "comparison": f"{left} vs {right}", "correlation": corr, "abs_correlation": abs(corr)})
    return pd.DataFrame(rows)


def interpretation_lines(df: pd.DataFrame, best: pd.DataFrame, corrs: pd.DataFrame) -> list[str]:
    means = df.groupby(["skew_ratio", "dropout_rate"], as_index=False)[SUMMARY_METRICS].mean()
    best_auc = best[best["criterion"] == "highest_mean_survival_auc"]
    best_cliff = best[best["criterion"] == "lowest_mean_cliffiness"]
    best_auroc = best[best["criterion"] == "best_auroc"]
    best_ap = best[best["criterion"] == "best_ap"]

    monotone_auc = []
    u_shaped_cliff = []
    modest_auc_improves = []
    for skew_ratio, group in means.groupby("skew_ratio"):
        group = group.sort_values("dropout_rate")
        auc_diffs = np.diff(group["minority_survival_auc"].to_numpy(dtype=float))
        monotone_auc.append(float(np.mean(auc_diffs >= -1e-9)) >= 0.8)
        cliff = group["minority_survival_cliffiness"].to_numpy(dtype=float)
        min_idx = int(np.argmin(cliff))
        u_shaped_cliff.append(0 < min_idx < len(cliff) - 1)
        baseline = float(group[group["dropout_rate"] == 0.0]["minority_survival_auc"].iloc[0])
        modest = group[(group["dropout_rate"] > 0.0) & (group["dropout_rate"] <= 0.2)]
        modest_auc_improves.append(bool((modest["minority_survival_auc"] > baseline).any()))

    elevation_positive = corrs[corrs["comparison"] == "dropout_rate vs elevation"]["correlation"].dropna().gt(0).mean()
    breadth_negative = corrs[corrs["comparison"] == "dropout_rate vs breadth"]["correlation"].dropna().lt(0).mean()
    best_auc_at_050 = best_auc["dropout_rate"].eq(0.5).mean()
    accessibility_differs = (
        best_auc.set_index("skew_ratio")["dropout_rate"] != best_auroc.set_index("skew_ratio")["dropout_rate"]
    ).mean()
    accessibility_differs_ap = (
        best_auc.set_index("skew_ratio")["dropout_rate"] != best_ap.set_index("skew_ratio")["dropout_rate"]
    ).mean()
    lines = [
        f"- Does survival AUC increase monotonically or near-monotonically with dropout? {'Yes' if all(monotone_auc) else 'No'}; near-monotone in {sum(monotone_auc)}/{len(monotone_auc)} skew regimes.",
        f"- Does elevation increase with dropout across skews? {'Yes' if elevation_positive >= 0.75 else 'No'}; positive dropout/elevation trend in {elevation_positive:.2f} of skew regimes.",
        f"- Does breadth decrease with dropout across skews? {'Yes' if breadth_negative >= 0.75 else 'No'}; negative dropout/breadth trend in {breadth_negative:.2f} of skew regimes.",
        f"- Does cliffiness have a U-shaped relationship with dropout? {'Yes' if all(u_shaped_cliff) else 'Mixed'}; interior cliffiness minimum in {sum(u_shaped_cliff)}/{len(u_shaped_cliff)} skew regimes.",
        f"- Is dropout 0.50 still best, or does the optimum move by skew? Dropout 0.50 is best for survival AUC in {best_auc_at_050:.2f} of skew regimes.",
        f"- Does the accessibility-optimal dropout differ from AUROC/AP optima? It differs from AUROC in {accessibility_differs:.2f} and from AP in {accessibility_differs_ap:.2f} of skew regimes.",
    ]
    if all(monotone_auc) and elevation_positive >= 0.75 and breadth_negative >= 0.75:
        mechanism = "accessibility_tradeoff"
    elif elevation_positive >= 0.75:
        mechanism = "elevation_enhancing_regularizer"
    elif breadth_negative < 0.25:
        mechanism = "breadth_expanding_regularizer"
    elif not all(monotone_auc) or best_auc_at_050 < 0.75:
        mechanism = "skew_dependent_effect"
    else:
        mechanism = "no_stable_effect"
    lines.append(f"- Mechanism classification: `{mechanism}`.")
    if any(modest_auc_improves):
        lines.append("- Modest dropout improves survival AUC over no dropout in at least one skew regime; this is not purely a ranking-performance perturbation.")
    else:
        lines.append("- Modest dropout does not improve survival AUC over no dropout in these regimes; accessibility gains are not established.")
    return lines


def build_weighted_dropout_sweep_report(df: pd.DataFrame) -> str:
    df = _ensure_skew_column(df)
    summary = dropout_level_summary(df)
    best = best_dropout_rates(df)
    corrs = tradeoff_correlations(df)
    lines = [
        "# Weighted BCE Feature-Dropout Sweep Summary",
        "",
        f"Rows: {len(df)}",
        "",
        "## Dropout x Skew Means",
        _markdown_table(summary),
        "",
        "## Best Dropout By Skew",
        _markdown_table(best),
        "",
        "## Trend Correlations By Skew",
        _markdown_table(corrs),
        "",
        "## Stability Summary And Interpretation",
        *interpretation_lines(df, best, corrs),
    ]
    return "\n".join(lines) + "\n"


def plot_metric_by_dropout(df: pd.DataFrame, metric: str, output_path: Path, ylabel: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for skew_ratio, group in df.groupby("skew_ratio"):
        summary = group.groupby("dropout_rate", as_index=False)[metric].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            summary["dropout_rate"],
            summary["mean"],
            yerr=summary["std"],
            marker="o",
            capsize=3,
            label=f"skew {skew_ratio}",
        )
    ax.set_xlabel("dropout_rate")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Weighted BCE Dropout Sweep: {ylabel}")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_breadth_elevation(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df["breadth"],
        df["elevation"],
        c=df["dropout_rate"],
        s=40.0 + 180.0 * df["minority_survival_auc"].clip(0.0, 1.0),
        cmap="viridis",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.35,
    )
    markers = ["o", "s", "^", "D", "P", "X"]
    for marker, (skew_ratio, group) in zip(markers, df.groupby("skew_ratio"), strict=False):
        ax.scatter(
            group["breadth"],
            group["elevation"],
            s=20,
            marker=marker,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            label=f"skew {skew_ratio}",
        )
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Weighted BCE Dropout Sweep: Breadth vs Elevation")
    fig.colorbar(scatter, ax=ax, label="dropout_rate")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _ensure_skew_column(df: pd.DataFrame) -> pd.DataFrame:
    if "skew_ratio" in df.columns:
        return df
    copy = df.copy()
    copy["skew_ratio"] = "unknown"
    return copy


def write_report(
    input_path: Path,
    output_md: Path,
    output_auc: Path,
    output_cliff: Path,
    output_breadth: Path,
    output_elevation: Path,
    output_morphology: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    df = pd.read_csv(input_path)
    df = _ensure_skew_column(df)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_weighted_dropout_sweep_report(df), encoding="utf-8")
    plot_metric_by_dropout(df, "minority_survival_auc", output_auc, "minority_survival_auc")
    plot_metric_by_dropout(df, "minority_survival_cliffiness", output_cliff, "minority_survival_cliffiness")
    plot_metric_by_dropout(df, "breadth", output_breadth, "breadth")
    plot_metric_by_dropout(df, "elevation", output_elevation, "elevation")
    plot_breadth_elevation(df, output_morphology)
    return output_md, output_auc, output_cliff, output_breadth, output_elevation, output_morphology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "topology" / "weighted_dropout_sweep_dense.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "reports" / "topology" / "weighted_dropout_sweep_dense_summary.md",
    )
    parser.add_argument(
        "--output-auc",
        type=Path,
        default=ROOT / "reports" / "topology" / "weighted_dropout_sweep_dense_survival_auc_by_skew.png",
    )
    parser.add_argument(
        "--output-cliffiness",
        type=Path,
        default=ROOT / "reports" / "topology" / "weighted_dropout_sweep_dense_cliffiness_by_skew.png",
    )
    parser.add_argument(
        "--output-breadth",
        type=Path,
        default=ROOT / "reports" / "topology" / "weighted_dropout_sweep_dense_breadth_by_skew.png",
    )
    parser.add_argument(
        "--output-elevation",
        type=Path,
        default=ROOT / "reports" / "topology" / "weighted_dropout_sweep_dense_elevation_by_skew.png",
    )
    parser.add_argument(
        "--output-morphology",
        type=Path,
        default=ROOT / "reports" / "topology" / "weighted_dropout_sweep_dense_breadth_elevation.png",
    )
    args = parser.parse_args()
    paths = write_report(args.input, args.output_md, args.output_auc, args.output_cliffiness, args.output_breadth, args.output_elevation, args.output_morphology)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
