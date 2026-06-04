"""Report minority information-density allocation morphology sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MEAN_METRICS = [
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


def regime_means(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("support_regime", as_index=False)[MEAN_METRICS].mean().sort_values("support_regime")


def model_regime_means(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "minority_survival_auc",
        "minority_survival_cliffiness",
        "breadth",
        "elevation",
        "auroc",
        "average_precision",
    ]
    return df.groupby(["model_id", "support_regime"], as_index=False)[cols].mean().sort_values(["model_id", "support_regime"])


def density_correlations(df: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "positive_density_proxy",
        "mean_positive_knn_distance",
        "local_positive_ratio_mean",
        "local_label_entropy_mean",
        "minority_cluster_count",
        "positives_per_cluster",
    ]
    outcomes = [
        "elevation",
        "breadth",
        "minority_survival_auc",
        "minority_survival_cliffiness",
    ]
    rows = []
    for predictor in predictors:
        for outcome in outcomes:
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "correlation": float(df[predictor].corr(df[outcome])),
                }
            )
    return pd.DataFrame(rows)


def model_robustness(df: pd.DataFrame) -> pd.DataFrame:
    low_density = df[df["support_regime"].isin(["fragmented_islands_20", "singleton_islands"])]
    if low_density.empty:
        low_density = df.sort_values("positive_density_proxy").head(max(1, len(df) // 3))
    return (
        low_density.groupby("model_id", as_index=False)[
            ["minority_survival_auc", "minority_survival_cliffiness", "elevation", "average_precision"]
        ]
        .mean()
        .sort_values(["minority_survival_auc", "minority_survival_cliffiness"], ascending=[False, True])
    )


def best_cells(df: pd.DataFrame) -> pd.DataFrame:
    means = model_regime_means(df)
    specs = [
        ("best_survival_auc", "minority_survival_auc", False),
        ("lowest_cliffiness", "minority_survival_cliffiness", True),
        ("highest_breadth", "breadth", False),
        ("highest_elevation", "elevation", False),
        ("best_auroc", "auroc", False),
        ("best_ap", "average_precision", False),
    ]
    rows = []
    for criterion, metric, ascending in specs:
        row = means.sort_values(metric, ascending=ascending).iloc[0]
        rows.append(
            {
                "criterion": criterion,
                "model_id": row["model_id"],
                "support_regime": row["support_regime"],
                "metric": metric,
                "mean_value": float(row[metric]),
            }
        )
    return pd.DataFrame(rows)


def explanatory_strength(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ["breadth", "elevation"]:
        density_score = max(
            abs(float(df["positive_density_proxy"].corr(df[outcome]))),
            abs(float(df["mean_positive_knn_distance"].corr(df[outcome]))),
            abs(float(df["local_positive_ratio_mean"].corr(df[outcome]))),
            abs(float(df["local_label_entropy_mean"].corr(df[outcome]))),
        )
        model_means = df.groupby("model_id")[outcome].mean()
        total_std = float(df[outcome].std(ddof=0))
        model_spread = float(model_means.std(ddof=0) / max(1e-9, total_std))
        rows.append(
            {
                "outcome": outcome,
                "max_abs_density_or_ambiguity_correlation": density_score,
                "model_family_spread_ratio": model_spread,
                "better_explained_by": "density_or_ambiguity" if density_score >= model_spread else "model_family",
            }
        )
    return pd.DataFrame(rows)


def interpretation_lines(df: pd.DataFrame) -> list[str]:
    corrs = density_correlations(df)
    robustness = model_robustness(df)
    strength = explanatory_strength(df)
    dens_elev = corrs[(corrs["predictor"] == "positive_density_proxy") & (corrs["outcome"] == "elevation")].iloc[0]
    dens_surv = corrs[(corrs["predictor"] == "positive_density_proxy") & (corrs["outcome"] == "minority_survival_auc")].iloc[0]
    frag_cliff = corrs[(corrs["predictor"] == "minority_cluster_count") & (corrs["outcome"] == "minority_survival_cliffiness")].iloc[0]
    entropy_cliff = corrs[(corrs["predictor"] == "local_label_entropy_mean") & (corrs["outcome"] == "minority_survival_cliffiness")].iloc[0]
    best_model = robustness.iloc[0]["model_id"] if not robustness.empty else "n/a"
    density_beats_model = int((strength["better_explained_by"] == "density_or_ambiguity").sum())

    return [
        f"- Does lower local minority density reduce elevation? {'Yes' if dens_elev['correlation'] > 0 else 'No'}; density proxy vs elevation correlation={dens_elev['correlation']:.4f}.",
        f"- Does lower local minority density reduce survival AUC? {'Yes' if dens_surv['correlation'] > 0 else 'No'}; density proxy vs survival AUC correlation={dens_surv['correlation']:.4f}.",
        f"- Does fragmented support increase cliffiness? {'Yes' if frag_cliff['correlation'] > 0 else 'No'}; cluster count vs cliffiness correlation={frag_cliff['correlation']:.4f}. Ambiguity also matters: local label entropy vs cliffiness correlation={entropy_cliff['correlation']:.4f}.",
        "- Does global skew matter less than local information density? This sweep holds global skew fixed, so it cannot estimate a global-skew coefficient directly; observed variation under fixed skew is evidence that local density/ambiguity/support structure matters independently of global imbalance.",
        f"- Which models are most robust to low-density minority support? {best_model} ranks highest by low-density survival AUC with cliffiness as tie-breaker.",
        f"- Are breadth/elevation better explained by density metrics than by model family? Density/ambiguity wins for {density_beats_model}/2 morphology outcomes by the simple correlation-vs-model-spread diagnostic.",
    ]


def build_information_density_report(df: pd.DataFrame) -> str:
    lines = [
        "# Minority Information Density Sweep",
        "",
        f"Rows: {len(df)}",
        f"Support regimes: {', '.join(sorted(df['support_regime'].unique()))}",
        f"Models: {', '.join(sorted(df['model_id'].unique()))}",
        "",
        "## Regime Means",
        _markdown_table(regime_means(df)),
        "",
        "## Model x Regime Means",
        _markdown_table(model_regime_means(df)),
        "",
        "## Best Model/Regime Cells",
        _markdown_table(best_cells(df)),
        "",
        "## Density And Ambiguity Correlations",
        _markdown_table(density_correlations(df)),
        "",
        "## Low-Density Robustness Ranking",
        _markdown_table(model_robustness(df)),
        "",
        "## Breadth/Elevation Explanatory Diagnostic",
        _markdown_table(explanatory_strength(df)),
        "",
        "## Interpretation",
        *interpretation_lines(df),
    ]
    return "\n".join(lines) + "\n"


def plot_breadth_elevation(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    means = model_regime_means(df).merge(
        regime_means(df)[["support_regime", "positive_density_proxy", "local_label_entropy_mean"]],
        on="support_regime",
        how="left",
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        means["breadth"],
        means["elevation"],
        c=means["positive_density_proxy"],
        s=90.0 + 180.0 * means["local_label_entropy_mean"].fillna(0.0),
        cmap="magma",
        edgecolors="black",
        linewidths=0.5,
    )
    for row in means.itertuples(index=False):
        ax.annotate(f"{row.support_regime}\n{row.model_id}", (row.breadth, row.elevation), fontsize=7)
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Information Density Morphology Space")
    fig.colorbar(scatter, ax=ax, label="positive_density_proxy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_metric_by_regime(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    means = model_regime_means(df)
    pivot = means.pivot(index="support_regime", columns="model_id", values=metric).sort_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    for idx, model_id in enumerate(pivot.columns):
        ax.bar(x + idx * width, pivot[model_id], width=width, label=model_id)
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2.0, labels=list(pivot.index), rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_report(input_path: Path, output_md: Path, output_morphology: Path, output_survival: Path, output_cliffiness: Path) -> tuple[Path, Path, Path, Path]:
    df = pd.read_csv(input_path)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_information_density_report(df), encoding="utf-8")
    plot_breadth_elevation(df, output_morphology)
    plot_metric_by_regime(df, "minority_survival_auc", output_survival, "Survival AUC By Regime")
    plot_metric_by_regime(df, "minority_survival_cliffiness", output_cliffiness, "Cliffiness By Regime")
    return output_md, output_morphology, output_survival, output_cliffiness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "topology" / "information_density_sweep.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "reports" / "topology" / "information_density_sweep_summary.md",
    )
    parser.add_argument(
        "--output-morphology",
        type=Path,
        default=ROOT / "reports" / "topology" / "information_density_breadth_elevation.png",
    )
    parser.add_argument(
        "--output-survival",
        type=Path,
        default=ROOT / "reports" / "topology" / "information_density_survival_by_regime.png",
    )
    parser.add_argument(
        "--output-cliffiness",
        type=Path,
        default=ROOT / "reports" / "topology" / "information_density_cliffiness_by_regime.png",
    )
    args = parser.parse_args()
    paths = write_report(args.input, args.output_md, args.output_morphology, args.output_survival, args.output_cliffiness)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
