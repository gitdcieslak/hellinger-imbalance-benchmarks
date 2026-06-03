"""Report allocation morphology breadth/elevation diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MEAN_COLUMNS = [
    "positive_histogram_entropy",
    "positive_effective_score_bins",
    "positive_top_bin_mass",
    "positive_max_bin_mass",
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


def build_run_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    successful = df[df["fit_failed"].astype(str).str.lower() != "true"].copy()
    run_level = successful.drop_duplicates(["objective", "seed", "model_name"]).copy()
    run_level["allocation_breadth"] = run_level["positive_histogram_entropy"]
    run_level["allocation_effective_breadth"] = run_level["positive_effective_score_bins"]
    run_level["allocation_elevation"] = run_level["positive_top_bin_mass"]
    run_level["allocation_peak_concentration"] = run_level["positive_max_bin_mass"]
    return run_level.reset_index(drop=True)


def classify_allocation_quadrants(run_level: pd.DataFrame) -> pd.DataFrame:
    classified = run_level.copy()
    breadth_median = float(classified["allocation_breadth"].median())
    elevation_median = float(classified["allocation_elevation"].median())

    def label(row: pd.Series) -> str:
        breadth = "high_breadth" if row["allocation_breadth"] >= breadth_median else "low_breadth"
        elevation = "high_elevation" if row["allocation_elevation"] >= elevation_median else "low_elevation"
        return f"{breadth}_{elevation}"

    classified["allocation_quadrant"] = classified.apply(label, axis=1)
    return classified


def correlation_summary(run_level: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("allocation_breadth", "minority_survival_auc", "breadth vs survival_auc"),
        ("allocation_elevation", "minority_survival_auc", "elevation vs survival_auc"),
        ("allocation_breadth", "minority_survival_cliffiness", "breadth vs cliffiness"),
        ("allocation_elevation", "minority_survival_cliffiness", "elevation vs cliffiness"),
    ]
    rows = []
    for left, right, label in pairs:
        values = run_level[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        corr = np.nan
        if len(values) >= 2 and values[left].nunique() > 1 and values[right].nunique() > 1:
            corr = float(values[left].corr(values[right]))
        rows.append({"comparison": label, "correlation": corr, "abs_correlation": abs(corr)})
    return pd.DataFrame(rows)


def quadrant_counts(run_level: pd.DataFrame) -> pd.DataFrame:
    return (
        run_level.groupby(["objective", "allocation_quadrant"])
        .size()
        .reset_index(name="n_runs")
        .sort_values(["objective", "allocation_quadrant"])
    )


def interpretation_lines(run_level: pd.DataFrame, corrs: pd.DataFrame) -> list[str]:
    means = run_level.groupby("objective", as_index=True)[
        ["allocation_breadth", "allocation_elevation"]
    ].mean()
    lines = []
    for variant, baseline in [
        ("bce_dropout_0_1", "bce"),
        ("bce_dropout_0_3", "bce"),
        ("weighted_bce_dropout_0_1", "weighted_bce"),
        ("weighted_bce_dropout_0_3", "weighted_bce"),
    ]:
        if variant in means.index and baseline in means.index:
            delta_breadth = means.loc[variant, "allocation_breadth"] - means.loc[baseline, "allocation_breadth"]
            delta_elevation = means.loc[variant, "allocation_elevation"] - means.loc[baseline, "allocation_elevation"]
            direction = "toward high-breadth / low-elevation" if delta_breadth > 0 and delta_elevation < 0 else "not cleanly toward high-breadth / low-elevation"
            lines.append(f"- `{variant}` moves {direction} relative to `{baseline}` (delta breadth={delta_breadth:.4f}, delta elevation={delta_elevation:.4f}).")

    if "weighted_bce" in means.index:
        elevation_rank = means["allocation_elevation"].rank(ascending=False).loc["weighted_bce"]
        lines.append(f"- `weighted_bce` elevation rank is {int(elevation_rank)} of {len(means)} objectives; lower rank means higher elevation.")

    corr_map = dict(zip(corrs["comparison"], corrs["abs_correlation"], strict=False))
    survival_tracks = "elevation" if corr_map.get("elevation vs survival_auc", 0.0) >= corr_map.get("breadth vs survival_auc", 0.0) else "breadth"
    cliff_tracks = "breadth" if corr_map.get("breadth vs cliffiness", 0.0) >= corr_map.get("elevation vs cliffiness", 0.0) else "elevation"
    lines.append(f"- Survival AUC tracks {survival_tracks} more strongly than the other morphology axis in this pilot.")
    lines.append(f"- Cliffiness tracks {cliff_tracks} more strongly than the other morphology axis in this pilot.")
    if survival_tracks != cliff_tracks:
        lines.append("- Allocation appears to have separable breadth and elevation axes for survival-level and survival-shape effects.")
    else:
        lines.append("- Allocation axes are not cleanly separable in this pilot; the same axis dominates both survival summaries.")
    return lines


def build_allocation_morphology_report(df: pd.DataFrame) -> str:
    run_level = classify_allocation_quadrants(build_run_level_dataset(df))
    objective_means = run_level.groupby("objective", as_index=False)[MEAN_COLUMNS].mean()
    corrs = correlation_summary(run_level)
    counts = quadrant_counts(run_level)

    lines = [
        "# Allocation Morphology Summary",
        "",
        f"Run-level observations: {len(run_level)}",
        "",
        "## Objective-Level Means",
        _markdown_table(objective_means),
        "",
        "## Breadth/Elevation Correlations",
        _markdown_table(corrs),
        "",
        "## Objective-Level Quadrant Counts",
        _markdown_table(counts),
        "",
        "## Interpretation",
        *interpretation_lines(run_level, corrs),
    ]
    return "\n".join(lines) + "\n"


def plot_breadth_vs_elevation(run_level: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    objectives = sorted(run_level["objective"].astype(str).unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(objectives))))
    for objective, color in zip(objectives, colors, strict=False):
        group = run_level[run_level["objective"] == objective]
        sizes = 40.0 + 180.0 * group["minority_survival_auc"].clip(0.0, 1.0)
        ax.scatter(
            group["positive_histogram_entropy"],
            group["positive_top_bin_mass"],
            s=sizes,
            color=color,
            alpha=0.75,
            label=objective,
            edgecolors="black",
            linewidths=0.4,
        )
    ax.set_xlabel("positive_histogram_entropy (allocation breadth)")
    ax.set_ylabel("positive_top_bin_mass (allocation elevation)")
    ax.set_title("Allocation Morphology: Breadth vs Elevation")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_report(input_path: Path, output_md: Path, output_png: Path) -> tuple[Path, Path]:
    df = pd.read_csv(input_path)
    run_level = classify_allocation_quadrants(build_run_level_dataset(df))
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_allocation_morphology_report(df), encoding="utf-8")
    plot_breadth_vs_elevation(run_level, output_png)
    return output_md, output_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "topology" / "mlp_objectives_topology.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "reports" / "topology" / "allocation_morphology_summary.md",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=ROOT / "reports" / "topology" / "allocation_morphology_breadth_vs_elevation.png",
    )
    args = parser.parse_args()
    md_path, png_path = write_report(args.input, args.output_md, args.output_png)
    print(f"wrote {md_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
