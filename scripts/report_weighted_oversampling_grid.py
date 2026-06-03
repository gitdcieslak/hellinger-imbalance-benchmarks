"""Report weighted BCE x oversampling allocation morphology grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

METRICS = [
    "breadth",
    "effective_breadth",
    "elevation",
    "peak_concentration",
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


def grid_means(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["class_weight_multiplier", "oversampling_multiplier"], as_index=False)[METRICS]
        .mean()
        .sort_values(["class_weight_multiplier", "oversampling_multiplier"])
    )


def best_cells(df: pd.DataFrame) -> pd.DataFrame:
    means = grid_means(df)
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
                "class_weight_multiplier": int(row["class_weight_multiplier"]),
                "oversampling_multiplier": int(row["oversampling_multiplier"]),
                "metric": metric,
                "mean_value": float(row[metric]),
            }
        )
    return pd.DataFrame(rows)


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    means = grid_means(df)
    rows = []
    for idx, row in means.iterrows():
        dominated = means[
            (means["breadth"] >= row["breadth"])
            & (means["elevation"] >= row["elevation"])
            & ((means["breadth"] > row["breadth"]) | (means["elevation"] > row["elevation"]))
        ]
        if dominated.empty:
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["breadth", "elevation"])


def interpretation_lines(df: pd.DataFrame, best: pd.DataFrame, frontier: pd.DataFrame) -> list[str]:
    means = grid_means(df)
    weight_corr_breadth = float(means["class_weight_multiplier"].corr(means["breadth"]))
    weight_corr_elevation = float(means["class_weight_multiplier"].corr(means["elevation"]))
    over_corr_breadth = float(means["oversampling_multiplier"].corr(means["breadth"]))
    over_corr_elevation = float(means["oversampling_multiplier"].corr(means["elevation"]))
    median_breadth = float(means["breadth"].median())
    median_elevation = float(means["elevation"].median())
    high_high = means[(means["breadth"] >= median_breadth) & (means["elevation"] >= median_elevation)]

    best_survival = best[best["criterion"] == "best_survival_auc"].iloc[0]
    best_auroc = best[best["criterion"] == "best_auroc"].iloc[0]
    best_ap = best[best["criterion"] == "best_ap"].iloc[0]
    baseline = means[(means["class_weight_multiplier"] == 1) & (means["oversampling_multiplier"] == 1)].iloc[0]
    combined = means[(means["class_weight_multiplier"] > 1) & (means["oversampling_multiplier"] > 1)]
    improves_without_cliff = combined[
        (combined["minority_survival_auc"] > baseline["minority_survival_auc"])
        & (combined["minority_survival_cliffiness"] <= baseline["minority_survival_cliffiness"])
    ]
    different_directions = np.sign(weight_corr_breadth) != np.sign(over_corr_breadth) or np.sign(weight_corr_elevation) != np.sign(over_corr_elevation)
    access_cell = (int(best_survival["class_weight_multiplier"]), int(best_survival["oversampling_multiplier"]))
    auroc_cell = (int(best_auroc["class_weight_multiplier"]), int(best_auroc["oversampling_multiplier"]))
    ap_cell = (int(best_ap["class_weight_multiplier"]), int(best_ap["oversampling_multiplier"]))

    return [
        f"- Do weighting and oversampling move in different morphology directions? {'Yes' if different_directions else 'No'}; weight correlations are breadth={weight_corr_breadth:.4f}, elevation={weight_corr_elevation:.4f}, oversampling correlations are breadth={over_corr_breadth:.4f}, elevation={over_corr_elevation:.4f}.",
        f"- Is there a high-breadth / high-elevation region? {'Yes' if not high_high.empty else 'No'}; {len(high_high)} mean grid cells are above both medians.",
        f"- Does the accessibility-optimal cell differ from the AUROC/AP-optimal cell? {'Yes' if access_cell not in {auroc_cell, ap_cell} else 'No'}; accessibility={access_cell}, AUROC={auroc_cell}, AP={ap_cell}.",
        f"- Does combining weighting + oversampling improve survival AUC without increasing cliffiness? {'Yes' if not improves_without_cliff.empty else 'No'}; qualifying combined cells={len(improves_without_cliff)}.",
        f"- Is there evidence of a Pareto frontier between breadth and elevation? {'Yes' if len(frontier) > 1 else 'Weak'}; frontier cells={len(frontier)}.",
        f"- Weighting and oversampling appear {'complementary' if different_directions or len(frontier) > 1 else 'redundant'} in this grid.",
    ]


def build_weighted_oversampling_grid_report(df: pd.DataFrame) -> str:
    means = grid_means(df)
    best = best_cells(df)
    frontier = pareto_frontier(df)
    lines = [
        "# Weighted BCE x Oversampling Allocation Morphology Grid",
        "",
        f"Rows: {len(df)}",
        "",
        "## Grid Cell Means",
        _markdown_table(means),
        "",
        "## Best Cells",
        _markdown_table(best),
        "",
        "## Breadth/Elevation Pareto Frontier",
        _markdown_table(frontier[["class_weight_multiplier", "oversampling_multiplier", "breadth", "elevation"]]),
        "",
        "## Interpretation",
        *interpretation_lines(df, best, frontier),
    ]
    return "\n".join(lines) + "\n"


def plot_breadth_elevation(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    means = grid_means(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        means["breadth"],
        means["elevation"],
        c=means["class_weight_multiplier"],
        s=60.0 + 220.0 * means["minority_survival_auc"].clip(0.0, 1.0),
        cmap="plasma",
        edgecolors="black",
        linewidths=0.5,
    )
    for row in means.itertuples(index=False):
        ax.annotate(f"w{row.class_weight_multiplier}/o{row.oversampling_multiplier}", (row.breadth, row.elevation), fontsize=7)
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Weighted x Oversampling Morphology Grid")
    fig.colorbar(scatter, ax=ax, label="class_weight_multiplier")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_heatmap(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    means = grid_means(df)
    pivot = means.pivot(index="class_weight_multiplier", columns="oversampling_multiplier", values=metric).sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[str(v) for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), labels=[str(v) for v in pivot.index])
    ax.set_xlabel("oversampling_multiplier")
    ax.set_ylabel("class_weight_multiplier")
    ax.set_title(title)
    for y_idx, weight in enumerate(pivot.index):
        for x_idx, over in enumerate(pivot.columns):
            ax.text(x_idx, y_idx, f"{pivot.loc[weight, over]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_report(input_path: Path, output_md: Path, output_morphology: Path, output_auc: Path, output_cliffiness: Path) -> tuple[Path, Path, Path, Path]:
    df = pd.read_csv(input_path)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_weighted_oversampling_grid_report(df), encoding="utf-8")
    plot_breadth_elevation(df, output_morphology)
    plot_heatmap(df, "minority_survival_auc", output_auc, "Survival AUC")
    plot_heatmap(df, "minority_survival_cliffiness", output_cliffiness, "Cliffiness")
    return output_md, output_morphology, output_auc, output_cliffiness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "topology" / "weighted_oversampling_grid.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "reports" / "topology" / "weighted_oversampling_grid_summary.md",
    )
    parser.add_argument(
        "--output-morphology",
        type=Path,
        default=ROOT / "reports" / "topology" / "weighted_oversampling_grid_breadth_elevation.png",
    )
    parser.add_argument(
        "--output-auc",
        type=Path,
        default=ROOT / "reports" / "topology" / "weighted_oversampling_grid_survival_auc_heatmap.png",
    )
    parser.add_argument(
        "--output-cliffiness",
        type=Path,
        default=ROOT / "reports" / "topology" / "weighted_oversampling_grid_cliffiness_heatmap.png",
    )
    args = parser.parse_args()
    paths = write_report(args.input, args.output_md, args.output_morphology, args.output_auc, args.output_cliffiness)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
