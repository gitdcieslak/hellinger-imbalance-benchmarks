"""Report density x separability factorial allocation morphology experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DENSITY_ORDER = ["high_density", "medium_density", "low_density"]
SEPARABILITY_ORDER = ["high_separability", "medium_separability", "low_separability"]
METRICS = [
    "auroc",
    "average_precision",
    "minority_survival_auc",
    "minority_survival_cliffiness",
    "breadth",
    "effective_breadth",
    "elevation",
    "peak_concentration",
    "mean_positive_knn_distance",
    "positive_density_proxy",
    "local_positive_ratio_mean",
    "local_label_entropy_mean",
]
CURVE_METRICS = ["minority_survival_auc", "minority_survival_cliffiness", "breadth", "elevation"]


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


def _safe_corr(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 2 or y.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _label(prefix: str, value: float) -> str:
    precision = 2 if prefix == "cov" else 1
    return f"{prefix}_{float(value):.{precision}f}".replace("-", "neg_").replace(".", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "density_label" not in out.columns:
        out["density_label"] = out.get("density_level", out["minority_cov"].map(lambda v: _label("cov", v)))
    if "separability_label" not in out.columns:
        out["separability_label"] = out.get("separability_level", out["centroid_distance"].map(lambda v: _label("dist", v)))
    if "density_level" not in out.columns:
        out["density_level"] = out["density_label"]
    if "separability_level" not in out.columns:
        out["separability_level"] = out["separability_label"]
    return out


def factor_features(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    out["density_numeric"] = out["minority_cov"].astype(float)
    out["separability_numeric"] = out["centroid_distance"].astype(float)
    out["density_x_separability"] = out["density_numeric"] * out["separability_numeric"]
    out["is_dropout"] = out["model_id"].astype(str).str.contains("dropout").astype(int)
    return out


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    return (
        df.groupby(["density_label", "separability_label", "minority_cov", "centroid_distance", "model_id"], as_index=False)[METRICS]
        .mean()
        .sort_values(["minority_cov", "centroid_distance", "model_id"])
    )


def factor_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    featured = factor_features(df)
    for outcome in ["elevation", "breadth", "minority_survival_auc", "minority_survival_cliffiness"]:
        for predictor in ["density_numeric", "separability_numeric", "density_x_separability", "is_dropout"]:
            rows.append(
                {
                    "outcome": outcome,
                    "predictor": predictor,
                    "correlation": _safe_corr(featured[predictor], featured[outcome]),
                }
            )
    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    featured = factor_features(df)
    rows = []
    for predictor in ["minority_cov", "centroid_distance", "density_x_separability"]:
        source = predictor if predictor != "density_x_separability" else "density_x_separability"
        for outcome in CURVE_METRICS:
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "correlation": _safe_corr(featured[source], featured[outcome]),
                }
            )
    return pd.DataFrame(rows)


def density_derivative_estimates(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    means = (
        df.groupby(["centroid_distance", "model_id", "minority_cov"], as_index=False)[CURVE_METRICS]
        .mean()
        .sort_values(["centroid_distance", "model_id", "minority_cov"])
    )
    rows = []
    for (centroid_distance, model_id), group in means.groupby(["centroid_distance", "model_id"]):
        group = group.sort_values("minority_cov")
        covs = group["minority_cov"].to_numpy(dtype=float)
        if covs.size < 2:
            continue
        for metric in CURVE_METRICS:
            values = group[metric].to_numpy(dtype=float)
            deltas = np.diff(values)
            dcovs = np.diff(covs)
            for idx, derivative in enumerate(deltas / dcovs):
                rows.append(
                    {
                        "centroid_distance": float(centroid_distance),
                        "model_id": model_id,
                        "metric": metric,
                        "cov_from": float(covs[idx]),
                        "cov_to": float(covs[idx + 1]),
                        "delta": float(deltas[idx]),
                        "derivative": float(derivative),
                        "abs_derivative": float(abs(derivative)),
                    }
                )
    return pd.DataFrame(rows)


def largest_adjacent_density_changes(df: pd.DataFrame) -> pd.DataFrame:
    derivatives = density_derivative_estimates(df)
    if derivatives.empty:
        return derivatives
    rows = []
    for metric, group in derivatives.groupby("metric"):
        row = group.sort_values("abs_derivative", ascending=False).iloc[0]
        rows.append(row.to_dict())
    return pd.DataFrame(rows)[["metric", "centroid_distance", "model_id", "cov_from", "cov_to", "delta", "derivative", "abs_derivative"]]


def candidate_threshold_regions(df: pd.DataFrame) -> pd.DataFrame:
    derivatives = density_derivative_estimates(df)
    if derivatives.empty:
        return derivatives
    rows = []
    for metric, group in derivatives.groupby("metric"):
        cutoff = float(group["abs_derivative"].quantile(0.90))
        candidates = group[group["abs_derivative"] >= cutoff].copy()
        candidates["threshold_reason"] = "top_10pct_abs_density_derivative"
        rows.append(candidates)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["metric", "abs_derivative"], ascending=[True, False])[
        ["metric", "centroid_distance", "model_id", "cov_from", "cov_to", "delta", "derivative", "abs_derivative", "threshold_reason"]
    ]


def dropout_benefit(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    base = df[~df["model_id"].astype(str).str.contains("dropout")]
    drop = df[df["model_id"].astype(str).str.contains("dropout")]
    if base.empty or drop.empty:
        return pd.DataFrame()
    keys = ["minority_cov", "centroid_distance", "density_label", "separability_label", "seed"]
    base_cols = keys + ["minority_survival_auc", "minority_survival_cliffiness", "breadth", "elevation"]
    drop_cols = base_cols
    merged = base[base_cols].merge(drop[drop_cols], on=keys, suffixes=("_base", "_dropout"))
    for metric in ["minority_survival_auc", "minority_survival_cliffiness", "breadth", "elevation"]:
        merged[f"delta_{metric}"] = merged[f"{metric}_dropout"] - merged[f"{metric}_base"]
    return (
        merged.groupby(["density_label", "separability_label", "minority_cov", "centroid_distance"], as_index=False)[
            [
                "delta_minority_survival_auc",
                "delta_minority_survival_cliffiness",
                "delta_breadth",
                "delta_elevation",
            ]
        ]
        .mean()
        .sort_values(["minority_cov", "centroid_distance"])
    )


def accessibility_failure_driver(df: pd.DataFrame) -> str:
    effects = factor_effects(df)
    surv = effects[effects["outcome"] == "minority_survival_auc"].set_index("predictor")["correlation"].abs()
    cliff = effects[effects["outcome"] == "minority_survival_cliffiness"].set_index("predictor")["correlation"].abs()
    scores = {
        "density": float(surv.get("density_numeric", 0.0) + cliff.get("density_numeric", 0.0)),
        "separability": float(surv.get("separability_numeric", 0.0) + cliff.get("separability_numeric", 0.0)),
        "interaction": float(surv.get("density_x_separability", 0.0) + cliff.get("density_x_separability", 0.0)),
    }
    return max(scores, key=scores.get)


def interpretation_lines(df: pd.DataFrame) -> list[str]:
    effects = factor_effects(df)
    benefit = dropout_benefit(df)
    driver = accessibility_failure_driver(df)

    def corr(outcome: str, predictor: str) -> float:
        row = effects[(effects["outcome"] == outcome) & (effects["predictor"] == predictor)].iloc[0]
        return float(row["correlation"])

    dens_elev = corr("elevation", "density_numeric")
    sep_elev = corr("elevation", "separability_numeric")
    dens_breadth = corr("breadth", "density_numeric")
    sep_breadth = corr("breadth", "separability_numeric")
    if benefit.empty:
        dropout_line = "- Does dropout help more in low-density or low-separability regimes? Not estimable because no paired dropout/base models were present."
    else:
        low_density_cutoff = benefit["minority_cov"].quantile(0.75)
        low_sep_cutoff = benefit["centroid_distance"].quantile(0.25)
        low_density = benefit[benefit["minority_cov"] >= low_density_cutoff]["delta_minority_survival_auc"].mean()
        low_sep = benefit[benefit["centroid_distance"] <= low_sep_cutoff]["delta_minority_survival_auc"].mean()
        target = "low-density" if low_density >= low_sep else "low-separability"
        dropout_line = f"- Does dropout help more in low-density or low-separability regimes? {target}; mean survival AUC delta low-density={low_density:.4f}, low-separability={low_sep:.4f}."

    return [
        f"- Does density independently affect elevation? {'Yes' if abs(dens_elev) >= 0.20 else 'Weak'}; minority_cov vs elevation correlation={dens_elev:.4f}.",
        f"- Does separability independently affect elevation? {'Yes' if abs(sep_elev) >= 0.20 else 'Weak'}; centroid_distance vs elevation correlation={sep_elev:.4f}.",
        f"- Does density independently affect breadth? {'Yes' if abs(dens_breadth) >= 0.20 else 'Weak'}; minority_cov vs breadth correlation={dens_breadth:.4f}.",
        f"- Does separability independently affect breadth? {'Yes' if abs(sep_breadth) >= 0.20 else 'Weak'}; centroid_distance vs breadth correlation={sep_breadth:.4f}.",
        f"- Is accessibility failure mostly density, separability, or interaction? {driver} by combined absolute survival/cliffiness correlation.",
        dropout_line,
    ]


def build_density_separability_report(df: pd.DataFrame) -> str:
    df = normalize_columns(df)
    lines = [
        "# Density x Separability Factorial",
        "",
        f"Rows: {len(df)}",
        "",
        "## Cell Means",
        _markdown_table(cell_means(df)),
        "",
        "## Factor And Interaction Features",
        _markdown_table(factor_features(df)[["density_label", "separability_label", "minority_cov", "centroid_distance", "density_numeric", "separability_numeric", "density_x_separability", "is_dropout"]].drop_duplicates()),
        "",
        "## Factor Effects",
        _markdown_table(factor_effects(df)),
        "",
        "## Continuous Factor Correlations",
        _markdown_table(correlation_table(df)),
        "",
        "## First Derivative Estimates",
        _markdown_table(density_derivative_estimates(df)),
        "",
        "## Largest Adjacent Density Changes",
        _markdown_table(largest_adjacent_density_changes(df)),
        "",
        "## Candidate Accessibility Threshold Regions",
        _markdown_table(candidate_threshold_regions(df)),
        "",
        "## Dropout Benefit",
        _markdown_table(dropout_benefit(df)),
        "",
        "## Interpretation",
        *interpretation_lines(df),
    ]
    return "\n".join(lines) + "\n"


def _ordered(values: list[str], order: list[str]) -> list[str]:
    return [item for item in order if item in values] + [item for item in sorted(values) if item not in order]


def plot_heatmap(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = normalize_columns(df)
    means = df.groupby(["density_label", "separability_label", "minority_cov", "centroid_distance"], as_index=False)[metric].mean()
    density_order = list(means.sort_values("minority_cov")["density_label"].drop_duplicates())
    sep_order = list(means.sort_values("centroid_distance")["separability_label"].drop_duplicates())
    pivot = means.pivot(index="density_label", columns="separability_label", values=metric).reindex(index=density_order, columns=sep_order)
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=list(pivot.columns), rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=list(pivot.index))
    ax.set_xlabel("centroid_distance")
    ax.set_ylabel("minority_cov")
    ax.set_title(title)
    for y_idx, density in enumerate(pivot.index):
        for x_idx, sep in enumerate(pivot.columns):
            value = pivot.loc[density, sep]
            ax.text(x_idx, y_idx, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_morphology(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    means = cell_means(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        means["breadth"],
        means["elevation"],
        c=means["minority_survival_auc"],
        s=80.0 + 160.0 * means["minority_survival_cliffiness"].clip(0.0, 1.0),
        cmap="plasma",
        edgecolors="black",
        linewidths=0.5,
    )
    for row in means.itertuples(index=False):
        ax.annotate(f"{row.density_label}\n{row.separability_label}\n{row.model_id}", (row.breadth, row.elevation), fontsize=7)
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Density x Separability Morphology Space")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="minority_survival_auc")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _mean_ci(df: pd.DataFrame, factor: str, metric: str) -> pd.DataFrame:
    grouped = df.groupby(factor)[metric]
    rows = []
    for value, series in grouped:
        arr = series.to_numpy(dtype=float)
        mean = float(np.mean(arr))
        ci = 1.96 * float(np.std(arr, ddof=1)) / np.sqrt(arr.size) if arr.size > 1 else 0.0
        rows.append({factor: float(value), "mean": mean, "ci": ci})
    return pd.DataFrame(rows).sort_values(factor)


def plot_factor_curves(df: pd.DataFrame, factor: str, output_path: Path, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = normalize_columns(df)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for ax, metric in zip(axes.ravel(), CURVE_METRICS, strict=False):
        stats = _mean_ci(df, factor, metric)
        ax.plot(stats[factor], stats["mean"], marker="o")
        ax.fill_between(stats[factor], stats["mean"] - stats["ci"], stats["mean"] + stats["ci"], alpha=0.20)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel(factor)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_report(
    input_path: Path,
    output_md: Path,
    output_survival: Path | None = None,
    output_cliffiness: Path | None = None,
    output_elevation: Path | None = None,
    output_breadth: Path | None = None,
    output_morphology: Path | None = None,
    output_density_curves: Path | None = None,
    output_separability_curves: Path | None = None,
) -> tuple[Path, ...]:
    df = normalize_columns(pd.read_csv(input_path))
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_density_separability_report(df), encoding="utf-8")
    base = output_md.parent
    paths = [output_md]
    plot_specs = [
        ("minority_survival_auc", output_survival or base / "density_separability_survival_auc_heatmap.png", "Survival AUC"),
        ("minority_survival_cliffiness", output_cliffiness or base / "density_separability_cliffiness_heatmap.png", "Cliffiness"),
        ("elevation", output_elevation or base / "density_separability_elevation_heatmap.png", "Elevation"),
        ("breadth", output_breadth or base / "density_separability_breadth_heatmap.png", "Breadth"),
    ]
    for metric, path, title in plot_specs:
        paths.append(plot_heatmap(df, metric, path, title))
    paths.append(plot_morphology(df, output_morphology or base / "density_separability_morphology_space.png"))
    paths.append(plot_factor_curves(df, "minority_cov", output_density_curves or base / "density_separability_density_curves.png", "Density Curves"))
    paths.append(plot_factor_curves(df, "centroid_distance", output_separability_curves or base / "density_separability_separability_curves.png", "Separability Curves"))
    return tuple(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "topology" / "density_separability_factorial.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "topology" / "density_separability_factorial_summary.md")
    parser.add_argument("--output-survival", type=Path, default=None)
    parser.add_argument("--output-cliffiness", type=Path, default=None)
    parser.add_argument("--output-elevation", type=Path, default=None)
    parser.add_argument("--output-breadth", type=Path, default=None)
    parser.add_argument("--output-morphology", type=Path, default=None)
    parser.add_argument("--output-density-curves", type=Path, default=None)
    parser.add_argument("--output-separability-curves", type=Path, default=None)
    args = parser.parse_args()
    paths = write_report(
        args.input,
        args.output_md,
        args.output_survival,
        args.output_cliffiness,
        args.output_elevation,
        args.output_breadth,
        args.output_morphology,
        args.output_density_curves,
        args.output_separability_curves,
    )
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
