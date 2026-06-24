"""Summarize HDDT benchmark accessibility validation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "real_validation" / "hddt_accessibility_validation.csv"
DEFAULT_OUTPUT_MD = ROOT / "reports" / "real_validation" / "hddt_accessibility_validation_summary.md"
DEFAULT_REPORT_DIR = ROOT / "reports" / "real_validation"
ATLAS_CSV = ROOT / "reports" / "topology" / "morphology_atlas_clusters.csv"
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float) or isinstance(value, np.floating):
                values.append(format(float(value), floatfmt) if pd.notna(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_successful(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "fit_failed" in df.columns:
        df = df[~df["fit_failed"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    return df.dropna(subset=["breadth", "elevation", TARGET, SURVIVAL]).reset_index(drop=True)


def support_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    if not ATLAS_CSV.exists() or df.empty:
        out = df.copy()
        out["nearest_support_distance"] = np.nan
        out["inside_axis_support"] = False
        return out, {"fraction_inside_support": np.nan, "mean_nearest_support_distance": np.nan}
    atlas = pd.read_csv(ATLAS_CSV).dropna(subset=["breadth", "elevation"])
    coords = atlas[["breadth", "elevation"]].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=1).fit(coords)
    distances, _ = nn.kneighbors(df[["breadth", "elevation"]].to_numpy(dtype=float))
    out = df.copy()
    out["nearest_support_distance"] = distances[:, 0]
    bmin, bmax = atlas["breadth"].min(), atlas["breadth"].max()
    emin, emax = atlas["elevation"].min(), atlas["elevation"].max()
    out["inside_axis_support"] = out["breadth"].between(bmin, bmax) & out["elevation"].between(emin, emax)
    return out, {"fraction_inside_support": float(out["inside_axis_support"].mean()), "mean_nearest_support_distance": float(out["nearest_support_distance"].mean())}


def cv_r2(df: pd.DataFrame, target: str, features: list[str]) -> float:
    data = df.dropna(subset=[target, *features])
    if len(data) < 10 or data[target].var(ddof=0) <= 0:
        return float("nan")
    cv = KFold(n_splits=min(5, len(data)), shuffle=True, random_state=42)
    pred = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), data[features], data[target], cv=cv)
    return float(r2_score(data[target], pred))


def regime_variance(df: pd.DataFrame) -> dict[str, float]:
    data = df.dropna(subset=["regime_id", TARGET]).copy()
    if data.empty or data["regime_id"].nunique() < 2:
        return {"between_regime_variance": np.nan, "within_regime_variance": np.nan, "between_within_ratio": np.nan}
    between = float(data.groupby("regime_id")[TARGET].mean().var(ddof=0))
    within = float(data.groupby("regime_id")[TARGET].var(ddof=0).fillna(0).mean())
    return {"between_regime_variance": between, "within_regime_variance": within, "between_within_ratio": between / max(within, 1e-9)}


def summarize(df: pd.DataFrame, supported: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "dataset_summary": df.groupby("dataset_name", as_index=False).agg(n_runs=("model_id", "size"), n_tasks=("task_id", "nunique"), mean_positive_fraction=("positive_fraction", "mean")),
        "model_summary": df.groupby("model_id", as_index=False).agg(n_runs=("task_id", "size"), mean_auroc=("auroc", "mean"), mean_average_precision=("average_precision", "mean"), mean_brier_score=("brier_score", "mean")),
        "accessibility_summary": df.groupby("model_id", as_index=False).agg(mean_survival_auc=(SURVIVAL, "mean"), mean_cliffiness=(TARGET, "mean"), mean_breadth=("breadth", "mean"), mean_elevation=("elevation", "mean")),
        "regime_counts": df.groupby(["dataset_name", "model_id", "regime_id"], as_index=False).size().rename(columns={"size": "n_runs"}),
        "outside_examples": supported[~supported["inside_axis_support"]].sort_values("nearest_support_distance", ascending=False).head(10)[["dataset_name", "task_id", "model_id", "breadth", "elevation", "nearest_support_distance"]],
    }


def plot_morphology(df: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    if ATLAS_CSV.exists():
        atlas = pd.read_csv(ATLAS_CSV).dropna(subset=["breadth", "elevation"])
        ax.scatter(atlas["breadth"], atlas["elevation"], s=10, alpha=0.15, color="gray", label="atlas support")
    ax.scatter(df["breadth"], df["elevation"], s=28, alpha=0.75, c=df[TARGET], cmap="magma", marker="x", label="HDDT validation")
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("HDDT Validation Morphology Space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_regime(df: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["breadth"], df["elevation"], c=df["regime_id"], cmap="tab10", s=32, alpha=0.8, marker="x")
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Projected Regime Assignments")
    fig.colorbar(sc, ax=ax, label="regime_id")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_survival(df: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(df["breadth"], df[SURVIVAL], alpha=0.7)
    axes[0].set_xlabel("breadth")
    axes[0].set_ylabel("survival AUC")
    axes[1].scatter(df["elevation"], df[SURVIVAL], alpha=0.7)
    axes[1].set_xlabel("elevation")
    axes[1].set_ylabel("survival AUC")
    fig.suptitle("Survival AUC vs Morphology")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_cliffiness(df: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [g[TARGET].dropna().to_numpy() for _, g in df.groupby("regime_id")]
    labels = [str(k) for k, _ in df.groupby("regime_id")]
    ax.boxplot(groups, tick_labels=labels)
    ax.set_xlabel("projected regime")
    ax.set_ylabel("cliffiness")
    ax.set_title("Cliffiness by Projected Regime")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_heatmap(df: pd.DataFrame, output: Path) -> Path:
    pivot = df.pivot_table(index="dataset_name", columns="model_id", values=SURVIVAL, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(pivot))))
    im = ax.imshow(pivot.fillna(0).to_numpy(dtype=float), cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Mean Survival AUC by Dataset and Model")
    fig.colorbar(im, ax=ax, label="survival AUC")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def conclusion(fraction_inside: float, survival_r2: float, ratio: float) -> str:
    if fraction_inside >= 0.7 and survival_r2 > 0.2 and ratio > 1.0:
        return "Strong Support"
    if survival_r2 > 0.2:
        return "Broader Partial Support"
    if fraction_inside >= 0.5 and (survival_r2 > 0.0 or ratio > 0.5):
        return "Partial Support"
    return "Weak / No Support"


def write_report(input_csv: Path, output_md: Path) -> tuple[Path, ...]:
    report_dir = output_md.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    df = load_successful(input_csv)
    supported, support = support_metrics(df)
    summaries = summarize(df, supported)
    survival_r2 = cv_r2(df, SURVIVAL, ["breadth", "elevation"])
    cliff_r2 = cv_r2(df, TARGET, ["breadth", "elevation"])
    var = regime_variance(df)
    status = conclusion(support["fraction_inside_support"], survival_r2, var["between_within_ratio"])
    models_missing = pd.read_csv(input_csv)
    if "failure_reason" not in models_missing.columns:
        models_missing["failure_reason"] = ""
    missing = models_missing[models_missing.get("fit_failed", False).astype(str).str.lower().isin(["true", "1", "yes"])][["model_id", "failure_reason"]].drop_duplicates() if "fit_failed" in models_missing.columns else pd.DataFrame()
    text = "\n".join([
        "# HDDT Dataset Accessibility Validation",
        "",
        f"Overall interpretation: **{status}**",
        "",
        "## 1. Dataset Inventory",
        _markdown_table(summaries["dataset_summary"]),
        "",
        "## 2. Task Inventory",
        f"Evaluated tasks: {df['task_id'].nunique()} across {df['dataset_name'].nunique()} datasets.",
        "",
        "## 3. Models Available / Missing",
        _markdown_table(summaries["model_summary"]),
        "",
        "Missing or failed model rows:",
        _markdown_table(missing.head(20)),
        "",
        "## 4. Conventional Metric Summary",
        _markdown_table(summaries["model_summary"]),
        "",
        "## 5. Accessibility Metric Summary",
        _markdown_table(summaries["accessibility_summary"]),
        "",
        "## 6. Morphology Placement",
        _markdown_table(pd.DataFrame([support])),
        "",
        "Outside-support examples:",
        _markdown_table(summaries["outside_examples"]),
        "",
        "## 7. Regime Projection Results",
        _markdown_table(summaries["regime_counts"].head(50)),
        "",
        "## 8. Agreement With Paper 2 Claims",
        _markdown_table(pd.DataFrame([{**support, "survival_morphology_cv_r2": survival_r2, "cliffiness_morphology_cv_r2": cliff_r2, **var}])),
        "",
        "## 9. Failure Cases",
        _markdown_table(missing.head(30)),
        "",
        "## 10. Manuscript Recommendation",
        recommendation(status),
    ]) + "\n"
    output_md.write_text(text, encoding="utf-8")
    outputs = [output_md]
    outputs.append(plot_morphology(supported, report_dir / "hddt_morphology_space.png"))
    outputs.append(plot_regime(supported, report_dir / "hddt_regime_projection.png"))
    outputs.append(plot_survival(supported, report_dir / "hddt_survival_vs_morphology.png"))
    outputs.append(plot_cliffiness(supported, report_dir / "hddt_cliffiness_by_regime.png"))
    outputs.append(plot_heatmap(supported, report_dir / "hddt_dataset_model_heatmap.png"))
    return tuple(outputs)


def recommendation(status: str) -> str:
    if status == "Strong Support":
        return "Add a short External Benchmark Validation subsection to v0.3. Frame as controlled benchmark support, not broad real-world proof."
    if status in {"Partial Support", "Broader Partial Support"}:
        return "Add a short validation subsection or appendix paragraph. Emphasize partial external support and dataset-specific regime behavior."
    return "Use as limitation/future-work evidence unless additional datasets or preprocessing improve support."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()
    for path in write_report(args.input, args.output_md):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
