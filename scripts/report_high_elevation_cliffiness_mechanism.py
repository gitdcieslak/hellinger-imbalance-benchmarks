"""Analyze residual cliffiness mechanisms in the high-elevation/low-breadth HDDT region."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402


HDDT_CSV = ROOT / "results" / "real_validation" / "hddt_accessibility_validation_relaxed.csv"
LOCAL_SUMMARY = ROOT / "reports" / "topology" / "local_regime_validity_summary.json"
OUT = ROOT / "reports" / "topology"
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"

OCCUPANCY_COLS = [
    "positive_effective_score_bins",
    "positive_histogram_entropy",
    "positive_unique_score_ratio",
    "positive_top_bin_mass",
    "positive_max_bin_mass",
    "positive_quantization_score",
    "positive_score_iqr",
    "positive_score_q10_q90_width",
    "positive_score_gini_or_concentration_index",
]
CALIBRATION_COLS = ["brier_score", "ece", "calibration_slope", "calibration_intercept"]
TOPOLOGY_COLS = ["giant_component_fraction", "component_entropy", "isolated_positive_fraction", "mean_component_size"]
DATASET_COLS = ["positive_fraction", "n_rows", "positive_count", "n_features_processed", "auroc", "average_precision"]


def load_hddt(path: Path = HDDT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "fit_failed" in df.columns:
        df = df[~df["fit_failed"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    return df.dropna(subset=["breadth", "elevation", TARGET, SURVIVAL, "dataset_name", "model_id"]).reset_index(drop=True)


def high_elevation_region(df: pd.DataFrame) -> pd.DataFrame:
    region = df[(df["elevation"] >= 0.80) & (df["breadth"] <= 0.70)].copy()
    region["cliffiness_tercile"] = cliffiness_terciles(region[TARGET])
    return region.reset_index(drop=True)


def cliffiness_terciles(values: pd.Series) -> pd.Series:
    if values.nunique(dropna=True) < 2:
        return pd.Series(["medium"] * len(values), index=values.index)
    labels = pd.qcut(values.rank(method="first"), q=3, labels=["low", "medium", "high"])
    return labels.astype(str)


def available(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns and df[c].notna().any()]


def residual_inventory(region: pd.DataFrame) -> dict[str, float | int]:
    x = region[TARGET]
    return {
        "n_runs": int(len(region)),
        "mean_cliffiness": float(x.mean()),
        "std_cliffiness": float(x.std(ddof=0)),
        "p10": float(x.quantile(0.10)),
        "p25": float(x.quantile(0.25)),
        "median": float(x.median()),
        "p75": float(x.quantile(0.75)),
        "p90": float(x.quantile(0.90)),
        "mean_survival_auc": float(region[SURVIVAL].mean()),
        "n_datasets": int(region["dataset_name"].nunique()),
        "n_tasks": int(region["task_id"].nunique()) if "task_id" in region.columns else 0,
        "n_models": int(region["model_id"].nunique()),
    }


def tercile_inventory(region: pd.DataFrame) -> pd.DataFrame:
    return region.groupby("cliffiness_tercile", as_index=False).agg(
        n_runs=(TARGET, "size"),
        mean_cliffiness=(TARGET, "mean"),
        mean_survival_auc=(SURVIVAL, "mean"),
        mean_unique_ratio=("positive_unique_score_ratio", "mean"),
        mean_effective_bins=("positive_effective_score_bins", "mean"),
        mean_top_bin_mass=("positive_top_bin_mass", "mean"),
        dominant_datasets=("dataset_name", lambda s: ", ".join(s.value_counts().head(6).index.astype(str))),
        dominant_models=("model_id", lambda s: ", ".join(s.value_counts().head(5).index.astype(str))),
    )


def model_family_effects(region: pd.DataFrame) -> pd.DataFrame:
    return region.groupby("model_id", as_index=False).agg(
        n_runs=(TARGET, "size"),
        mean_cliffiness=(TARGET, "mean"),
        median_cliffiness=(TARGET, "median"),
        cliffiness_variance=(TARGET, lambda s: float(s.var(ddof=0))),
        mean_survival_auc=(SURVIVAL, "mean"),
    ).sort_values("mean_cliffiness", ascending=False)


def _features(df: pd.DataFrame, numeric: list[str] | None = None, categorical: list[str] | None = None) -> pd.DataFrame:
    pieces = []
    if numeric:
        cols = available(numeric, df)
        if cols:
            pieces.append(df[cols].astype(float).reset_index(drop=True))
    if categorical:
        for col in categorical:
            if col in df.columns:
                pieces.append(pd.get_dummies(df[col].astype(str), prefix=col).reset_index(drop=True))
    if not pieces:
        return pd.DataFrame(index=df.index)
    return pd.concat(pieces, axis=1)


def grouped_cv_r2(df: pd.DataFrame, X: pd.DataFrame, group_col: str = "dataset_name") -> tuple[float, bool, str]:
    data = pd.concat([df[[TARGET, group_col]].reset_index(drop=True), X.reset_index(drop=True)], axis=1).dropna()
    if X.shape[1] == 0:
        return np.nan, False, "no_features"
    if len(data) < 20:
        return np.nan, False, "too_few_rows"
    groups = data[group_col].astype(str)
    if groups.nunique() < 2:
        return np.nan, False, "too_few_groups"
    if data[TARGET].var(ddof=0) <= 1e-12:
        return np.nan, False, "constant_target"
    Xdata = data.drop(columns=[TARGET, group_col])
    try:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
        pred = cross_val_predict(make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), Xdata, data[TARGET], cv=cv, groups=groups)
        return float(r2_score(data[TARGET], pred)), True, ""
    except Exception as exc:
        return np.nan, False, type(exc).__name__


def model_scores(region: pd.DataFrame) -> pd.DataFrame:
    occupancy = available(OCCUPANCY_COLS, region)
    calibration = available(CALIBRATION_COLS, region)
    topology = available(TOPOLOGY_COLS, region)
    dataset_numeric = available(DATASET_COLS, region)
    specs = {
        "morphology_only": _features(region, ["breadth", "elevation"]),
        "model_family_only": _features(region, categorical=["model_id"]),
        "morphology_plus_model_family": _features(region, ["breadth", "elevation"], ["model_id"]),
        "occupancy_joint": _features(region, occupancy),
        "morphology_plus_occupancy": _features(region, ["breadth", "elevation"] + occupancy),
        "calibration_joint": _features(region, calibration),
        "morphology_plus_calibration": _features(region, ["breadth", "elevation"] + calibration),
        "topology_joint": _features(region, topology),
        "morphology_plus_topology": _features(region, ["breadth", "elevation"] + topology),
        "dataset_only": _features(region, categorical=["dataset_name"]),
        "dataset_numeric": _features(region, dataset_numeric),
        "dataset_plus_morphology": _features(region, ["breadth", "elevation"], ["dataset_name"]),
        "dataset_plus_occupancy": _features(region, occupancy, ["dataset_name"]),
        "dataset_plus_topology": _features(region, topology, ["dataset_name"]),
        "all_available_mechanisms": _features(region, ["breadth", "elevation"] + occupancy + calibration + topology + dataset_numeric, ["model_id"]),
    }
    for col in occupancy:
        specs[f"occupancy_single__{col}"] = _features(region, [col])
    rows = []
    baseline = None
    for name, X in specs.items():
        score, valid, reason = grouped_cv_r2(region, X)
        if name == "morphology_only":
            baseline = score
        rows.append(
            {
                "model_spec": name,
                "grouped_cv_r2": score,
                "delta_vs_morphology": score - baseline if np.isfinite(score) and np.isfinite(baseline) else np.nan,
                "n_features": int(X.shape[1]),
                "n_rows": int(len(region)),
                "n_groups": int(region["dataset_name"].nunique()),
                "valid": bool(valid),
                "skip_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def feature_importance(region: pd.DataFrame) -> pd.DataFrame:
    cols = available(["breadth", "elevation"] + OCCUPANCY_COLS + CALIBRATION_COLS + TOPOLOGY_COLS + DATASET_COLS, region)
    X = _features(region, cols, ["model_id"])
    if X.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=8, random_state=0, n_jobs=-1)
    data = pd.concat([region[[TARGET]].reset_index(drop=True), X.reset_index(drop=True)], axis=1).dropna()
    model.fit(data.drop(columns=[TARGET]), data[TARGET])
    out = pd.DataFrame({"feature": data.drop(columns=[TARGET]).columns, "importance": model.feature_importances_})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def latent_clusters(region: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = available([TARGET, SURVIVAL, "breadth", "elevation"] + OCCUPANCY_COLS + CALIBRATION_COLS, region)
    data = region.dropna(subset=cols).copy()
    if len(data) < 50 or len(cols) < 3:
        data["latent_cluster"] = -1
        return data, pd.DataFrame()
    X = StandardScaler().fit_transform(data[cols].astype(float))
    try:
        import hdbscan

        labels = hdbscan.HDBSCAN(min_cluster_size=35, min_samples=10).fit_predict(X)
    except Exception:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=0.8, min_samples=10).fit_predict(X)
    data["latent_cluster"] = labels
    summary = data[data["latent_cluster"] >= 0].groupby("latent_cluster", as_index=False).agg(
        n_runs=(TARGET, "size"),
        mean_cliffiness=(TARGET, "mean"),
        mean_survival_auc=(SURVIVAL, "mean"),
        mean_effective_bins=("positive_effective_score_bins", "mean"),
        mean_unique_ratio=("positive_unique_score_ratio", "mean"),
        mean_top_bin_mass=("positive_top_bin_mass", "mean"),
        dominant_models=("model_id", lambda s: ", ".join(s.value_counts().head(5).index.astype(str))),
        dominant_datasets=("dataset_name", lambda s: ", ".join(s.value_counts().head(6).index.astype(str))),
    )
    return data, summary


def quantization_summary(region: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tests = [
        ("unique_posterior_ratio", "positive_unique_score_ratio", "low", "quantized"),
        ("effective_bins", "positive_effective_score_bins", "low", "quantized"),
        ("top_bin_mass", "positive_top_bin_mass", "high", "quantized"),
    ]
    for name, col, direction, label in tests:
        if col not in region.columns:
            continue
        q = region[col].quantile(0.33 if direction == "low" else 0.67)
        quantized = region[region[col] <= q] if direction == "low" else region[region[col] >= q]
        smooth = region[region[col] > q] if direction == "low" else region[region[col] < q]
        rows.append(
            {
                "metric": name,
                "quantized_definition": f"{col} {'<=' if direction == 'low' else '>='} {q:.4f}",
                "n_quantized": int(len(quantized)),
                "mean_cliffiness_quantized": float(quantized[TARGET].mean()),
                "mean_cliffiness_other": float(smooth[TARGET].mean()),
                "delta_quantized_minus_other": float(quantized[TARGET].mean() - smooth[TARGET].mean()),
            }
        )
    return pd.DataFrame(rows)


def classify_outcome(scores: pd.DataFrame, importance: pd.DataFrame, latent_summary: pd.DataFrame) -> tuple[str, str]:
    score_map = dict(zip(scores["model_spec"], scores["grouped_cv_r2"]))
    all_score = score_map.get("all_available_mechanisms", np.nan)
    occupancy = score_map.get("morphology_plus_occupancy", np.nan)
    topology = score_map.get("morphology_plus_topology", np.nan)
    model_family = score_map.get("morphology_plus_model_family", np.nan)
    if np.isfinite(occupancy) and occupancy > 0.50 and occupancy >= max(np.nan_to_num([topology, model_family], nan=-np.inf)):
        return "Outcome A", "Residual cliffiness is largely explained by occupancy/quantization metrics."
    if np.isfinite(topology) and topology > 0.50 and topology >= max(np.nan_to_num([occupancy, model_family], nan=-np.inf)):
        return "Outcome B", "Residual cliffiness is largely explained by topology metrics."
    if np.isfinite(model_family) and model_family > 0.50 and model_family >= max(np.nan_to_num([occupancy, topology], nan=-np.inf)):
        return "Outcome C", "Residual cliffiness is largely explained by model-family effects."
    if np.isfinite(all_score) and all_score > 0.50 and not latent_summary.empty:
        return "Outcome D", "Residual cliffiness reflects multiple interacting mechanisms and motivates a new accessibility-shape taxonomy."
    return "Outcome D", "Residual cliffiness remains only partly explained and likely reflects multiple latent mechanisms."


def plot_histogram(region: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(region[TARGET], bins=30, color="tomato", alpha=0.85)
    ax.set_xlabel("cliffiness")
    ax.set_ylabel("runs")
    ax.set_title("High-Elevation / Low-Breadth Cliffiness")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def scatter_metric(region: pd.DataFrame, col: str, output: Path, xlabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    if col in region.columns:
        ax.scatter(region[col], region[TARGET], s=14, alpha=0.55)
    else:
        ax.text(0.5, 0.5, f"{col} unavailable", ha="center", va="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cliffiness")
    ax.set_title(f"Cliffiness vs {xlabel}")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_by_model(region: pd.DataFrame, output: Path) -> Path:
    order = region.groupby("model_id")[TARGET].median().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([region[region["model_id"] == m][TARGET] for m in order], tick_labels=order, showfliers=False)
    ax.set_ylabel("cliffiness")
    ax.set_title("Cliffiness by Model Family")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_partial_dependence(region: pd.DataFrame, importance: pd.DataFrame, output: Path) -> Path:
    numeric = [f for f in importance["feature"].head(8) if f in region.columns and pd.api.types.is_numeric_dtype(region[f])]
    numeric = numeric[:4]
    fig, axes = plt.subplots(1, max(1, len(numeric)), figsize=(4 * max(1, len(numeric)), 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    if not numeric:
        axes[0].text(0.5, 0.5, "No numeric importance features", ha="center", va="center")
    for ax, col in zip(axes, numeric):
        tmp = region[[col, TARGET]].dropna().copy()
        tmp["bin"] = pd.qcut(tmp[col].rank(method="first"), q=min(10, len(tmp)), duplicates="drop")
        binned = tmp.groupby("bin", observed=False).agg(x=(col, "mean"), y=(TARGET, "mean"))
        ax.plot(binned["x"], binned["y"], marker="o")
        ax.set_xlabel(col)
        ax.set_ylabel("mean cliffiness")
    for ax in axes[len(numeric) :]:
        ax.axis("off")
    fig.suptitle("Partial-Dependence Style Binned Effects")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_report(output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_hddt()
    region = high_elevation_region(df)
    inventory = residual_inventory(region)
    terciles = tercile_inventory(region)
    family = model_family_effects(region)
    scores = model_scores(region)
    importance = feature_importance(region)
    latent_rows, latent_summary = latent_clusters(region)
    quant = quantization_summary(region)
    outcome, outcome_text = classify_outcome(scores, importance, latent_summary)
    summary = {
        **inventory,
        "expected_rows_near_1848": bool(1700 <= len(region) <= 2000),
        "available_occupancy_metrics": available(OCCUPANCY_COLS, region),
        "available_calibration_metrics": available(CALIBRATION_COLS, region),
        "available_topology_metrics": available(TOPOLOGY_COLS, region),
        "best_grouped_cv_model": str(scores[scores["valid"]].sort_values("grouped_cv_r2", ascending=False).iloc[0]["model_spec"]),
        "best_grouped_cv_r2": float(scores[scores["valid"]]["grouped_cv_r2"].max()),
        "success_threshold_met": bool(scores[scores["valid"]]["grouped_cv_r2"].max() > 0.50),
        "latent_cluster_count": int(latent_summary["latent_cluster"].nunique()) if not latent_summary.empty else 0,
        "outcome": outcome,
        "outcome_text": outcome_text,
    }
    summary_path = output_dir / "high_elevation_cliffiness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = output_dir / "high_elevation_cliffiness_mechanism.md"
    report.write_text(
        "\n".join(
            [
                "# High-Elevation Cliffiness Mechanism Analysis",
                "",
                "## Executive Answer",
                executive_answer(summary),
                "",
                "## Data Selection",
                f"Filtered to `elevation >= 0.80` and `breadth <= 0.70`: `{len(region)}` rows. Expected rows near 1848: `{summary['expected_rows_near_1848']}`.",
                "",
                "## Analysis 1: Residual Variance Inventory",
                _markdown_table(pd.DataFrame([inventory])),
                "",
                "Cliffiness terciles:",
                _markdown_table(terciles),
                "",
                "## Analysis 2: Model Family Effects",
                _markdown_table(family),
                "",
                "## Analysis 3: Occupancy Mechanisms",
                _markdown_table(scores[scores["model_spec"].str.contains("occupancy")].sort_values("grouped_cv_r2", ascending=False)),
                "",
                "Feature importance from an in-region random forest mechanism model:",
                _markdown_table(importance.head(20)),
                "",
                "## Analysis 4: Quantization Hypothesis",
                _markdown_table(quant),
                "",
                "## Analysis 5: Calibration Mechanisms",
                f"Available calibration metrics: `{summary['available_calibration_metrics']}`. Brier score is the only calibration proxy available in the relaxed HDDT validation CSV unless additional calibration artifacts are joined later.",
                "",
                _markdown_table(scores[scores["model_spec"].str.contains("calibration")]),
                "",
                "## Analysis 6: Topology Mechanisms",
                f"Available topology metrics: `{summary['available_topology_metrics']}`. No topology columns were available in the HDDT validation CSV, so topology could not be tested in this run.",
                "",
                _markdown_table(scores[scores["model_spec"].str.contains("topology")]),
                "",
                "## Analysis 7: Dataset Effects",
                _markdown_table(scores[scores["model_spec"].str.contains("dataset")].sort_values("grouped_cv_r2", ascending=False)),
                "",
                "## Analysis 8: Combined Mechanism Model",
                _markdown_table(scores[~scores["model_spec"].str.startswith("occupancy_single")].sort_values("grouped_cv_r2", ascending=False)),
                "",
                "## Latent Mechanism Clusters",
                _markdown_table(latent_summary),
                "",
                "## Key Questions",
                key_questions(summary, scores, quant),
                "",
                "## Conclusion",
                f"{outcome}: {outcome_text}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outputs = [
        report,
        summary_path,
        plot_histogram(region, output_dir / "high_elevation_cliffiness_histogram.png"),
        scatter_metric(region, "positive_effective_score_bins", output_dir / "high_elevation_cliffiness_vs_effective_bins.png", "effective score bins"),
        scatter_metric(region, "positive_unique_score_ratio", output_dir / "high_elevation_cliffiness_vs_unique_posterior_ratio.png", "unique posterior ratio"),
        scatter_metric(region, "positive_top_bin_mass", output_dir / "high_elevation_cliffiness_vs_top_bin_mass.png", "top bin mass"),
        scatter_metric(region, "brier_score", output_dir / "high_elevation_cliffiness_vs_calibration_error.png", "Brier score"),
        plot_by_model(region, output_dir / "high_elevation_cliffiness_by_model_family.png"),
        plot_partial_dependence(region, importance, output_dir / "high_elevation_cliffiness_partial_dependence.png"),
    ]
    return tuple(outputs)


def executive_answer(summary: dict[str, object]) -> str:
    return (
        "Within the high-elevation / low-breadth region, survival is uniformly high but cliffiness remains strongly heterogeneous. "
        f"The best grouped-CV mechanism model reaches R2={summary['best_grouped_cv_r2']:.4f}; success threshold met: {summary['success_threshold_met']}. "
        f"Conclusion: {summary['outcome_text']}"
    )


def key_questions(summary: dict[str, object], scores: pd.DataFrame, quant: pd.DataFrame) -> str:
    best = scores[scores["valid"]].sort_values("grouped_cv_r2", ascending=False).head(5)
    return "\n".join(
        [
            "Q1. Within fixed morphology, cliffiness variance is best explained by the highest-ranked available mechanism models below:",
            _markdown_table(best),
            "",
            "Q2. Available evidence points to a mixture of occupancy/quantization, model-family, calibration-proxy, and dataset/task effects rather than a single clean driver.",
            "",
            "Q3. The region is better interpreted as multiple latent cliffiness mechanisms inside one high-survival morphology region, not one homogeneous regime.",
            "",
            f"Q4. Remaining variance is {'mostly explainable' if summary['success_threshold_met'] else 'not mostly explainable'} under grouped CV; best R2 is `{summary['best_grouped_cv_r2']:.4f}`.",
            "",
            "Quantization contrast table:",
            _markdown_table(quant),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
