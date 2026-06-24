"""Test whether posterior concentration explains cliffiness residuals."""

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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_definitive_reachability_topology import HDDT_CURVES, CLASSICAL_CURVES, load_curve_files, pivot_curves  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402


OUT = ROOT / "reports" / "topology"
HDDT_VALIDATION = ROOT / "results" / "real_validation" / "hddt_accessibility_validation_relaxed.csv"
CLASSICAL_MORPHOLOGY = ROOT / "results" / "topology" / "classical_allocator_morphology.csv"
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"
CONCENTRATION_FEATURES = [
    "positive_score_entropy",
    "positive_score_gini",
    "top_1pct_mass",
    "top_5pct_mass",
    "top_10pct_mass",
    "p95_to_median_score_ratio",
    "p99_to_p50_score_ratio",
    "effective_support",
    "density_around_090",
    "density_around_095",
    "density_around_099",
    "tail_mass_above_99pct",
    "joined_positive_score_gini_or_concentration_index",
    "joined_positive_unique_score_ratio",
    "joined_positive_histogram_entropy",
    "joined_positive_effective_score_bins",
    "joined_positive_top_bin_mass",
    "joined_brier_score",
]


def entropy(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    total = w.sum()
    if w.size == 0 or total <= 0:
        return 0.0
    p = w / total
    return float(-(p * np.log(p)).sum())


def gini(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w >= 0)]
    if w.size == 0 or w.sum() <= 0:
        return 0.0
    w = np.sort(w)
    n = w.size
    return float((2 * np.arange(1, n + 1) @ w) / (n * w.sum()) - (n + 1) / n)


def score_distribution_from_curve(curve: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    drops = np.maximum(0.0, curve[:-1] - curve[1:])
    scores = thresholds[1:]
    tail = max(0.0, float(curve[-1]))
    if tail > 0:
        scores = np.append(scores, thresholds[-1])
        drops = np.append(drops, tail)
    total = drops.sum()
    if total <= 0:
        return np.asarray([0.0]), np.asarray([1.0])
    return scores, drops / total


def weighted_quantile(scores: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(scores)
    s = scores[order]
    w = weights[order]
    cdf = np.cumsum(w) / max(1e-12, w.sum())
    return float(s[np.searchsorted(cdf, q, side="left").clip(0, len(s) - 1)])


def top_mass(scores: np.ndarray, weights: np.ndarray, frac: float) -> float:
    order = np.argsort(scores)[::-1]
    w = weights[order]
    if w.size == 0:
        return 0.0
    # Threshold-grid proxy for top-k observations: mass in the top fraction of score bins.
    n = max(1, int(np.ceil(frac * len(w))))
    return float(w[:n].sum())


def local_density(scores: np.ndarray, weights: np.ndarray, threshold: float, bandwidth: float = 0.025) -> float:
    mask = np.abs(scores - threshold) <= bandwidth
    return float(weights[mask].sum() / max(1e-9, 2 * bandwidth))


def concentration_features(matrix: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for _, row in matrix.iterrows():
        curve = row.to_numpy(dtype=float)
        scores, weights = score_distribution_from_curve(curve, thresholds)
        p50 = max(1e-9, weighted_quantile(scores, weights, 0.50))
        p95 = weighted_quantile(scores, weights, 0.95)
        p99 = weighted_quantile(scores, weights, 0.99)
        rows.append(
            {
                "positive_score_entropy": entropy(weights),
                "positive_score_gini": gini(weights),
                "top_1pct_mass": top_mass(scores, weights, 0.01),
                "top_5pct_mass": top_mass(scores, weights, 0.05),
                "top_10pct_mass": top_mass(scores, weights, 0.10),
                "p95_to_median_score_ratio": float(p95 / p50),
                "p99_to_p50_score_ratio": float(p99 / p50),
                "effective_support": float(1.0 / max(1e-12, np.sum(weights**2))),
                "density_around_090": local_density(scores, weights, 0.90),
                "density_around_095": local_density(scores, weights, 0.95),
                "density_around_099": local_density(scores, weights, 0.99),
                "tail_mass_above_99pct": float(weights[scores >= p99].sum()),
            }
        )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def join_existing_metrics(meta: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    joined = meta[["run_id"]].copy()
    frames = []
    if HDDT_VALIDATION.exists():
        hddt = pd.read_csv(HDDT_VALIDATION)
        if "fit_failed" in hddt.columns:
            hddt = hddt[~hddt["fit_failed"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if {"task_id", "model_id", "seed", "split_id"}.issubset(hddt.columns):
            hddt = hddt.copy()
            hddt["run_id"] = hddt["task_id"].astype(str) + "|" + hddt["model_id"].astype(str) + "|seed=" + hddt["seed"].astype(int).astype(str) + "|split=" + hddt["split_id"].astype(int).astype(str)
            frames.append(hddt)
    if CLASSICAL_MORPHOLOGY.exists():
        classical = pd.read_csv(CLASSICAL_MORPHOLOGY)
        if {"model_id", "seed", "skew_ratio"}.issubset(classical.columns):
            classical = classical.copy()
            classical["run_id"] = "classical|skew=" + classical["skew_ratio"].astype(int).astype(str) + "|model=" + classical["model_id"].astype(str) + "|seed=" + classical["seed"].astype(int).astype(str)
            frames.append(classical)
    if not frames:
        return out
    metrics = pd.concat(frames, ignore_index=True)
    cols = [
        "run_id",
        "positive_score_gini_or_concentration_index",
        "positive_unique_score_ratio",
        "positive_histogram_entropy",
        "positive_effective_score_bins",
        "positive_top_bin_mass",
        "brier_score",
    ]
    cols = [c for c in cols if c in metrics.columns]
    merged = joined.merge(metrics[cols], on="run_id", how="left")
    for col in cols:
        if col != "run_id":
            out[f"joined_{col}"] = merged[col].to_numpy()
    return out


def grouped_predict(meta: pd.DataFrame, X: pd.DataFrame | np.ndarray, target: str, group_col: str = "dataset_id") -> tuple[np.ndarray, float, bool, str]:
    Xdf = pd.DataFrame(X).reset_index(drop=True)
    Xdf.columns = [str(c) for c in Xdf.columns]
    data = pd.concat([meta[[target, group_col]].reset_index(drop=True), Xdf], axis=1).dropna()
    pred = np.full(len(meta), np.nan)
    if len(data) < 20:
        return pred, np.nan, False, "too_few_rows"
    groups = data[group_col].astype(str)
    if groups.nunique() < 2:
        return pred, np.nan, False, "too_few_groups"
    try:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
        Xdata = data.drop(columns=[target, group_col])
        models = [
            make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)),
            RandomForestRegressor(n_estimators=250, min_samples_leaf=4, random_state=0, n_jobs=-1),
        ]
        best_pred = None
        best_score = -np.inf
        for model in models:
            candidate = cross_val_predict(model, Xdata, data[target], cv=cv, groups=groups)
            score = float(r2_score(data[target], candidate))
            if score > best_score:
                best_score = score
                best_pred = candidate
        pred[data.index.to_numpy()] = best_pred
        return pred, float(best_score), True, ""
    except Exception as exc:
        return pred, np.nan, False, type(exc).__name__


def residual_regression(meta: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in CONCENTRATION_FEATURES if c in features.columns and features[c].notna().any()]
    groups = {
        "curve_concentration": [c for c in cols if not c.startswith("joined_")],
        "joined_concentration": [c for c in cols if c.startswith("joined_")],
        "all_concentration": cols,
    }
    rows = []
    predictions = pd.DataFrame(index=meta.index)
    for name, group_cols in groups.items():
        if not group_cols:
            rows.append({"feature_group": name, "residual_r2": np.nan, "n_features": 0, "valid": False, "skip_reason": "no_features"})
            continue
        pred, score, valid, reason = grouped_predict(meta, features[group_cols], "cliffiness_residual")
        predictions[name] = pred
        rows.append({"feature_group": name, "residual_r2": score, "n_features": len(group_cols), "valid": valid, "skip_reason": reason})
    return pd.DataFrame(rows), predictions


def feature_importance(meta: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CONCENTRATION_FEATURES if c in features.columns and features[c].notna().any()]
    data = pd.concat([meta[["cliffiness_residual"]].reset_index(drop=True), features[cols].reset_index(drop=True)], axis=1).dropna()
    if len(data) < 50 or not cols:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    X = data.drop(columns=["cliffiness_residual"])
    y = data["cliffiness_residual"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=6, random_state=0, n_jobs=-1).fit(X_train, y_train)
    imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
    return pd.DataFrame({"feature": X.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std}).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def variance_partition(topology_r2: float, concentration_r2: float, calibration_r2: float) -> dict[str, float]:
    top = max(0.0, min(1.0, topology_r2))
    conc = max(0.0, (1.0 - top) * max(0.0, concentration_r2 if np.isfinite(concentration_r2) else 0.0))
    cal = max(0.0, (1.0 - top) * max(0.0, calibration_r2 if np.isfinite(calibration_r2) else 0.0))
    return {
        "topology_fraction": top,
        "concentration_fraction": conc,
        "calibration_fraction": cal,
        "unexplained_fraction": max(0.0, 1.0 - top - conc - cal),
        "topology_r2": topology_r2,
        "concentration_residual_r2": concentration_r2,
        "calibration_residual_r2": calibration_r2,
    }


def decide(residual_r2: float) -> tuple[str, str]:
    if residual_r2 > 0.50:
        return "Outcome A", "Posterior concentration explains accessibility fragility beyond reachability topology."
    if residual_r2 > 0.20:
        return "Outcome B", "Posterior concentration is a meaningful secondary modifier but not a distinct layer."
    return "Outcome C", "Remaining cliffiness variation is mostly stochastic or model-specific under these features."


def family_profiles(meta: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([meta[["model_id", TARGET, "cliffiness_residual"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    cols = [c for c in ["positive_score_entropy", "positive_score_gini", "top_5pct_mass", "effective_support", "joined_positive_score_gini_or_concentration_index"] if c in data.columns]
    agg = {"n_runs": (TARGET, "size"), "mean_cliffiness": (TARGET, "mean"), "mean_residual": ("cliffiness_residual", "mean")}
    for col in cols:
        agg[f"mean_{col}"] = (col, "mean")
    return data.groupby("model_id", as_index=False).agg(**agg).sort_values("mean_residual", ascending=False)


def plot_scatter(meta: pd.DataFrame, features: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ["positive_score_gini", "top_5pct_mass", "effective_support"], strict=True):
        ax.scatter(features[col], meta["cliffiness_residual"], s=12, alpha=0.5)
        ax.set_xlabel(col)
        ax.set_ylabel("cliffiness residual")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_umap(meta: pd.DataFrame, features: pd.DataFrame, output: Path) -> Path:
    cols = [c for c in CONCENTRATION_FEATURES if c in features.columns and features[c].notna().any()]
    X = features[cols].fillna(features[cols].median(numeric_only=True)).fillna(0.0)
    try:
        import umap
        emb = umap.UMAP(n_neighbors=25, min_dist=0.05, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    except Exception:
        emb = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=meta["cliffiness_residual"], cmap="coolwarm", s=12, alpha=0.75)
    fig.colorbar(sc, ax=ax, label="cliffiness residual")
    ax.set_title("Concentration UMAP")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_importance(importance: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(12).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], color="steelblue")
    ax.set_title("Concentration Feature Importance")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_partition(partition: dict[str, float], output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    keys = ["topology_fraction", "concentration_fraction", "calibration_fraction", "unexplained_fraction"]
    ax.bar(keys, [partition[k] for k in keys], color=["#4c78a8", "#f58518", "#54a24b", "#bab0ac"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("variance fraction")
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Cliffiness Variance Partition")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_family_profiles(profiles: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(profiles["model_id"], profiles["mean_residual"], color="tomato")
    axes[0].set_title("Mean Residual by Family")
    axes[0].tick_params(axis="x", rotation=25)
    ycol = "mean_positive_score_gini" if "mean_positive_score_gini" in profiles.columns else profiles.columns[-1]
    axes[1].bar(profiles["model_id"], profiles[ycol], color="steelblue")
    axes[1].set_title(ycol)
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_report(output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = load_curve_files([HDDT_CURVES, CLASSICAL_CURVES])
    meta, matrix, thresholds = pivot_curves(curves)
    topology_pred, topology_r2, _, _ = grouped_predict(meta, matrix, TARGET)
    meta = meta.copy()
    meta["predicted_cliffiness_from_topology"] = topology_pred
    meta["cliffiness_residual"] = meta[TARGET] - meta["predicted_cliffiness_from_topology"]
    features = join_existing_metrics(meta, concentration_features(matrix, thresholds))
    residual_scores, _ = residual_regression(meta, features)
    best = residual_scores[residual_scores["valid"]].sort_values("residual_r2", ascending=False).iloc[0]
    calibration_r2 = float(residual_scores[residual_scores["feature_group"] == "joined_concentration"]["residual_r2"].iloc[0]) if "joined_concentration" in set(residual_scores["feature_group"]) else np.nan
    partition = variance_partition(topology_r2, float(best["residual_r2"]), calibration_r2)
    outcome, outcome_text = decide(float(best["residual_r2"]))
    importance = feature_importance(meta, features)
    profiles = family_profiles(meta, features)
    summary = {
        "n_runs": int(len(meta)),
        "topology_r2": topology_r2,
        "best_residual_feature_group": str(best["feature_group"]),
        "best_residual_r2": float(best["residual_r2"]),
        "outcome": outcome,
        "outcome_text": outcome_text,
        **partition,
    }
    summary_path = output_dir / "posterior_concentration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = output_dir / "posterior_concentration_hypothesis.md"
    report.write_text("\n".join([
        "# Posterior Concentration Hypothesis",
        "",
        "## Executive Answer",
        f"Topology-only cliffiness R2 is `{topology_r2:.4f}`. Best concentration residual R2 is `{best['residual_r2']:.4f}` from `{best['feature_group']}`. Decision: **{outcome}**.",
        "",
        "## Residual Regression",
        _markdown_table(residual_scores),
        "",
        "## Feature Importance",
        _markdown_table(importance.head(20)),
        "",
        "## Variance Partition",
        _markdown_table(pd.DataFrame([partition])),
        "",
        "## Family Concentration Profiles",
        _markdown_table(profiles),
        "",
        "## Interpretation",
        outcome_text,
    ]) + "\n", encoding="utf-8")
    return (
        report,
        plot_scatter(meta, features, output_dir / "concentration_vs_residual.png"),
        plot_umap(meta, features, output_dir / "concentration_umap.png"),
        plot_importance(importance, output_dir / "concentration_feature_importance.png"),
        plot_partition(partition, output_dir / "concentration_variance_partition.png"),
        plot_family_profiles(profiles, output_dir / "family_concentration_profiles.png"),
        summary_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
