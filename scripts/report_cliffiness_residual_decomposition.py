"""Decompose cliffiness residuals after observed reachability-topology prediction."""

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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
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
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"

JUMP_FEATURES = ["first_major_drop_threshold", "largest_drop_threshold", "median_drop_threshold", "weighted_drop_center"]
CONCENTRATION_FEATURES = ["largest_drop", "top_3_drop_mass", "top_5_drop_mass", "drop_gini", "effective_drop_count"]
CURVATURE_FEATURES = ["mean_slope", "max_slope", "slope_variance", "mean_curvature", "max_curvature", "curvature_entropy"]
COMPRESSION_FEATURES = ["threshold_span_50pct_loss", "threshold_span_25pct_loss", "threshold_span_10pct_loss", "compression_ratio"]
ALPHABET_FEATURES = ["n_unique_posteriors_proxy", "unique_ratio_proxy", "posterior_entropy_proxy", "posterior_gini_proxy"]
CALIBRATION_FEATURES = ["brier_score"]


def drop_gini(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals >= 0)]
    if vals.size == 0 or vals.sum() <= 0:
        return 0.0
    sorted_vals = np.sort(vals)
    n = sorted_vals.size
    return float((2 * np.arange(1, n + 1) @ sorted_vals) / (n * sorted_vals.sum()) - (n + 1) / n)


def entropy_from_weights(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    total = vals.sum()
    if vals.size == 0 or total <= 0:
        return 0.0
    p = vals / total
    return float(-(p * np.log(p)).sum())


def span_for_loss(thresholds: np.ndarray, curve: np.ndarray, loss_fraction: float) -> float:
    total_loss = float(curve[0] - curve[-1])
    if total_loss <= 0:
        return 0.0
    target = total_loss * float(loss_fraction)
    drops = np.maximum(0.0, curve[:-1] - curve[1:])
    if drops.sum() <= 0:
        return 0.0
    best = float(thresholds[-1] - thresholds[0])
    for start in range(drops.size):
        mass = 0.0
        for end in range(start, drops.size):
            mass += drops[end]
            if mass >= target:
                best = min(best, float(thresholds[end + 1] - thresholds[start]))
                break
    return best


def curve_features(matrix: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    t = np.asarray(thresholds, dtype=float)
    dt = np.diff(t)
    for _, row in matrix.iterrows():
        curve = row.to_numpy(dtype=float)
        drops = np.maximum(0.0, curve[:-1] - curve[1:])
        total_drop = float(drops.sum())
        drop_positions = t[1:]
        major_cut = max(0.05, 0.25 * float(drops.max()) if drops.size else 0.0)
        major = np.where(drops >= major_cut)[0]
        first_major = float(drop_positions[major[0]]) if major.size else np.nan
        largest_idx = int(np.argmax(drops)) if drops.size else 0
        largest_threshold = float(drop_positions[largest_idx]) if drops.size else np.nan
        if total_drop > 0:
            cdf = np.cumsum(drops) / total_drop
            median_threshold = float(drop_positions[np.searchsorted(cdf, 0.5)])
            weighted_center = float(np.sum(drop_positions * drops) / total_drop)
            effective_drop_count = float(np.exp(entropy_from_weights(drops)))
        else:
            median_threshold = np.nan
            weighted_center = np.nan
            effective_drop_count = 0.0
        slope = np.diff(curve) / np.maximum(dt, 1e-12)
        curvature = np.diff(slope) / np.maximum(dt[1:], 1e-12) if slope.size > 1 else np.asarray([], dtype=float)
        unique_vals = np.unique(np.round(curve, 6))
        rows.append(
            {
                "first_major_drop_threshold": first_major,
                "largest_drop_threshold": largest_threshold,
                "median_drop_threshold": median_threshold,
                "weighted_drop_center": weighted_center,
                "largest_drop": float(drops.max()) if drops.size else 0.0,
                "top_3_drop_mass": float(np.sort(drops)[-3:].sum()) if drops.size else 0.0,
                "top_5_drop_mass": float(np.sort(drops)[-5:].sum()) if drops.size else 0.0,
                "drop_gini": drop_gini(drops),
                "effective_drop_count": effective_drop_count,
                "mean_slope": float(np.mean(np.abs(slope))) if slope.size else 0.0,
                "max_slope": float(np.max(np.abs(slope))) if slope.size else 0.0,
                "slope_variance": float(np.var(slope)) if slope.size else 0.0,
                "mean_curvature": float(np.mean(np.abs(curvature))) if curvature.size else 0.0,
                "max_curvature": float(np.max(np.abs(curvature))) if curvature.size else 0.0,
                "curvature_entropy": entropy_from_weights(np.abs(curvature)),
                "threshold_span_50pct_loss": span_for_loss(t, curve, 0.50),
                "threshold_span_25pct_loss": span_for_loss(t, curve, 0.25),
                "threshold_span_10pct_loss": span_for_loss(t, curve, 0.10),
                "compression_ratio": float(total_drop / max(1e-9, span_for_loss(t, curve, 0.50))),
                "n_unique_posteriors_proxy": int(unique_vals.size),
                "unique_ratio_proxy": float(unique_vals.size / max(1, curve.size)),
                "posterior_entropy_proxy": entropy_from_weights(np.bincount(np.searchsorted(unique_vals, np.round(curve, 6)), minlength=unique_vals.size)),
                "posterior_gini_proxy": drop_gini(np.bincount(np.searchsorted(unique_vals, np.round(curve, 6)), minlength=unique_vals.size)),
            }
        )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def grouped_regression(meta: pd.DataFrame, X: pd.DataFrame | np.ndarray, target: str, group_col: str = "dataset_id") -> tuple[np.ndarray, float, bool, str]:
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


def model_feature_group(meta: pd.DataFrame, features: pd.DataFrame, feature_cols: list[str], target: str) -> dict[str, object]:
    cols = [c for c in feature_cols if c in features.columns and features[c].notna().any()]
    if not cols:
        return {"feature_group": target, "residual_r2": np.nan, "n_features": 0, "valid": False, "skip_reason": "no_features"}
    _, score, valid, reason = grouped_regression(meta, features[cols], "cliffiness_residual")
    return {"feature_group": target, "residual_r2": score, "n_features": len(cols), "valid": valid, "skip_reason": reason}


def join_optional_features(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    if not HDDT_VALIDATION.exists():
        return out
    hddt = pd.read_csv(HDDT_VALIDATION)
    if "fit_failed" in hddt.columns:
        hddt = hddt[~hddt["fit_failed"].astype(str).str.lower().isin(["true", "1", "yes"])]
    hddt = hddt.copy()
    hddt["run_id"] = hddt["task_id"].astype(str) + "|" + hddt["model_id"].astype(str) + "|seed=" + hddt["seed"].astype(int).astype(str) + "|split=" + hddt["split_id"].astype(int).astype(str)
    optional = ["run_id", "brier_score", "positive_unique_score_ratio", "positive_histogram_entropy", "positive_score_gini_or_concentration_index"]
    optional = [c for c in optional if c in hddt.columns]
    return out.merge(hddt[optional], on="run_id", how="left", suffixes=("", "_joined"))


def residual_models(meta: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    joined = join_optional_features(meta)
    all_features = features.copy()
    for col in ["brier_score", "positive_unique_score_ratio", "positive_histogram_entropy", "positive_score_gini_or_concentration_index"]:
        if col in joined.columns:
            all_features[col] = joined[col].to_numpy()
    groups = {
        "jump_features": JUMP_FEATURES,
        "concentration_features": CONCENTRATION_FEATURES,
        "curvature_features": CURVATURE_FEATURES,
        "compression_features": COMPRESSION_FEATURES,
        "alphabet_features": ALPHABET_FEATURES + ["positive_unique_score_ratio", "positive_histogram_entropy", "positive_score_gini_or_concentration_index"],
        "calibration_features": CALIBRATION_FEATURES,
        "all_geometry_features": JUMP_FEATURES + CONCENTRATION_FEATURES + CURVATURE_FEATURES + COMPRESSION_FEATURES + ALPHABET_FEATURES,
        "all_available_features": JUMP_FEATURES + CONCENTRATION_FEATURES + CURVATURE_FEATURES + COMPRESSION_FEATURES + ALPHABET_FEATURES + ["positive_unique_score_ratio", "positive_histogram_entropy", "positive_score_gini_or_concentration_index", "brier_score"],
    }
    rows = [model_feature_group(meta, all_features, cols, name) for name, cols in groups.items()]
    return pd.DataFrame(rows), all_features


def permutation_table(meta: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in features.columns if features[c].notna().any()]
    data = pd.concat([meta[["cliffiness_residual"]].reset_index(drop=True), features[cols].reset_index(drop=True)], axis=1).dropna()
    if len(data) < 50:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    X = data.drop(columns=["cliffiness_residual"])
    y = data["cliffiness_residual"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=6, random_state=0, n_jobs=-1).fit(X_train, y_train)
    imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
    return pd.DataFrame({"feature": X.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std}).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def variance_partition(topology_r2: float, residual_scores: pd.DataFrame) -> dict[str, float]:
    geometry = float(residual_scores[residual_scores["feature_group"] == "all_geometry_features"]["residual_r2"].iloc[0]) if "all_geometry_features" in set(residual_scores["feature_group"]) else np.nan
    calibration = float(residual_scores[residual_scores["feature_group"] == "calibration_features"]["residual_r2"].iloc[0]) if "calibration_features" in set(residual_scores["feature_group"]) else np.nan
    topology_fraction = max(0.0, min(1.0, topology_r2))
    geometry_fraction = max(0.0, (1.0 - topology_fraction) * max(0.0, geometry if np.isfinite(geometry) else 0.0))
    calibration_fraction = max(0.0, (1.0 - topology_fraction) * max(0.0, calibration if np.isfinite(calibration) else 0.0))
    unexplained = max(0.0, 1.0 - topology_fraction - geometry_fraction - calibration_fraction)
    return {
        "topology_fraction": topology_fraction,
        "geometry_fraction": geometry_fraction,
        "calibration_fraction": calibration_fraction,
        "unexplained_fraction": unexplained,
        "topology_r2": topology_r2,
        "geometry_residual_r2": geometry,
        "calibration_residual_r2": calibration,
    }


def plot_scatter(meta: pd.DataFrame, features: pd.DataFrame, col: str, output: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    if col in features.columns:
        ax.scatter(features[col], meta["cliffiness_residual"], s=12, alpha=0.5)
    ax.set_xlabel(col)
    ax.set_ylabel("cliffiness residual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_importance(importance: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(12).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], color="steelblue")
    ax.set_title("Residual Feature Importance")
    ax.set_xlabel("permutation importance")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_partition(partition: dict[str, float], output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    keys = ["topology_fraction", "geometry_fraction", "calibration_fraction", "unexplained_fraction"]
    ax.bar(keys, [partition[k] for k in keys], color=["#4c78a8", "#f58518", "#54a24b", "#bab0ac"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("variance fraction")
    ax.set_title("Cliffiness Variance Partition")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_umap(meta: pd.DataFrame, features: pd.DataFrame, output: Path) -> Path:
    X = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    try:
        import umap
        emb = umap.UMAP(n_neighbors=25, min_dist=0.05, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    except Exception:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=meta["cliffiness_residual"], cmap="coolwarm", s=12, alpha=0.75)
    fig.colorbar(sc, ax=ax, label="cliffiness residual")
    ax.set_title("Topology vs Geometry Residual Manifold")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_report(output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = load_curve_files([HDDT_CURVES, CLASSICAL_CURVES])
    meta, matrix, thresholds = pivot_curves(curves)
    pred, topology_r2, valid, reason = grouped_regression(meta, matrix, TARGET)
    meta = meta.copy()
    meta["predicted_cliffiness_from_topology"] = pred
    meta["cliffiness_residual"] = meta[TARGET] - meta["predicted_cliffiness_from_topology"]
    features = curve_features(matrix, thresholds)
    residual_scores, all_features = residual_models(meta, features)
    importance = permutation_table(meta, all_features)
    partition = variance_partition(topology_r2, residual_scores)
    residuals_path = output_dir / "cliffiness_residuals.csv"
    meta.to_csv(residuals_path, index=False)
    importance_path = output_dir / "residual_feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    partition_path = output_dir / "variance_partition.json"
    partition_path.write_text(json.dumps(partition, indent=2, sort_keys=True), encoding="utf-8")
    best_residual = residual_scores[residual_scores["valid"]].sort_values("residual_r2", ascending=False).iloc[0]
    outcome = "geometry_supported" if float(best_residual["residual_r2"]) > 0.50 else "residual_mostly_noise_or_unresolved"
    report = output_dir / "cliffiness_residual_decomposition.md"
    report.write_text("\n".join([
        "# Cliffiness Residual Decomposition",
        "",
        "## Executive Answer",
        f"Topology-only observed-curve cliffiness R2 is `{topology_r2:.4f}`. The best residual model is `{best_residual['feature_group']}` with residual R2 `{best_residual['residual_r2']:.4f}`. Outcome: `{outcome}`.",
        "",
        "## Residual Model Scores",
        _markdown_table(residual_scores),
        "",
        "## Residual Feature Importance",
        _markdown_table(importance.head(20)),
        "",
        "## Variance Partition",
        _markdown_table(pd.DataFrame([partition])),
        "",
        "## Interpretation",
        "Strong evidence for a distinct Accessibility Geometry layer requires residual R2 > 0.50 after topology prediction. Treat negative or weak residual R2 as unresolved residual variation rather than proof of noise.",
    ]) + "\n", encoding="utf-8")
    return (
        report,
        plot_scatter(meta, all_features, "max_curvature", output_dir / "cliffiness_residual_vs_curvature.png", "Residual vs Curvature"),
        plot_scatter(meta, all_features, "compression_ratio", output_dir / "cliffiness_residual_vs_compression.png", "Residual vs Compression"),
        plot_scatter(meta, all_features, "unique_ratio_proxy", output_dir / "cliffiness_residual_vs_alphabet.png", "Residual vs Alphabet Proxy"),
        plot_importance(importance, output_dir / "residual_feature_importance.png"),
        plot_partition(partition, output_dir / "cliffiness_variance_partition.png"),
        plot_umap(meta, all_features, output_dir / "topology_vs_geometry_umap.png"),
        residuals_path,
        importance_path,
        partition_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
