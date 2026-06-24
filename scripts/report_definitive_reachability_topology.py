"""Analyze observed empirical reachability curves as accessibility topology."""

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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402


HDDT_CURVES = ROOT / "results" / "real_validation" / "hddt_reachability_curves.csv"
CLASSICAL_CURVES = ROOT / "results" / "topology" / "classical_reachability_curves.csv"
OUT = ROOT / "reports" / "topology"
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"
SURROGATE = {
    "cliffiness_r2": 0.9691,
    "survival_r2": 0.9995,
    "breadth_r2": 0.8651,
    "elevation_r2": 0.9642,
    "regime_accuracy": 0.9459,
    "model_family_accuracy": 0.5562,
}


def load_curve_files(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No reachability curve files found")
    df = pd.concat(frames, ignore_index=True)
    required = ["run_id", "dataset_id", "task_id", "model_id", "seed", "threshold", "minority_reachability", SURVIVAL, TARGET, "breadth", "elevation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required reachability columns: {missing}")
    return df.dropna(subset=["run_id", "threshold", "minority_reachability"]).copy()


def pivot_curves(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    curves = curves.copy()
    curves["threshold_key"] = curves["threshold"].astype(float).round(6)
    matrix = curves.pivot_table(index="run_id", columns="threshold_key", values="minority_reachability", aggfunc="mean").sort_index(axis=1)
    meta_cols = ["run_id", "dataset_id", "task_id", "model_id", "seed", SURVIVAL, TARGET, "breadth", "elevation"]
    if "regime_id" in curves.columns:
        meta_cols.append("regime_id")
    meta = curves.sort_values("threshold").groupby("run_id", as_index=False)[meta_cols].first().set_index("run_id").loc[matrix.index].reset_index()
    return meta, matrix.reset_index(drop=True), matrix.columns.to_numpy(dtype=float)


def grouped_regression(meta: pd.DataFrame, X: pd.DataFrame | np.ndarray, target: str, group_col: str = "dataset_id") -> tuple[float, bool, str]:
    Xdf = pd.DataFrame(X).reset_index(drop=True)
    Xdf.columns = [str(c) for c in Xdf.columns]
    data = pd.concat([meta[[target, group_col]].reset_index(drop=True), Xdf], axis=1).dropna()
    if len(data) < 20:
        return np.nan, False, "too_few_rows"
    groups = data[group_col].astype(str)
    if groups.nunique() < 2:
        return np.nan, False, "too_few_groups"
    scores = []
    try:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
        Xdata = data.drop(columns=[target, group_col])
        for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=250, min_samples_leaf=4, random_state=0, n_jobs=-1)]:
            pred = cross_val_predict(model, Xdata, data[target], cv=cv, groups=groups)
            scores.append(float(r2_score(data[target], pred)))
        return max(scores), True, ""
    except Exception as exc:
        return np.nan, False, type(exc).__name__


def grouped_classifier(meta: pd.DataFrame, X: pd.DataFrame | np.ndarray, target: str, group_col: str = "dataset_id") -> tuple[float, float, float, bool, str]:
    if target not in meta.columns:
        return np.nan, np.nan, np.nan, False, "missing_target"
    Xdf = pd.DataFrame(X).reset_index(drop=True)
    Xdf.columns = [str(c) for c in Xdf.columns]
    data = pd.concat([meta[[target, group_col]].reset_index(drop=True), Xdf], axis=1).dropna()
    y = data[target].astype(str)
    groups = data[group_col].astype(str)
    if len(data) < 20 or y.nunique() < 2 or groups.nunique() < 2:
        return np.nan, np.nan, np.nan, False, "insufficient_data"
    baseline = float(y.value_counts(normalize=True).max())
    try:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, random_state=0, n_jobs=-1)
        pred = cross_val_predict(clf, data.drop(columns=[target, group_col]), y, cv=cv, groups=groups)
        return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro")), baseline, True, ""
    except Exception as exc:
        return np.nan, np.nan, baseline, False, type(exc).__name__


def embedding_and_clusters(matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = matrix.to_numpy(dtype=float)
    pca = PCA(n_components=2, random_state=0).fit_transform(X)
    try:
        import umap

        emb = umap.UMAP(n_neighbors=25, min_dist=0.05, random_state=0).fit_transform(X)
    except Exception:
        emb = pca
    try:
        import hdbscan

        labels = hdbscan.HDBSCAN(min_cluster_size=max(10, min(45, len(matrix) // 20)), min_samples=5).fit_predict(StandardScaler().fit_transform(X))
    except Exception:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(StandardScaler().fit_transform(X))
    return pca, emb, labels


def reconstruction_scores(meta: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in [SURVIVAL, TARGET, "breadth", "elevation"]:
        score, valid, reason = grouped_regression(meta, matrix, target)
        rows.append({"target": target, "metric": "r2", "score": score, "baseline": np.nan, "valid": valid, "skip_reason": reason})
    for target in ["regime_id", "model_id"]:
        acc, f1, baseline, valid, reason = grouped_classifier(meta, matrix, target)
        rows.append({"target": target, "metric": "accuracy", "score": acc, "macro_f1": f1, "baseline": baseline, "valid": valid, "skip_reason": reason})
    return pd.DataFrame(rows)


def cluster_profiles(meta: pd.DataFrame, matrix: pd.DataFrame, labels: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    data = meta.copy()
    data["topology_cluster"] = labels
    rows = []
    for cluster, group in data[data["topology_cluster"] >= 0].groupby("topology_cluster"):
        idx = group.index.to_numpy()
        mean_curve = matrix.iloc[idx].to_numpy(dtype=float).mean(axis=0)
        largest_drop = float(np.max(np.maximum(0.0, mean_curve[:-1] - mean_curve[1:]))) if mean_curve.size > 1 else 0.0
        rows.append(
            {
                "topology_cluster": int(cluster),
                "n_runs": int(len(group)),
                "mean_survival_auc": float(group[SURVIVAL].mean()),
                "mean_cliffiness": float(group[TARGET].mean()),
                "mean_breadth": float(group["breadth"].mean()),
                "mean_elevation": float(group["elevation"].mean()),
                "dominant_models": ", ".join(group["model_id"].value_counts().head(5).index.astype(str)),
                "dominant_datasets": ", ".join(group["dataset_id"].value_counts().head(6).index.astype(str)),
                "mean_curve_largest_drop": largest_drop,
                "archetype": archetype_name(group, largest_drop),
            }
        )
    return pd.DataFrame(rows).sort_values("n_runs", ascending=False).reset_index(drop=True)


def archetype_name(group: pd.DataFrame, largest_drop: float) -> str:
    cliff = float(group[TARGET].mean())
    survival = float(group[SURVIVAL].mean())
    dominant = str(group["model_id"].value_counts().idxmax())
    if cliff > 0.85 or largest_drop > 0.75:
        return "Single-Jump Allocator"
    if cliff < 0.30 and survival > 0.70:
        return "Smooth Persistent Allocator"
    if dominant in {"xgboost", "lightgbm"}:
        return "Multi-Step Allocator"
    return "Layered Transition Allocator"


def decide(scores: pd.DataFrame) -> tuple[str, str]:
    lookup = dict(zip(scores["target"], scores["score"]))
    regime = lookup.get("regime_id", np.nan)
    regime_base = float(scores[scores["target"] == "regime_id"]["baseline"].iloc[0]) if (scores["target"] == "regime_id").any() else np.nan
    model = lookup.get("model_id", np.nan)
    if lookup.get(SURVIVAL, -np.inf) > 0.90 and lookup.get(TARGET, -np.inf) > 0.80 and lookup.get("breadth", -np.inf) > 0.60 and lookup.get("elevation", -np.inf) > 0.60 and np.isfinite(regime) and regime > max(0.50, regime_base + 0.10) and (not np.isfinite(model) or model < 0.85):
        return "observed_supported", "Observed reachability topology reconstructs accessibility metrics and regimes without merely memorizing model family."
    return "observed_weaker", "Observed reachability topology is weaker than the surrogate hypothesis test; preserve surrogate result as hypothesis-generating."


def compare_to_surrogate(scores: pd.DataFrame) -> pd.DataFrame:
    mapping = {SURVIVAL: "survival_r2", TARGET: "cliffiness_r2", "breadth": "breadth_r2", "elevation": "elevation_r2", "regime_id": "regime_accuracy", "model_id": "model_family_accuracy"}
    rows = []
    for target, key in mapping.items():
        match = scores[scores["target"] == target]
        if not match.empty:
            observed = float(match["score"].iloc[0])
            rows.append({"target": target, "observed_score": observed, "surrogate_score": SURROGATE[key], "delta_observed_minus_surrogate": observed - SURROGATE[key]})
    return pd.DataFrame(rows)


def plot_embedding(meta: pd.DataFrame, emb: np.ndarray, color_col: str, output: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    if color_col in meta.columns and pd.api.types.is_numeric_dtype(meta[color_col]):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=meta[color_col], cmap="viridis", s=12, alpha=0.75)
        fig.colorbar(sc, ax=ax, label=color_col)
    else:
        codes, _ = pd.factorize(meta[color_col].astype(str) if color_col in meta.columns else pd.Series(["missing"] * len(meta)))
        ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap="tab20", s=12, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_archetypes(profiles: pd.DataFrame, matrix: pd.DataFrame, labels: np.ndarray, thresholds: np.ndarray, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in profiles["topology_cluster"].head(10):
        idx = np.where(labels == cluster)[0]
        if idx.size:
            ax.plot(thresholds, matrix.iloc[idx].mean(axis=0), label=f"cluster {cluster}")
    ax.set_xlabel("threshold")
    ax.set_ylabel("observed R(t)")
    ax.set_title("Observed Reachability Archetypes")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_report(curve_paths: list[Path], output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = load_curve_files(curve_paths)
    meta, matrix, thresholds = pivot_curves(curves)
    _, emb, labels = embedding_and_clusters(matrix)
    scores = reconstruction_scores(meta, matrix)
    profiles = cluster_profiles(meta, matrix, labels, thresholds)
    comparison = compare_to_surrogate(scores)
    outcome, outcome_text = decide(scores)
    source_counts = meta.groupby("dataset_id", as_index=False).agg(n_runs=("run_id", "size")).sort_values("n_runs", ascending=False)
    summary = {
        "n_runs": int(len(meta)),
        "n_curve_rows": int(len(curves)),
        "topology_clusters": int(pd.Series(labels[labels >= 0]).nunique()),
        "survival_r2": float(scores[scores["target"] == SURVIVAL]["score"].iloc[0]),
        "cliffiness_r2": float(scores[scores["target"] == TARGET]["score"].iloc[0]),
        "breadth_r2": float(scores[scores["target"] == "breadth"]["score"].iloc[0]),
        "elevation_r2": float(scores[scores["target"] == "elevation"]["score"].iloc[0]),
        "regime_accuracy": float(scores[scores["target"] == "regime_id"]["score"].iloc[0]) if (scores["target"] == "regime_id").any() else np.nan,
        "model_family_accuracy": float(scores[scores["target"] == "model_id"]["score"].iloc[0]),
        "top_dataset_run_counts": {str(row.dataset_id): int(row.n_runs) for row in source_counts.head(10).itertuples(index=False)},
        "outcome": outcome,
        "outcome_text": outcome_text,
    }
    summary_path = output_dir / "definitive_reachability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = output_dir / "definitive_reachability_topology.md"
    report.write_text("\n".join([
        "# Definitive Reachability Topology",
        "",
        "## Executive Answer",
        outcome_text,
        "",
        "## Observed-Curve Reconstruction",
        _markdown_table(scores),
        "",
        "## Input Run Counts",
        _markdown_table(source_counts.head(20)),
        "",
        "## Comparison To Surrogate Test",
        _markdown_table(comparison),
        "",
        "## Observed Reachability Archetypes",
        _markdown_table(profiles),
        "",
        "## Manuscript Recommendation",
        "Upgrade the manuscript claim only if observed scores remain close to the surrogate benchmark. Otherwise, keep the surrogate analysis framed as hypothesis-generating.",
    ]) + "\n", encoding="utf-8")
    return (
        report,
        plot_embedding(meta, emb, TARGET, output_dir / "definitive_reachability_umap.png", "Observed Reachability UMAP"),
        plot_archetypes(profiles, matrix, labels, thresholds, output_dir / "definitive_reachability_archetypes.png"),
        plot_embedding(meta, emb, TARGET, output_dir / "definitive_reachability_vs_cliffiness.png", "Observed Reachability vs Cliffiness"),
        plot_embedding(meta, emb, SURVIVAL, output_dir / "definitive_reachability_vs_survival.png", "Observed Reachability vs Survival"),
        summary_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hddt-curves", type=Path, default=HDDT_CURVES)
    parser.add_argument("--classical-curves", type=Path, default=CLASSICAL_CURVES)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report([args.hddt_curves, args.classical_curves], args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
