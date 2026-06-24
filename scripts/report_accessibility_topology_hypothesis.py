"""Test whether reachability-trajectory topology is the latent accessibility object.

The relaxed HDDT validation CSV does not store full threshold trajectories. This
report therefore builds a surrogate reachability matrix from saved survival-shape
summaries and treats results as a hypothesis stress test, not final proof.
"""

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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402


HDDT_CSV = ROOT / "results" / "real_validation" / "hddt_accessibility_validation_relaxed.csv"
OUT = ROOT / "reports" / "topology"
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"
MAX_DROP = "minority_survival_max_drop"
EFFECTIVE_DROPS = "minority_survival_effective_drop_count"
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


def load_data(path: Path = HDDT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "fit_failed" in df.columns:
        df = df[~df["fit_failed"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    required = ["dataset_name", "model_id", "regime_id", "breadth", "elevation", SURVIVAL, TARGET, MAX_DROP, EFFECTIVE_DROPS]
    return df.dropna(subset=required).reset_index(drop=True)


def morphology_region(df: pd.DataFrame) -> pd.Series:
    labels = []
    for b, e in zip(df["breadth"], df["elevation"]):
        if e >= 0.80 and b < 0.70:
            labels.append("high_elevation_low_breadth")
        elif e >= 0.80:
            labels.append("high_elevation_moderate_breadth")
        elif 0.40 <= e < 0.80:
            labels.append("mid_elevation")
        elif b >= 0.70:
            labels.append("low_elevation_high_breadth")
        else:
            labels.append("low_elevation_low_breadth")
    return pd.Series(labels, index=df.index)


def surrogate_reachability_curve(survival_auc: float, cliffiness: float, effective_drop_count: float, n_grid: int = 101) -> np.ndarray:
    """Construct a monotone surrogate R(t) from saved shape summaries."""

    grid = np.linspace(0.0, 1.0, n_grid)
    auc = float(np.clip(survival_auc, 0.0, 1.0))
    cliff = float(np.clip(cliffiness, 0.0, 1.0))
    k = int(np.clip(round(effective_drop_count), 1, 20))
    if cliff <= 0.0:
        return np.full(n_grid, auc)
    first_drop = max(cliff, 1.0 / max(k, 1))
    remaining = max(0.0, 1.0 - first_drop)
    drops = [first_drop]
    if k > 1:
        drops.extend([remaining / (k - 1)] * (k - 1))
    center = auc
    spread = min(0.45, 0.02 + 0.015 * k)
    positions = np.linspace(max(0.01, center - spread), min(0.99, center + spread), k)
    curve = np.ones(n_grid)
    for pos, drop in zip(positions, drops, strict=False):
        curve -= drop * (grid >= pos)
    curve = np.maximum.accumulate(curve[::-1])[::-1]
    curve = np.clip(curve, 0.0, 1.0)
    # Shift the step locations slightly if needed so area matches the stored AUC.
    for _ in range(6):
        err = auc - float(np.trapezoid(curve, grid))
        if abs(err) < 0.005:
            break
        positions = np.clip(positions + err * 0.4, 0.01, 0.99)
        curve = np.ones(n_grid)
        for pos, drop in zip(positions, drops, strict=False):
            curve -= drop * (grid >= pos)
        curve = np.maximum.accumulate(curve[::-1])[::-1]
        curve = np.clip(curve, 0.0, 1.0)
    return curve


def build_reachability_matrix(df: pd.DataFrame, n_grid: int = 101) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, n_grid)
    matrix = np.vstack([surrogate_reachability_curve(row[SURVIVAL], row[TARGET], row[EFFECTIVE_DROPS], n_grid=n_grid) for _, row in df.iterrows()])
    return matrix, grid


def grouped_regression(df: pd.DataFrame, X: pd.DataFrame | np.ndarray, target: str) -> tuple[float, bool, str]:
    Xdf = pd.DataFrame(X).reset_index(drop=True)
    Xdf.columns = [str(c) for c in Xdf.columns]
    data = pd.concat([df[[target, "dataset_name"]].reset_index(drop=True), Xdf], axis=1).dropna()
    if len(data) < 20:
        return np.nan, False, "too_few_rows"
    groups = data["dataset_name"].astype(str)
    if groups.nunique() < 2:
        return np.nan, False, "too_few_groups"
    try:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
        Xdata = data.drop(columns=[target, "dataset_name"])
        models = [
            make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)),
            RandomForestRegressor(n_estimators=250, min_samples_leaf=4, random_state=0, n_jobs=-1),
        ]
        scores = []
        for model in models:
            pred = cross_val_predict(model, Xdata, data[target], cv=cv, groups=groups)
            scores.append(float(r2_score(data[target], pred)))
        return float(max(scores)), True, ""
    except Exception as exc:
        return np.nan, False, type(exc).__name__


def grouped_classifier(df: pd.DataFrame, X: pd.DataFrame | np.ndarray, target: str) -> tuple[float, float, bool, str]:
    Xdf = pd.DataFrame(X).reset_index(drop=True)
    Xdf.columns = [str(c) for c in Xdf.columns]
    data = pd.concat([df[[target, "dataset_name"]].reset_index(drop=True), Xdf], axis=1).dropna()
    y = data[target].astype(str)
    groups = data["dataset_name"].astype(str)
    if y.nunique() < 2 or groups.nunique() < 2:
        return np.nan, np.nan, False, "insufficient_classes_or_groups"
    try:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, random_state=0, n_jobs=-1)
        pred = cross_val_predict(clf, data.drop(columns=[target, "dataset_name"]), y, cv=cv, groups=groups)
        return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro")), True, ""
    except Exception as exc:
        return np.nan, np.nan, False, type(exc).__name__


def topology_embedding(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=0).fit_transform(matrix)
    try:
        import umap

        emb = umap.UMAP(n_neighbors=25, min_dist=0.05, random_state=0).fit_transform(matrix)
    except Exception:
        emb = pca
    try:
        import hdbscan

        labels = hdbscan.HDBSCAN(min_cluster_size=45, min_samples=10).fit_predict(StandardScaler().fit_transform(matrix))
    except Exception:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=0.8, min_samples=10).fit_predict(StandardScaler().fit_transform(matrix))
    return pca, emb, labels


def metric_reconstruction(df: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    rows = []
    for target in [TARGET, SURVIVAL, "breadth", "elevation"]:
        score, valid, reason = grouped_regression(df, matrix, target)
        rows.append({"target": target, "feature_set": "reachability_surrogate", "grouped_cv_score": score, "metric": "r2", "valid": valid, "skip_reason": reason})
    for target in ["regime_id", "model_id"]:
        acc, f1, valid, reason = grouped_classifier(df, matrix, target)
        rows.append({"target": target, "feature_set": "reachability_surrogate", "grouped_cv_score": acc, "macro_f1": f1, "metric": "accuracy", "valid": valid, "skip_reason": reason})
    return pd.DataFrame(rows)


def variance_decomposition(df: pd.DataFrame, matrix: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    occ = [c for c in OCCUPANCY_COLS if c in df.columns]
    cluster = pd.get_dummies(pd.Series(labels).astype(str), prefix="topology_cluster")
    specs = {
        "occupancy_only": df[occ].reset_index(drop=True),
        "reachability_only": pd.DataFrame(matrix),
        "occupancy_plus_reachability": pd.concat([df[occ].reset_index(drop=True), pd.DataFrame(matrix)], axis=1),
        "occupancy_plus_topology_cluster": pd.concat([df[occ].reset_index(drop=True), cluster.reset_index(drop=True)], axis=1),
    }
    rows = []
    for target in [TARGET, SURVIVAL, "breadth", "elevation"]:
        for spec, X in specs.items():
            score, valid, reason = grouped_regression(df, X, target)
            rows.append({"target": target, "model_spec": spec, "grouped_cv_r2": score, "valid": valid, "skip_reason": reason})
    return pd.DataFrame(rows)


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray, matrix: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    data = df.copy()
    data["topology_cluster"] = labels
    data["morphology_region"] = morphology_region(data)
    valid = data[data["topology_cluster"] >= 0].copy()
    rows = []
    for cluster, group in valid.groupby("topology_cluster"):
        idx = group.index.to_numpy()
        mean_curve = matrix[idx].mean(axis=0)
        largest_drop = float(np.max(np.maximum(0, mean_curve[:-1] - mean_curve[1:])))
        archetype = archetype_name(group, largest_drop)
        rows.append({
            "topology_cluster": int(cluster),
            "n_runs": int(len(group)),
            "mean_cliffiness": float(group[TARGET].mean()),
            "mean_survival_auc": float(group[SURVIVAL].mean()),
            "mean_breadth": float(group["breadth"].mean()),
            "mean_elevation": float(group["elevation"].mean()),
            "dominant_models": ", ".join(group["model_id"].value_counts().head(5).index.astype(str)),
            "dominant_datasets": ", ".join(group["dataset_name"].value_counts().head(6).index.astype(str)),
            "dominant_region": str(group["morphology_region"].value_counts().idxmax()),
            "mean_curve_largest_drop": largest_drop,
            "archetype": archetype,
        })
    return pd.DataFrame(rows).sort_values("n_runs", ascending=False).reset_index(drop=True)


def archetype_name(group: pd.DataFrame, largest_drop: float) -> str:
    cliff = float(group[TARGET].mean())
    survival = float(group[SURVIVAL].mean())
    eff = float(group[EFFECTIVE_DROPS].mean())
    dominant_model = str(group["model_id"].value_counts().idxmax())
    if cliff > 0.85 and eff <= 2:
        return "Single-Jump Allocator"
    if eff > 10 and cliff < 0.35:
        return "Smooth Allocator"
    if survival > 0.85 and cliff < 0.45:
        return "Broad Persistent Allocator"
    if dominant_model in {"lightgbm", "xgboost"} and eff > 2:
        return "Multi-Step Allocator"
    return "Layered Transition Allocator"


def decide_outcome(scores: pd.DataFrame) -> tuple[str, str]:
    lookup = {row["target"]: row["grouped_cv_score"] for _, row in scores.iterrows()}
    if lookup.get(TARGET, 0) > 0.90 and lookup.get(SURVIVAL, 0) > 0.90 and lookup.get("breadth", 0) > 0.80 and lookup.get("elevation", 0) > 0.80:
        return "A", "Reachability topology reconstructs survival, cliffiness, and morphology; accessibility metrics are projections of one topology object."
    if lookup.get(TARGET, 0) > 0.80 and lookup.get(SURVIVAL, 0) > 0.80:
        return "B", "Topology explains survival and cliffiness, but morphology is not fully reconstructed."
    return "C", "Topology adds little beyond occupancy under this surrogate test."


def plot_embedding(df: pd.DataFrame, emb: np.ndarray, color_col: str, output: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    if pd.api.types.is_numeric_dtype(df[color_col]):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=df[color_col], cmap="viridis", s=12, alpha=0.75)
        fig.colorbar(sc, ax=ax, label=color_col)
    else:
        codes, _ = pd.factorize(df[color_col].astype(str))
        ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap="tab20", s=12, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_clusters(df: pd.DataFrame, emb: np.ndarray, labels: np.ndarray, output: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, values, title in [(axes[0], labels, "Topology Clusters"), (axes[1], df["model_id"], "Model Family"), (axes[2], df["regime_id"], "Regime")]:
        codes, _ = pd.factorize(pd.Series(values).astype(str))
        ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap="tab20", s=10, alpha=0.7)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_morphology(df: pd.DataFrame, emb: np.ndarray, output: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col in zip(axes, ["breadth", "elevation"], strict=True):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=df[col], cmap="viridis", s=10, alpha=0.7)
        ax.set_title(col)
        fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_archetypes(profiles: pd.DataFrame, matrix: np.ndarray, labels: np.ndarray, grid: np.ndarray, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in profiles["topology_cluster"].head(8):
        idx = np.where(labels == cluster)[0]
        if idx.size:
            ax.plot(grid, matrix[idx].mean(axis=0), label=f"cluster {cluster}")
    ax.set_xlabel("threshold")
    ax.set_ylabel("R(t)")
    ax.set_title("Mean Reachability Archetypes")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_cluster_profiles(profiles: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    top = profiles.head(12)
    axes[0].bar(top["topology_cluster"].astype(str), top["mean_cliffiness"])
    axes[0].set_title("Mean Cliffiness")
    axes[1].bar(top["topology_cluster"].astype(str), top["mean_survival_auc"])
    axes[1].set_title("Mean Survival")
    axes[2].bar(top["topology_cluster"].astype(str), top["n_runs"])
    axes[2].set_title("Cluster Size")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_report(output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data()
    matrix, grid = build_reachability_matrix(df)
    pca, emb, labels = topology_embedding(matrix)
    df = df.copy()
    df["morphology_region"] = morphology_region(df)
    scores = metric_reconstruction(df, matrix)
    decomp = variance_decomposition(df, matrix, labels)
    profiles = cluster_profiles(df, labels, matrix, grid)
    outcome, outcome_text = decide_outcome(scores)
    summary = {
        "n_runs": int(len(df)),
        "topology_clusters": int(pd.Series(labels[labels >= 0]).nunique()),
        "cliffiness_r2": float(scores[scores["target"] == TARGET]["grouped_cv_score"].iloc[0]),
        "survival_r2": float(scores[scores["target"] == SURVIVAL]["grouped_cv_score"].iloc[0]),
        "breadth_r2": float(scores[scores["target"] == "breadth"]["grouped_cv_score"].iloc[0]),
        "elevation_r2": float(scores[scores["target"] == "elevation"]["grouped_cv_score"].iloc[0]),
        "regime_accuracy": float(scores[scores["target"] == "regime_id"]["grouped_cv_score"].iloc[0]),
        "model_family_accuracy": float(scores[scores["target"] == "model_id"]["grouped_cv_score"].iloc[0]),
        "best_archetype": str(profiles.sort_values("n_runs", ascending=False)["archetype"].iloc[0]) if not profiles.empty else "none",
        "outcome": outcome,
        "outcome_text": outcome_text,
        "caveat": "Full reachability trajectories were not saved; this analysis uses surrogate R(t) curves reconstructed from survival AUC, cliffiness, and effective drop-count summaries.",
    }
    summary_path = output_dir / "accessibility_topology_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = output_dir / "accessibility_topology_hypothesis.md"
    report.write_text("\n".join([
        "# Accessibility Topology Hypothesis",
        "",
        "## Executive Answer",
        executive_answer(summary),
        "",
        "## Data Caveat",
        summary["caveat"],
        "",
        "## Q1-Q3: Reconstruction from Reachability Topology",
        _markdown_table(scores),
        "",
        "## Variance Decomposition",
        _markdown_table(decomp),
        "",
        "## Topology Archetypes",
        _markdown_table(profiles),
        "",
        "## Decision",
        f"Outcome {outcome}: {outcome_text}",
    ]) + "\n", encoding="utf-8")
    outputs = [
        report,
        plot_embedding(df, emb, TARGET, output_dir / "accessibility_topology_umap.png", "Accessibility Topology UMAP"),
        plot_clusters(df, emb, labels, output_dir / "accessibility_topology_clusters.png"),
        plot_embedding(df, emb, TARGET, output_dir / "accessibility_topology_vs_cliffiness.png", "Topology vs Cliffiness"),
        plot_embedding(df, emb, SURVIVAL, output_dir / "accessibility_topology_vs_survival.png", "Topology vs Survival"),
        plot_morphology(df, emb, output_dir / "accessibility_topology_vs_morphology.png"),
        plot_archetypes(profiles, matrix, labels, grid, output_dir / "reachability_archetypes.png"),
        plot_cluster_profiles(profiles, output_dir / "topology_cluster_profiles.png"),
        summary_path,
    ]
    return tuple(outputs)


def executive_answer(summary: dict[str, object]) -> str:
    return (
        f"Surrogate reachability topology reconstructs cliffiness R2={summary['cliffiness_r2']:.4f}, "
        f"survival R2={summary['survival_r2']:.4f}, breadth R2={summary['breadth_r2']:.4f}, "
        f"and elevation R2={summary['elevation_r2']:.4f}. "
        f"Regime accuracy is {summary['regime_accuracy']:.4f}; model-family accuracy is {summary['model_family_accuracy']:.4f}. "
        f"Outcome {summary['outcome']}: {summary['outcome_text']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
