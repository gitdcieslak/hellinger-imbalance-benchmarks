"""Test whether accessibility is governed by full reachability shape."""

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
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, pairwise_distances, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_accessibility_morphology_decomposition import CAPACITY_FEATURES, load_decomposition_dataset, score_for  # noqa: E402
from report_accessibility_morphology_sufficiency import CLASSICAL_CURVES, HDDT_CURVES  # noqa: E402
from report_definitive_reachability_topology import load_curve_files, pivot_curves  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402


OUT = ROOT / "reports" / "topology"
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"
TARGETS = [TARGET, SURVIVAL, "elevation"]


def load_shape_dataset(curve_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    meta, capacity, _, full = load_decomposition_dataset(curve_paths)
    curves = load_curve_files(curve_paths)
    curve_meta, matrix, thresholds = pivot_curves(curves)
    if not np.array_equal(meta["run_id"].to_numpy(), curve_meta["run_id"].to_numpy()):
        order = curve_meta.set_index("run_id").loc[meta["run_id"]].index
        matrix = matrix.loc[curve_meta.set_index("run_id").loc[order].reset_index().index].reset_index(drop=True)
    capacity = capacity[[c for c in CAPACITY_FEATURES if c in capacity.columns]].copy()
    return meta, capacity, matrix.reset_index(drop=True), full, thresholds


def grouped_score(meta: pd.DataFrame, features: pd.DataFrame, target: str) -> tuple[float, float]:
    data = pd.concat([meta[[target, "dataset_id"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    groups = data["dataset_id"].astype(str)
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    X = data.drop(columns=[target, "dataset_id"])
    y = data[target]
    best_r2, best_mae = -np.inf, np.nan
    for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=61, n_jobs=1)]:
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        score = float(r2_score(y, pred))
        if score > best_r2:
            best_r2 = score
            best_mae = float(mean_absolute_error(y, pred))
    return best_r2, best_mae


def model_scores(meta: pd.DataFrame, capacity: pd.DataFrame, matrix: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    shape = matrix.copy()
    shape.columns = [f"R_{i:03d}" for i in range(shape.shape[1])]
    specs = {
        "capacity": capacity,
        "trajectory_shape": shape,
        "capacity_plus_trajectory": pd.concat([capacity.add_prefix("capacity__"), shape.add_prefix("shape__")], axis=1),
        "full_morphology": full,
    }
    rows = []
    for target in TARGETS:
        for name, frame in specs.items():
            r2, mae = grouped_score(meta, frame, target)
            rows.append({"target": target, "model_spec": name, "grouped_cv_r2": r2, "grouped_cv_mae": mae, "n_features": frame.shape[1]})
    scores = pd.DataFrame(rows)
    for target in TARGETS:
        cap = score_for(scores, target, "capacity")
        full_r2 = score_for(scores, target, "full_morphology")
        scores.loc[scores["target"] == target, "delta_vs_capacity"] = scores.loc[scores["target"] == target, "grouped_cv_r2"] - cap
        scores.loc[scores["target"] == target, "gap_to_full_morphology"] = full_r2 - scores.loc[scores["target"] == target, "grouped_cv_r2"]
    return scores


def variance_partition(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        cap = max(0.0, score_for(scores, target, "capacity"))
        shape = max(0.0, score_for(scores, target, "trajectory_shape"))
        both = max(0.0, score_for(scores, target, "capacity_plus_trajectory"))
        rows.extend([
            {"target": target, "component": "capacity_unique", "variance_fraction": max(0.0, both - shape)},
            {"target": target, "component": "shape_unique", "variance_fraction": max(0.0, both - cap)},
            {"target": target, "component": "shared", "variance_fraction": max(0.0, min(cap, shape))},
            {"target": target, "component": "unexplained", "variance_fraction": max(0.0, 1.0 - both)},
        ])
    return pd.DataFrame(rows)


def pca_components(matrix: pd.DataFrame, thresholds: np.ndarray, n_components: int = 10) -> tuple[pd.DataFrame, np.ndarray, PCA]:
    X = matrix.to_numpy(dtype=float)
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]), random_state=63).fit(X)
    rows = []
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        rows.append({"component": i + 1, "explained_variance_ratio": float(ratio), "cumulative_variance_ratio": float(np.sum(pca.explained_variance_ratio_[: i + 1]))})
    return pd.DataFrame(rows), pca.transform(X), pca


def shape_embedding(matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = matrix.to_numpy(dtype=float)
    pca2 = PCA(n_components=2, random_state=65).fit_transform(X)
    try:
        import umap
        um = umap.UMAP(n_neighbors=25, min_dist=0.08, random_state=65).fit_transform(X)
    except Exception:
        um = pca2
    return pca2, um


def archetypes(meta: pd.DataFrame, matrix: pd.DataFrame, n_clusters: int = 6) -> tuple[pd.DataFrame, np.ndarray]:
    X = StandardScaler().fit_transform(matrix.to_numpy(dtype=float))
    labels = AgglomerativeClustering(n_clusters=min(n_clusters, len(meta))).fit_predict(X)
    data = meta.copy(); data["shape_archetype"] = labels
    rows = []
    for label, group in data.groupby("shape_archetype"):
        idx = group.index.to_numpy()
        mean_curve = matrix.iloc[idx].mean(axis=0).to_numpy(dtype=float)
        drops = np.maximum(0, mean_curve[:-1] - mean_curve[1:])
        eff = 1.0 / max(1e-12, np.sum((drops / max(1e-12, drops.sum())) ** 2)) if drops.size else 0.0
        if drops.max() / max(1e-12, drops.sum()) > 0.75:
            name = "Single Collapse"
        elif eff < 4:
            name = "Few Collapse"
        elif mean_curve[-1] > 0.25:
            name = "Persistent Support"
        elif drops.size and drops[-max(1, drops.size // 4):].sum() / max(1e-12, drops.sum()) > 0.5:
            name = "Late Collapse"
        else:
            name = "Continuous Decay"
        rows.append({"shape_archetype": int(label), "archetype_name": name, "n_runs": int(len(group)), "mean_cliffiness": float(group[TARGET].mean()), "mean_survival_auc": float(group[SURVIVAL].mean()), "mean_elevation": float(group["elevation"].mean()), "dominant_family": str(group["model_id"].value_counts().idxmax())})
    return pd.DataFrame(rows).sort_values("n_runs", ascending=False), labels


def distance_analysis(meta: pd.DataFrame, matrix: pd.DataFrame, max_pairs: int = 100_000) -> pd.DataFrame:
    X = matrix.to_numpy(dtype=float)
    n = len(meta)
    rng = np.random.default_rng(67)
    pairs = rng.integers(0, n, size=(min(max_pairs, n * 20), 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    rows = []
    for metric in ["euclidean", "cosine"]:
        shape_d = np.asarray([pairwise_distances(X[[i]], X[[j]], metric=metric)[0, 0] for i, j in pairs])
        cliff_d = np.abs(meta[TARGET].to_numpy()[pairs[:, 0]] - meta[TARGET].to_numpy()[pairs[:, 1]])
        rows.append({"distance_metric": metric, "n_pairs": int(len(pairs)), "pearson_corr_shape_vs_cliffiness_distance": float(np.corrcoef(shape_d, cliff_d)[0, 1])})
    return pd.DataFrame(rows)


def frontier(meta: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([meta.reset_index(drop=True), capacity[["effective_support"]].reset_index(drop=True)], axis=1).dropna(subset=["effective_support", TARGET])
    data["capacity_bin"] = pd.qcut(data["effective_support"].rank(method="first"), 10, labels=False) + 1
    return data.groupby("capacity_bin", as_index=False).agg(mean_effective_support=("effective_support", "mean"), p10_cliffiness=(TARGET, lambda s: float(s.quantile(.1))), median_cliffiness=(TARGET, "median"), p90_cliffiness=(TARGET, lambda s: float(s.quantile(.9))), max_cliffiness=(TARGET, "max"))


def plot_embedding(meta: pd.DataFrame, emb: np.ndarray, output: Path, title: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, [TARGET, SURVIVAL, "model_id"], strict=True):
        vals = meta[col]
        if pd.api.types.is_numeric_dtype(vals):
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=vals, cmap="viridis", s=12, alpha=.7); fig.colorbar(sc, ax=ax, label=col)
        else:
            codes, _ = pd.factorize(vals.astype(str)); ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap="tab20", s=12, alpha=.7)
        ax.set_title(col)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_archetypes(matrix: pd.DataFrame, labels: np.ndarray, thresholds: np.ndarray, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in sorted(set(labels)):
        idx = np.where(labels == label)[0]
        ax.plot(thresholds, matrix.iloc[idx].mean(axis=0), label=f"cluster {label}")
    ax.set_xlabel("threshold"); ax.set_ylabel("R(t)"); ax.set_title("Reachability Shape Archetypes"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_modes(pca: PCA, thresholds: np.ndarray, output: Path) -> Path:
    mean = pca.mean_
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, i in zip(axes.ravel(), range(min(6, len(pca.components_)))):
        scale = np.sqrt(pca.explained_variance_[i])
        ax.plot(thresholds, mean, color="black", label="mean")
        ax.plot(thresholds, np.clip(mean + scale * pca.components_[i], 0, 1), label="+")
        ax.plot(thresholds, np.clip(mean - scale * pca.components_[i], 0, 1), label="-")
        ax.set_title(f"mode {i + 1}")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_distance(dist: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5)); ax.bar(dist["distance_metric"], dist["pearson_corr_shape_vs_cliffiness_distance"], color="steelblue")
    ax.set_ylabel("corr(shape distance, cliffiness distance)"); ax.set_title("Shape Distance vs Accessibility Distance")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_frontier(front: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5)); ax.fill_between(front["mean_effective_support"], front["p10_cliffiness"], front["p90_cliffiness"], alpha=.25); ax.plot(front["mean_effective_support"], front["median_cliffiness"], marker="o"); ax.plot(front["mean_effective_support"], front["max_cliffiness"], linestyle="--")
    ax.set_xlabel("effective support"); ax.set_ylabel("cliffiness"); ax.set_title("Trajectory Frontier Envelope"); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def write_report(curve_paths: list[Path], output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, capacity, matrix, full, thresholds = load_shape_dataset(curve_paths)
    scores = model_scores(meta, capacity, matrix, full)
    partition = variance_partition(scores)
    components, pca_scores, pca = pca_components(matrix, thresholds)
    pca2, um = shape_embedding(matrix)
    arche, labels = archetypes(meta, matrix)
    dist = distance_analysis(meta, matrix)
    front = frontier(meta, capacity)
    cap_shape = score_for(scores, TARGET, "capacity_plus_trajectory")
    full_r2 = score_for(scores, TARGET, "full_morphology")
    cap = score_for(scores, TARGET, "capacity")
    shape = score_for(scores, TARGET, "trajectory_shape")
    outcome = "Outcome A" if full_r2 - cap_shape <= 0.03 and shape > cap else "Outcome B" if shape > cap else "Outcome C"
    summary = {"n_runs": int(len(meta)), "trajectory_shape_r2": shape, "capacity_r2": cap, "capacity_plus_trajectory_r2": cap_shape, "full_morphology_r2": full_r2, "gap_to_full_morphology": full_r2 - cap_shape, "trajectory_delta_vs_capacity": shape - cap, "outcome": outcome, "pca_components_for_95pct_variance": int((components["cumulative_variance_ratio"] < .95).sum() + 1)}
    report = output_dir / "reachability_shape_hypothesis.md"; summary_path = output_dir / "reachability_shape_summary.json"; scores_path = output_dir / "reachability_shape_model_scores.csv"; part_path = output_dir / "reachability_shape_variance_partition.csv"; arche_path = output_dir / "reachability_shape_archetypes.csv"; comp_path = output_dir / "reachability_shape_components.csv"; dist_path = output_dir / "reachability_shape_distance_analysis.csv"
    scores.to_csv(scores_path, index=False); partition.to_csv(part_path, index=False); arche.to_csv(arche_path, index=False); components.to_csv(comp_path, index=False); dist.to_csv(dist_path, index=False); summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text("\n".join(["# Reachability Shape Hypothesis", "", "## Executive Answer", f"{outcome}: capacity+trajectory gap to full morphology is `{full_r2 - cap_shape:.4f}`; trajectory-only delta vs capacity is `{shape - cap:.4f}`.", "", "## Functional Prediction", _markdown_table(scores), "", "## Variance Partition", _markdown_table(partition), "", "## Functional PCA", _markdown_table(components.head(10)), "", "## Shape Archetypes", _markdown_table(arche), "", "## Shape Distance Analysis", _markdown_table(dist)]) + "\n", encoding="utf-8")
    return (report, plot_embedding(meta, um, output_dir / "reachability_umap.png", "Reachability UMAP"), plot_embedding(meta, pca2, output_dir / "reachability_pca.png", "Reachability PCA"), plot_archetypes(matrix, labels, thresholds, output_dir / "reachability_archetypes.png"), plot_modes(pca, thresholds, output_dir / "reachability_functional_modes.png"), plot_distance(dist, output_dir / "trajectory_distance_vs_accessibility_distance.png"), plot_frontier(front, output_dir / "trajectory_frontier_envelope.png"), summary_path, scores_path, part_path, arche_path, comp_path, dist_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", action="append", type=Path, dest="curves", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.curves or [HDDT_CURVES, CLASSICAL_CURVES], args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
