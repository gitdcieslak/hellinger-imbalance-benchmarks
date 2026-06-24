"""Test whether allocation morphology is sufficient for cliffiness."""

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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_definitive_reachability_topology import CLASSICAL_CURVES, HDDT_CURVES, load_curve_files, pivot_curves  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402
from report_posterior_concentration_hypothesis import concentration_features, join_existing_metrics  # noqa: E402


OUT = ROOT / "reports" / "topology"
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"


MORPHOLOGY_BASE_COLS = [
    "breadth",
    "elevation",
    SURVIVAL,
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


def load_dataset(curve_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    curves = load_curve_files(curve_paths)
    meta, matrix, thresholds = pivot_curves(curves)
    curve_concentration = concentration_features(matrix, thresholds)
    morphology = pd.concat([meta[["breadth", "elevation", SURVIVAL]].reset_index(drop=True), curve_concentration.reset_index(drop=True)], axis=1)
    morphology = join_existing_metrics(meta, morphology)
    morphology = add_distribution_shape_features(morphology)
    topology = topology_features_from_curves(matrix, thresholds)
    return meta, morphology, topology, thresholds


def add_distribution_shape_features(morphology: pd.DataFrame) -> pd.DataFrame:
    out = morphology.copy()
    out["elevation_to_breadth_ratio"] = out["elevation"] / out["breadth"].replace(0, np.nan)
    out["survival_loss"] = 1.0 - out[SURVIVAL]
    if "effective_support" in out.columns:
        out["inverse_effective_support"] = 1.0 / out["effective_support"].replace(0, np.nan)
    if {"top_10pct_mass", "top_1pct_mass"}.issubset(out.columns):
        out["top10_to_top1_ratio"] = out["top_10pct_mass"] / out["top_1pct_mass"].replace(0, np.nan)
    if {"joined_positive_top_bin_mass", "joined_positive_histogram_entropy"}.issubset(out.columns):
        out["joined_concentration_entropy_product"] = out["joined_positive_top_bin_mass"] * out["joined_positive_histogram_entropy"]
    return out.replace([np.inf, -np.inf], np.nan)


def topology_features_from_curves(matrix: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    t = np.asarray(thresholds, dtype=float)
    for _, row in matrix.iterrows():
        curve = row.to_numpy(dtype=float)
        drops = np.maximum(0.0, curve[:-1] - curve[1:])
        active = drops[drops > 1e-12]
        weights = active / active.sum() if active.sum() > 0 else np.asarray([1.0])
        if drops.size:
            drop_thresholds = t[1:]
            max_idx = int(np.argmax(drops))
            largest_drop_threshold = float(drop_thresholds[max_idx])
        else:
            largest_drop_threshold = np.nan
        rows.append(
            {
                "giant_component_fraction": float(curve[-1]),
                "component_entropy": float(-(weights * np.log(weights + 1e-12)).sum()),
                "isolated_positive_fraction": float(drops[drops > 0.10].sum()) if drops.size else 0.0,
                "mean_component_size": float(active.mean()) if active.size else 0.0,
                "median_component_size": float(np.median(active)) if active.size else 0.0,
                "max_component_size": float(active.max()) if active.size else 0.0,
                "n_components": int(active.size),
                "effective_component_count": float(1.0 / np.sum(weights**2)),
                "largest_drop_threshold": largest_drop_threshold,
                "early_drop_mass": float(drops[drop_thresholds <= 0.25].sum()) if drops.size else 0.0,
                "mid_drop_mass": float(drops[(drop_thresholds > 0.25) & (drop_thresholds <= 0.75)].sum()) if drops.size else 0.0,
                "late_drop_mass": float(drops[drop_thresholds > 0.75].sum()) if drops.size else 0.0,
                "reachability_auc_proxy": float(np.trapezoid(curve, t)) if curve.size == t.size else np.nan,
                "reachability_persistence_50": float(curve[np.searchsorted(t, 0.50, side="left")]) if curve.size == t.size else np.nan,
                "reachability_persistence_90": float(curve[np.searchsorted(t, 0.90, side="left")]) if curve.size == t.size else np.nan,
                "curve_roughness": float(np.sum(np.abs(np.diff(drops)))) if drops.size > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def available_numeric(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
            cols.append(col)
    return cols


def aligned_data(meta: pd.DataFrame, features: pd.DataFrame, cols: list[str], target: str = TARGET, group_col: str = "dataset_id") -> pd.DataFrame:
    data = pd.concat([meta[[target, group_col]].reset_index(drop=True), features[cols].reset_index(drop=True)], axis=1)
    return data.replace([np.inf, -np.inf], np.nan).dropna()


def grouped_regression(meta: pd.DataFrame, features: pd.DataFrame, cols: list[str], group_col: str = "dataset_id") -> tuple[np.ndarray, dict[str, object]]:
    pred = np.full(len(meta), np.nan)
    data = aligned_data(meta, features, cols, group_col=group_col)
    if len(data) < 20:
        return pred, {"r2": np.nan, "mae": np.nan, "valid": False, "skip_reason": "too_few_rows"}
    groups = data[group_col].astype(str)
    if groups.nunique() < 2:
        return pred, {"r2": np.nan, "mae": np.nan, "valid": False, "skip_reason": "too_few_groups"}
    try:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
        X = data.drop(columns=[TARGET, group_col])
        y = data[TARGET]
        candidates = [
            make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)),
            RandomForestRegressor(n_estimators=350, min_samples_leaf=4, random_state=3, n_jobs=-1),
        ]
        best = None
        best_r2 = -np.inf
        best_mae = np.nan
        for model in candidates:
            candidate = cross_val_predict(model, X, y, cv=cv, groups=groups)
            score = float(r2_score(y, candidate))
            if score > best_r2:
                best_r2 = score
                best = candidate
                best_mae = float(mean_absolute_error(y, candidate))
        pred[data.index.to_numpy()] = best
        return pred, {"r2": float(best_r2), "mae": best_mae, "valid": True, "skip_reason": ""}
    except Exception as exc:
        return pred, {"r2": np.nan, "mae": np.nan, "valid": False, "skip_reason": type(exc).__name__}


def model_scores(meta: pd.DataFrame, morphology: pd.DataFrame, topology: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    morph_cols = available_numeric(morphology)
    topo_cols = available_numeric(topology)
    specs = {
        "topology_only": (topology, topo_cols),
        "morphology_only": (morphology, morph_cols),
        "topology_plus_morphology": (pd.concat([topology.add_prefix("topology__"), morphology.add_prefix("morphology__")], axis=1), [f"topology__{c}" for c in topo_cols] + [f"morphology__{c}" for c in morph_cols]),
    }
    rows = []
    predictions = pd.DataFrame({"run_id": meta["run_id"], TARGET: meta[TARGET]})
    cols_by_spec = {}
    for name, (frame, cols) in specs.items():
        pred, score = grouped_regression(meta, frame, cols)
        predictions[name] = pred
        cols_by_spec[name] = cols
        rows.append({"model_spec": name, "grouped_cv_r2": score["r2"], "grouped_cv_mae": score["mae"], "n_features": len(cols), "valid": score["valid"], "skip_reason": score["skip_reason"]})
    scores = pd.DataFrame(rows)
    base = score_for(scores, "topology_only")
    morph = score_for(scores, "morphology_only")
    full = score_for(scores, "topology_plus_morphology")
    scores["delta_vs_topology"] = scores["grouped_cv_r2"] - base
    scores["delta_vs_morphology"] = scores["grouped_cv_r2"] - morph
    scores["incremental_over_best_single"] = scores["grouped_cv_r2"] - max(base, morph)
    scores.loc[scores["model_spec"] != "topology_plus_morphology", "incremental_over_best_single"] = np.nan
    return scores, predictions, cols_by_spec


def score_for(scores: pd.DataFrame, spec: str) -> float:
    match = scores[scores["model_spec"] == spec]
    return float(match["grouped_cv_r2"].iloc[0]) if not match.empty else np.nan


def variance_partition(scores: pd.DataFrame) -> pd.DataFrame:
    top = max(0.0, score_for(scores, "topology_only"))
    morph = max(0.0, score_for(scores, "morphology_only"))
    full = max(0.0, score_for(scores, "topology_plus_morphology"))
    topology_unique = max(0.0, full - morph)
    morphology_unique = max(0.0, full - top)
    shared = max(0.0, min(top, morph, full - topology_unique - morphology_unique))
    unexplained = max(0.0, 1.0 - full)
    total = topology_unique + morphology_unique + shared + unexplained
    rows = [
        {"component": "topology_unique", "variance_fraction": topology_unique / total if total else np.nan},
        {"component": "morphology_unique", "variance_fraction": morphology_unique / total if total else np.nan},
        {"component": "shared_topology_morphology", "variance_fraction": shared / total if total else np.nan},
        {"component": "unexplained", "variance_fraction": unexplained / total if total else np.nan},
    ]
    return pd.DataFrame(rows)


def permutation_partition(meta: pd.DataFrame, morphology: pd.DataFrame, topology: pd.DataFrame, scores: pd.DataFrame, n_repeats: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    morph_cols = available_numeric(morphology)
    topo_cols = available_numeric(topology)
    full_frame = pd.concat([topology.add_prefix("topology__"), morphology.add_prefix("morphology__")], axis=1)
    full_cols = [f"topology__{c}" for c in topo_cols] + [f"morphology__{c}" for c in morph_cols]
    base_full = score_for(scores, "topology_plus_morphology")
    rows = []
    for block_name, block_cols in [("permute_topology", [f"topology__{c}" for c in topo_cols]), ("permute_morphology", [f"morphology__{c}" for c in morph_cols])]:
        perm_scores = []
        for _ in range(n_repeats):
            shuffled = full_frame.copy()
            for col in block_cols:
                shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            _, score = grouped_regression(meta, shuffled, full_cols)
            perm_scores.append(score["r2"])
        rows.append({"partition_test": block_name, "base_full_r2": base_full, "mean_permuted_r2": float(np.nanmean(perm_scores)), "mean_r2_drop": float(base_full - np.nanmean(perm_scores))})
    return pd.DataFrame(rows)


def feature_importance(meta: pd.DataFrame, morphology: pd.DataFrame) -> pd.DataFrame:
    cols = available_numeric(morphology)
    data = aligned_data(meta, morphology, cols)
    if len(data) < 50 or not cols:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    X = data.drop(columns=[TARGET, "dataset_id"])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
    model = RandomForestRegressor(n_estimators=350, min_samples_leaf=5, random_state=4, n_jobs=-1).fit(X_train, y_train)
    imp = permutation_importance(model, X_test, y_test, n_repeats=12, random_state=4, n_jobs=-1)
    return pd.DataFrame({"feature": X.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std}).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def morphology_embedding(morphology: pd.DataFrame) -> np.ndarray:
    X = morphology[available_numeric(morphology)].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    scaled = StandardScaler().fit_transform(X)
    try:
        import umap

        return umap.UMAP(n_neighbors=25, min_dist=0.08, random_state=5).fit_transform(scaled)
    except Exception:
        return PCA(n_components=2, random_state=5).fit_transform(scaled)


def counterfactual_pairs(meta: pd.DataFrame, morphology: pd.DataFrame, topology: pd.DataFrame, max_pairs: int = 100) -> pd.DataFrame:
    morph_X = prepare_scaled(morphology)
    topo_X = prepare_scaled(topology)
    if len(meta) < 3:
        return pd.DataFrame()
    morph_nn = NearestNeighbors(n_neighbors=min(10, len(meta))).fit(morph_X)
    topo_nn = NearestNeighbors(n_neighbors=min(10, len(meta))).fit(topo_X)
    topo_dist, topo_ind = topo_nn.kneighbors(topo_X)
    morph_dist, morph_ind = morph_nn.kneighbors(morph_X)
    rows = []
    for i in range(len(meta)):
        topo_candidates = [j for j in topo_ind[i, 1:] if meta.loc[i, "dataset_id"] != meta.loc[j, "dataset_id"]]
        morph_candidates = [j for j in morph_ind[i, 1:] if meta.loc[i, "dataset_id"] != meta.loc[j, "dataset_id"]]
        if topo_candidates:
            j = int(max(topo_candidates, key=lambda c: np.linalg.norm(morph_X[i] - morph_X[c])))
            rows.append(pair_row(meta, morph_X, topo_X, i, j, "similar_topology_different_morphology"))
        if morph_candidates:
            j = int(max(morph_candidates, key=lambda c: np.linalg.norm(topo_X[i] - topo_X[c])))
            rows.append(pair_row(meta, morph_X, topo_X, i, j, "similar_morphology_different_topology"))
        if len(rows) >= max_pairs:
            break
    return pd.DataFrame(rows)


def prepare_scaled(features: pd.DataFrame) -> np.ndarray:
    X = features[available_numeric(features)].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return StandardScaler().fit_transform(X)


def pair_row(meta: pd.DataFrame, morph_X: np.ndarray, topo_X: np.ndarray, i: int, j: int, pair_type: str) -> dict[str, object]:
    return {
        "pair_type": pair_type,
        "run_a": meta.loc[i, "run_id"],
        "run_b": meta.loc[j, "run_id"],
        "model_a": meta.loc[i, "model_id"],
        "model_b": meta.loc[j, "model_id"],
        "morphology_distance": float(np.linalg.norm(morph_X[i] - morph_X[j])),
        "topology_distance": float(np.linalg.norm(topo_X[i] - topo_X[j])),
        "cliffiness_a": float(meta.loc[i, TARGET]),
        "cliffiness_b": float(meta.loc[j, TARGET]),
        "cliffiness_abs_delta": float(abs(meta.loc[i, TARGET] - meta.loc[j, TARGET])),
    }


def counterfactual_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    return pairs.groupby("pair_type", as_index=False).agg(n_pairs=("pair_type", "size"), mean_cliffiness_abs_delta=("cliffiness_abs_delta", "mean"), median_cliffiness_abs_delta=("cliffiness_abs_delta", "median"), mean_morphology_distance=("morphology_distance", "mean"), mean_topology_distance=("topology_distance", "mean"))


def decide(scores: pd.DataFrame) -> tuple[str, str]:
    top = score_for(scores, "topology_only")
    morph = score_for(scores, "morphology_only")
    full = score_for(scores, "topology_plus_morphology")
    full_vs_morph = full - morph if np.isfinite(full) and np.isfinite(morph) else np.nan
    if np.isfinite(morph) and morph >= 0.65 and np.isfinite(full_vs_morph) and full_vs_morph < 0.03:
        return "strong_support", "Morphology alone explains most cliffiness variation and topology adds little beyond it."
    if np.isfinite(morph) and morph >= 0.50:
        return "moderate_support", "Morphology explains substantial cliffiness variation, but topology still contributes or overlaps materially."
    return "weak_support", "Morphology contributes little beyond topology; topology remains the fundamental explanatory layer."


def plot_importance(importance: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(15).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], color="steelblue")
    ax.set_title("Morphology Feature Importance")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_morphology_umap(meta: pd.DataFrame, emb: np.ndarray, output: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, col, title in [(axes[0, 0], TARGET, "Cliffiness"), (axes[0, 1], "model_id", "Model Family"), (axes[1, 0], SURVIVAL, "Survival AUC"), (axes[1, 1], "elevation", "Elevation")]:
        values = meta[col] if col in meta.columns else pd.Series(np.nan, index=meta.index)
        if pd.api.types.is_numeric_dtype(values):
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, cmap="viridis", s=12, alpha=0.75)
            fig.colorbar(sc, ax=ax, label=col)
        else:
            codes, _ = pd.factorize(values.astype(str))
            ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap="tab20", s=12, alpha=0.75)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_partition(partition: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(partition["component"], partition["variance_fraction"], color=["#4c78a8", "#f58518", "#54a24b", "#bab0ac"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("variance fraction")
    ax.set_title("Morphology/Topology Variance Partition")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_family_occupancy(meta: pd.DataFrame, emb: np.ndarray, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))
    for model, group in meta.assign(x=emb[:, 0], y=emb[:, 1]).groupby("model_id"):
        ax.scatter(group["x"], group["y"], s=14, alpha=0.55, label=str(model))
    ax.set_title("Family Occupancy In Morphology Space")
    ax.set_xlabel("morphology axis 1")
    ax.set_ylabel("morphology axis 2")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_counterfactuals(pairs: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    if pairs.empty:
        ax.text(0.5, 0.5, "No counterfactual pairs", ha="center")
    else:
        summary = counterfactual_summary(pairs)
        ax.bar(summary["pair_type"], summary["mean_cliffiness_abs_delta"], color=["#e45756", "#72b7b2"])
        ax.set_ylabel("mean |cliffiness delta|")
        ax.tick_params(axis="x", rotation=20)
    ax.set_title("Topology vs Morphology Counterfactuals")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_report(curve_paths: list[Path], output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, morphology, topology, _ = load_dataset(curve_paths)
    scores, predictions, _ = model_scores(meta, morphology, topology)
    partition = variance_partition(scores)
    permutation = permutation_partition(meta, morphology, topology, scores)
    importance = feature_importance(meta, morphology)
    emb = morphology_embedding(morphology)
    pairs = counterfactual_pairs(meta, morphology, topology)
    pair_summary = counterfactual_summary(pairs)
    outcome, outcome_text = decide(scores)
    summary = {
        "n_runs": int(len(meta)),
        "n_morphology_features": int(len(available_numeric(morphology))),
        "n_topology_features": int(len(available_numeric(topology))),
        "topology_r2": score_for(scores, "topology_only"),
        "morphology_r2": score_for(scores, "morphology_only"),
        "topology_plus_morphology_r2": score_for(scores, "topology_plus_morphology"),
        "topology_plus_morphology_delta_vs_topology": score_for(scores, "topology_plus_morphology") - score_for(scores, "topology_only"),
        "topology_plus_morphology_delta_vs_morphology": score_for(scores, "topology_plus_morphology") - score_for(scores, "morphology_only"),
        "outcome": outcome,
        "outcome_text": outcome_text,
    }
    scores_path = output_dir / "morphology_model_scores.csv"
    partition_path = output_dir / "morphology_variance_partition.csv"
    importance_path = output_dir / "morphology_feature_importance.csv"
    summary_path = output_dir / "accessibility_morphology_sufficiency_summary.json"
    report_path = output_dir / "accessibility_morphology_sufficiency.md"
    scores.to_csv(scores_path, index=False)
    pd.concat([partition.assign(method="overlap_partition"), permutation.rename(columns={"partition_test": "component", "mean_r2_drop": "variance_fraction"}).assign(method="permutation_drop")], ignore_index=True, sort=False).to_csv(partition_path, index=False)
    importance.to_csv(importance_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Accessibility Morphology Sufficiency",
                "",
                "## Executive Answer",
                outcome_text,
                "",
                "## Model Scores",
                _markdown_table(scores),
                "",
                "## Variance Partition",
                _markdown_table(partition),
                "",
                "## Permutation Partition",
                _markdown_table(permutation),
                "",
                "## Top Morphology Features",
                _markdown_table(importance.head(20)),
                "",
                "## Counterfactual Summary",
                _markdown_table(pair_summary),
                "",
                "## Interpretation",
                "Strong support requires morphology-only `R^2 >= 0.65` and topology+morphology improvement over morphology-only `< 0.03 R^2`, matching RQ3's test of topology's incremental value once morphology is known.",
                "The originally stated shorthand `topology+morphology over topology-only < 0.03` is reported in the score table as `delta_vs_topology`, but it is not the incremental-topology test.",
                "Observed topology is represented by empirical reachability-curve features; morphology is represented by scalar allocation, occupancy, survival, and concentration summaries.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        report_path,
        plot_importance(importance, output_dir / "morphology_feature_importance.png"),
        plot_morphology_umap(meta, emb, output_dir / "morphology_umap.png"),
        plot_partition(partition, output_dir / "morphology_variance_partition.png"),
        plot_family_occupancy(meta, emb, output_dir / "family_morphology_occupancy.png"),
        plot_counterfactuals(pairs, output_dir / "topology_vs_morphology_counterfactuals.png"),
        scores_path,
        partition_path,
        importance_path,
        summary_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", action="append", type=Path, dest="curves", default=None, help="Reachability-curve CSV. May be repeated.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    curves = args.curves or [HDDT_CURVES, CLASSICAL_CURVES]
    for output in write_report(curves, args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
