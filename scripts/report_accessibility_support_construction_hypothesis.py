"""Test whether support construction mechanisms explain accessibility morphology."""

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
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, normalized_mutual_info_score, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_accessibility_quantization_hypothesis import quantization_features  # noqa: E402
from report_accessibility_support_granularity_hypothesis import GRANULARITY_COLS, granularity_features  # noqa: E402
from report_accessibility_support_topology_hypothesis import CAPACITY_COLS, TARGET, support_topology_features  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402
from report_minority_reachability_cohorts import DEFAULT_POSITIVE_SCORES, load_positive_scores  # noqa: E402
from report_support_topology_family_holdout import _fit_predict  # noqa: E402


OUT = ROOT / "reports" / "topology"
TOPOLOGY_LOFO = OUT / "support_topology_family_holdout_scores.csv"

META_COLS = {"run_id", "dataset_id", "task_id", "model_id", TARGET, "minority_survival_auc", "breadth", "elevation", "n_positives"}
QUANTIZATION_COLS = [
    "n_unique_positive_posteriors",
    "effective_posterior_alphabet",
    "posterior_hhi",
    "largest_posterior_mass",
    "posterior_entropy",
    "gini_posterior_mass",
    "mean_posterior_gap",
    "median_posterior_gap",
    "max_posterior_gap",
    "mass_at_largest_gap",
    "mass_above_largest_gap",
    "posterior_range",
]
TOPOLOGY_COLS = [
    "n_support_islands",
    "largest_island_mass",
    "island_entropy",
    "island_gini",
    "small_island_fraction",
    "high_threshold_survivor_concentration",
    "persistence_weighted_mass",
    "fragile_support_mass",
    "stable_support_mass",
    "stable_to_fragile_ratio",
]


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()


def build_feature_matrix(scores: pd.DataFrame) -> pd.DataFrame:
    topology = support_topology_features(scores)
    quant = quantization_features(scores)
    gran = granularity_features(scores)
    base = topology.copy()
    add_quant = quant.drop(columns=[c for c in quant.columns if c in base.columns and c != "run_id"], errors="ignore")
    add_gran = gran.drop(columns=[c for c in gran.columns if c in base.columns and c != "run_id"], errors="ignore")
    features = base.merge(add_quant, on="run_id", how="left").merge(add_gran, on="run_id", how="left")
    features = _dedupe_columns(features)
    features["broad_allocator_index"] = features[["support_redundancy", "mass_outside_top3", "stable_support_mass"]].mean(axis=1)
    features["concentrated_allocator_index"] = features[["top1_group_mass", "group_mass_hhi", "largest_posterior_mass"]].mean(axis=1)
    features["quantized_allocator_index"] = features[["posterior_hhi", "largest_posterior_mass", "max_posterior_gap"]].mean(axis=1)
    features["continuous_allocator_index"] = features[["n_unique_positive_posteriors", "effective_posterior_alphabet", "posterior_entropy"]].rank(pct=True).mean(axis=1)
    features["fragmented_allocator_index"] = features[["n_support_islands", "small_island_fraction", "island_entropy"]].rank(pct=True).mean(axis=1)
    features["redundant_allocator_index"] = features[["support_redundancy", "mass_outside_top5", "effective_support_groups"]].rank(pct=True).mean(axis=1)
    features["hierarchical_allocator_index"] = features[["top1_group_mass", "largest_to_median_ratio", "group_mass_gini"]].rank(pct=True).mean(axis=1)
    return features.replace([np.inf, -np.inf], np.nan)


def clean(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)


def feature_blocks(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    capacity = features[CAPACITY_COLS].copy()
    geometry_cols = [c for c in [*TOPOLOGY_COLS, *QUANTIZATION_COLS, *GRANULARITY_COLS] if c in features.columns]
    geometry = _dedupe_columns(features[geometry_cols])
    construction_cols = [
        "minority_survival_auc",
        "n_positives",
        *CAPACITY_COLS,
        "broad_allocator_index",
        "concentrated_allocator_index",
        "quantized_allocator_index",
        "continuous_allocator_index",
        "fragmented_allocator_index",
        "redundant_allocator_index",
        "hierarchical_allocator_index",
        "support_redundancy",
        "mass_outside_top3",
        "mass_outside_top5",
        "persistence_weighted_mass",
        "posterior_range",
        "max_posterior_gap",
        "mass_at_largest_gap",
        "mass_above_largest_gap",
    ]
    construction = _dedupe_columns(features[[c for c in construction_cols if c in features.columns]])
    full_cols = [c for c in features.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(features[c])]
    full = _dedupe_columns(features[full_cols])
    return {"capacity": capacity, "geometry": geometry, "construction": construction, "full": full}


def grouped_score(meta: pd.DataFrame, X: pd.DataFrame, target: str = TARGET) -> tuple[float, float]:
    data = pd.concat([meta[[target, "dataset_id"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    data = clean(data)
    groups = data["dataset_id"].astype(str)
    y = data[target].astype(float)
    Xdata = data.drop(columns=[target, "dataset_id"])
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    best_r2, best_mae = -np.inf, np.nan
    for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=220, min_samples_leaf=5, random_state=121, n_jobs=-1)]:
        pred = cross_val_predict(model, Xdata, y, cv=cv, groups=groups)
        score = float(r2_score(y, pred))
        if score > best_r2:
            best_r2, best_mae = score, float(mean_absolute_error(y, pred))
    return best_r2, best_mae


def predictive_scores(features: pd.DataFrame) -> pd.DataFrame:
    blocks = feature_blocks(features)
    rows = []
    for spec, block in blocks.items():
        r2, mae = grouped_score(features, block)
        rows.append({"model_spec": spec, "grouped_cv_r2": r2, "grouped_cv_mae": mae, "n_features": block.shape[1]})
    out = pd.DataFrame(rows)
    full_r2 = float(out.loc[out["model_spec"] == "full", "grouped_cv_r2"].iloc[0])
    cap_r2 = float(out.loc[out["model_spec"] == "capacity", "grouped_cv_r2"].iloc[0])
    out["delta_vs_capacity"] = out["grouped_cv_r2"] - cap_r2
    out["gap_to_full"] = full_r2 - out["grouped_cv_r2"]
    out["gap_closed_vs_capacity"] = np.where(full_r2 > cap_r2, (out["grouped_cv_r2"] - cap_r2) / (full_r2 - cap_r2), np.nan)
    return out


def lofo_scores(features: pd.DataFrame, topology_baseline: Path = TOPOLOGY_LOFO) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = feature_blocks(features)
    X = clean(blocks["construction"]).reset_index(drop=True)
    y = features[TARGET].astype(float).reset_index(drop=True)
    families = sorted(features["model_id"].astype(str).unique())
    baseline = pd.read_csv(topology_baseline) if topology_baseline.exists() else pd.DataFrame()
    rows = []
    pred_rows = []
    for family in families:
        test_mask = features["model_id"].astype(str).reset_index(drop=True) == family
        train_mask = ~test_mask
        pred = _fit_predict(X.loc[train_mask], y.loc[train_mask], X.loc[test_mask], features.loc[train_mask, "dataset_id"])
        actual = y.loc[test_mask].to_numpy(dtype=float)
        r2 = float(r2_score(actual, pred)) if np.var(actual) > 0 else np.nan
        topo = np.nan
        if not baseline.empty:
            match = baseline[baseline["held_out_family"].astype(str) == family]
            if not match.empty:
                topo = float(match["support_topology_r2"].iloc[0])
        rows.append({"held_out_family": family, "construction_r2": r2, "construction_mae": float(mean_absolute_error(actual, pred)), "topology_lofo_r2": topo, "improvement_vs_topology": r2 - topo if np.isfinite(topo) else np.nan, "n_test": int(test_mask.sum())})
        for run_id, a, p in zip(features.loc[test_mask, "run_id"], actual, pred, strict=False):
            pred_rows.append({"held_out_family": family, "run_id": run_id, "actual_cliffiness": float(a), "predicted_cliffiness": float(p)})
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def discover_regimes(features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = clean(feature_blocks(features)["construction"])
    scaled = StandardScaler().fit_transform(X)
    try:
        import umap

        emb = umap.UMAP(n_neighbors=30, min_dist=0.08, random_state=123).fit_transform(scaled)
    except Exception:
        emb = PCA(n_components=2, random_state=123).fit_transform(scaled)
    try:
        import hdbscan

        labels = hdbscan.HDBSCAN(min_cluster_size=max(20, len(features) // 35), min_samples=8).fit_predict(scaled)
    except Exception:
        labels = DBSCAN(eps=0.9, min_samples=8).fit_predict(scaled)
        if len(set(labels)) <= 1:
            labels = KMeans(n_clusters=min(6, max(2, len(features) // 100)), random_state=123, n_init=20).fit_predict(scaled)
    out = features[["run_id", "model_id", TARGET]].copy()
    out["construction_cluster"] = labels
    out["construction_umap_x"] = emb[:, 0]
    out["construction_umap_y"] = emb[:, 1]
    return out, emb


def cluster_summary(regimes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, group in regimes.groupby("construction_cluster"):
        fam = group["model_id"].astype(str).value_counts(normalize=True)
        rows.append({"construction_cluster": int(cluster), "n_runs": int(len(group)), "mean_cliffiness": float(group[TARGET].mean()), "dominant_family": fam.index[0], "dominant_family_fraction": float(fam.iloc[0])})
    return pd.DataFrame(rows).sort_values("construction_cluster").reset_index(drop=True)


def cluster_alignment(regimes: pd.DataFrame) -> dict[str, float]:
    labels = regimes["construction_cluster"].to_numpy()
    fam_codes, _ = pd.factorize(regimes["model_id"].astype(str))
    return {"family_cluster_ari": float(adjusted_rand_score(fam_codes, labels)), "family_cluster_nmi": float(normalized_mutual_info_score(fam_codes, labels)), "n_clusters": int(len(set(labels) - {-1})), "noise_fraction": float(np.mean(labels == -1))}


def mediation_scores(features: pd.DataFrame) -> pd.DataFrame:
    blocks = feature_blocks(features)
    construction = clean(blocks["construction"])
    geometry = clean(blocks["geometry"])
    capacity = clean(blocks["capacity"])
    groups = features["dataset_id"].astype(str)
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=180, min_samples_leaf=5, random_state=127, n_jobs=-1))
    geom_pred = cross_val_predict(model, construction, geometry, cv=cv, groups=groups)
    cap_pred = cross_val_predict(model, geometry, capacity, cv=cv, groups=groups)
    rows = [
        {"path": "construction_to_geometry", "r2": float(r2_score(geometry, geom_pred, multioutput="variance_weighted")), "mae": np.nan},
        {"path": "geometry_to_capacity", "r2": float(r2_score(capacity, cap_pred, multioutput="variance_weighted")), "mae": np.nan},
    ]
    for path, block in [("capacity_to_cliffiness", capacity), ("geometry_to_cliffiness", geometry), ("construction_to_cliffiness", construction)]:
        r2, mae = grouped_score(features, block)
        rows.append({"path": path, "r2": r2, "mae": mae})
    return pd.DataFrame(rows)


def variance_partition(scores: pd.DataFrame) -> pd.DataFrame:
    lookup = dict(zip(scores["model_spec"], scores["grouped_cv_r2"]))
    cap = max(0.0, lookup["capacity"])
    geom = max(0.0, lookup["geometry"])
    cons = max(0.0, lookup["construction"])
    full = max(0.0, lookup["full"])
    return pd.DataFrame([
        {"component": "construction_unique_over_geometry", "variance_fraction": max(0.0, full - geom)},
        {"component": "geometry_unique_over_construction", "variance_fraction": max(0.0, full - cons)},
        {"component": "construction_gain_over_capacity", "variance_fraction": max(0.0, cons - cap)},
        {"component": "geometry_gain_over_capacity", "variance_fraction": max(0.0, geom - cap)},
        {"component": "unexplained_full", "variance_fraction": max(0.0, 1.0 - full)},
    ])


def plot_embedding(regimes: pd.DataFrame, color_col: str, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    values = regimes[color_col]
    if pd.api.types.is_numeric_dtype(values):
        sc = ax.scatter(regimes["construction_umap_x"], regimes["construction_umap_y"], c=values, cmap="magma", s=12, alpha=0.7)
        fig.colorbar(sc, ax=ax, label=color_col)
    else:
        codes, _ = pd.factorize(values.astype(str))
        ax.scatter(regimes["construction_umap_x"], regimes["construction_umap_y"], c=codes, cmap="tab20", s=12, alpha=0.7)
    ax.set_title(f"Construction Space: {color_col}")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def decision(scores: pd.DataFrame, lofo: pd.DataFrame, mediation: pd.DataFrame) -> tuple[str, str]:
    lookup = dict(zip(scores["model_spec"], scores["grouped_cv_r2"]))
    construction_gt_geometry = lookup["construction"] > lookup["geometry"]
    full = lookup["full"]
    gap_closed = float(scores.loc[scores["model_spec"] == "construction", "gap_closed_vs_capacity"].iloc[0]) if full > lookup["capacity"] else np.nan
    lofo_better = float((lofo["improvement_vs_topology"] > 0).mean()) if not lofo.empty else 0.0
    c_to_g = float(mediation.loc[mediation["path"] == "construction_to_geometry", "r2"].iloc[0])
    if construction_gt_geometry and gap_closed > 0.75 and lofo_better > 0.5 and c_to_g > 0.80:
        return "strong_success", "Construction features outperform geometry, generalize better than topology, and explain most geometry variance."
    if (construction_gt_geometry or lofo_better > 0.5) and c_to_g > 0.80:
        return "partial_success", "Construction is closer to the mechanism but still retains family-dependent limitations."
    return "failure", "Construction adds little beyond geometry/capacity or does not improve family holdout generalization."


def write_report(positive_scores: Path = DEFAULT_POSITIVE_SCORES, output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_raw = load_positive_scores(positive_scores)
    features = build_feature_matrix(scores_raw)
    scores = predictive_scores(features)
    lofo, predictions = lofo_scores(features)
    regimes, _ = discover_regimes(features)
    clusters = cluster_summary(regimes)
    alignment = cluster_alignment(regimes)
    mediation = mediation_scores(features)
    partition = variance_partition(scores)
    outcome, text = decision(scores, lofo, mediation)
    summary = {"outcome": outcome, "outcome_text": text, **alignment, "lofo_construction_beats_topology_fraction": float((lofo["improvement_vs_topology"] > 0).mean())}
    for row in scores.itertuples(index=False):
        summary[f"{row.model_spec}_r2"] = float(row.grouped_cv_r2)
        summary[f"{row.model_spec}_gap_closed_vs_capacity"] = float(row.gap_closed_vs_capacity) if np.isfinite(row.gap_closed_vs_capacity) else None
    report = output_dir / "accessibility_support_construction_hypothesis.md"
    summary_path = output_dir / "accessibility_support_construction_summary.json"
    features_path = output_dir / "support_construction_features.csv"
    scores_path = output_dir / "support_construction_model_scores.csv"
    lofo_path = output_dir / "support_construction_family_holdout.csv"
    pred_path = output_dir / "support_construction_family_holdout_predictions.csv"
    regimes_path = output_dir / "support_construction_regimes.csv"
    clusters_path = output_dir / "support_construction_cluster_summary.csv"
    mediation_path = output_dir / "support_construction_mediation.csv"
    partition_path = output_dir / "support_construction_variance_partition.csv"
    features.to_csv(features_path, index=False); scores.to_csv(scores_path, index=False); lofo.to_csv(lofo_path, index=False); predictions.to_csv(pred_path, index=False); regimes.to_csv(regimes_path, index=False); clusters.to_csv(clusters_path, index=False); mediation.to_csv(mediation_path, index=False); partition.to_csv(partition_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text("\n".join(["# Accessibility Support Construction Hypothesis", "", "## Executive Answer", text, "", "## Predictive Scores", _markdown_table(scores), "", "## Family Holdout", _markdown_table(lofo), "", "## Construction Clusters", _markdown_table(clusters), "", "## Cluster-Family Alignment", _markdown_table(pd.DataFrame([alignment])), "", "## Mediation", _markdown_table(mediation), "", "## Variance Partition", _markdown_table(partition)]) + "\n", encoding="utf-8")
    return (
        report,
        summary_path,
        features_path,
        scores_path,
        lofo_path,
        pred_path,
        regimes_path,
        clusters_path,
        mediation_path,
        partition_path,
        plot_embedding(regimes, "model_id", output_dir / "support_construction_umap_by_family.png"),
        plot_embedding(regimes, TARGET, output_dir / "support_construction_umap_by_cliffiness.png"),
        plot_embedding(regimes, "construction_cluster", output_dir / "support_construction_umap_by_cluster.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-scores", type=Path, default=DEFAULT_POSITIVE_SCORES)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.positive_scores, args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
