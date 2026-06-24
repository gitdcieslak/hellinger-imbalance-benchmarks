"""Test whether families are occupancy distributions in a shared accessibility manifold."""

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
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_accessibility_coordinates_hypothesis import coordinate_feature_matrix, learn_coordinates  # noqa: E402
from report_accessibility_support_topology_hypothesis import TARGET  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402
from report_minority_reachability_cohorts import DEFAULT_POSITIVE_SCORES, load_positive_scores  # noqa: E402


OUT = ROOT / "reports" / "topology"
SURVIVAL = "minority_survival_auc"
PERSISTENCE = "persistence_weighted_mass"


def clean(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)


def onehot(series: pd.Series) -> pd.DataFrame:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    arr = enc.fit_transform(series.astype(str).to_frame())
    return pd.DataFrame(arr, columns=[f"family__{c}" for c in enc.categories_[0]], index=series.index)


def grouped_score(meta: pd.DataFrame, X: pd.DataFrame, target: str = TARGET) -> tuple[float, float]:
    data = pd.concat([meta[[target, "dataset_id"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    data = clean(data)
    groups = data["dataset_id"].astype(str)
    y = data[target].astype(float)
    Xdata = data.drop(columns=[target, "dataset_id"])
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    best_r2, best_mae = -np.inf, np.nan
    for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=140, min_samples_leaf=5, random_state=171, n_jobs=-1)]:
        pred = cross_val_predict(model, Xdata, y, cv=cv, groups=groups)
        score = float(r2_score(y, pred))
        if score > best_r2:
            best_r2, best_mae = score, float(mean_absolute_error(y, pred))
    return best_r2, best_mae


def build_coordinates(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, X = coordinate_feature_matrix(scores)
    coords, _, _, _ = learn_coordinates(X, n_components=5)
    coord_cols = [c for c in coords.columns if c.startswith("coord_pc")][:5]
    data = pd.concat([features.reset_index(drop=True), coords[coord_cols + ["coord_umap1", "coord_umap2"]].reset_index(drop=True)], axis=1)
    return data, data[coord_cols]


def neighborhood_consistency(data: pd.DataFrame, coords: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    X = StandardScaler().fit_transform(coords)
    inds = NearestNeighbors(n_neighbors=min(k + 1, len(data))).fit(X).kneighbors(X, return_distance=False)
    rows = []
    targets = [c for c in [TARGET, SURVIVAL, PERSISTENCE] if c in data.columns]
    rng = np.random.default_rng(173)
    for i, neigh in enumerate(inds):
        neigh = neigh[neigh != i][:k]
        random_neigh = rng.choice(np.delete(np.arange(len(data)), i), size=len(neigh), replace=False)
        families = data.iloc[neigh]["model_id"].astype(str)
        shares = families.value_counts(normalize=True)
        row = {"run_id": data.iloc[i]["run_id"], "model_id": data.iloc[i]["model_id"], "family_purity": float(shares.max()), "family_mixing": float(1.0 - shares.max()), "n_neighbor_families": int(families.nunique())}
        for target in targets:
            own = float(data.iloc[i][target])
            row[f"local_delta_{target}"] = float(np.abs(data.iloc[neigh][target].to_numpy(dtype=float) - own).mean())
            row[f"random_delta_{target}"] = float(np.abs(data.iloc[random_neigh][target].to_numpy(dtype=float) - own).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def family_overlap(data: pd.DataFrame, coords: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    X = StandardScaler().fit_transform(coords)
    inds = NearestNeighbors(n_neighbors=min(k + 1, len(data))).fit(X).kneighbors(X, return_distance=False)
    rows = []
    for family, group in data.groupby("model_id"):
        idx = group.index.to_numpy()
        neighbor_families = []
        for i in idx:
            neigh = inds[i]
            neigh = neigh[neigh != i][:k]
            neighbor_families.extend(data.iloc[neigh]["model_id"].astype(str).tolist())
        counts = pd.Series(neighbor_families).value_counts(normalize=True) if neighbor_families else pd.Series(dtype=float)
        rows.append({"model_id": family, "self_neighbor_fraction": float(counts.get(family, 0.0)), "other_family_fraction": float(1.0 - counts.get(family, 0.0)), "n_overlapping_families": int((counts > 0).sum())})
    return pd.DataFrame(rows)


def distance_correlation(data: pd.DataFrame, coords: pd.DataFrame, max_pairs: int = 250000) -> pd.DataFrame:
    X = StandardScaler().fit_transform(coords)
    y = data[[TARGET, SURVIVAL, PERSISTENCE]].copy()
    y = clean(y[[c for c in y.columns if c in data.columns]])
    n = len(data)
    if n * (n - 1) // 2 <= max_pairs:
        coord_dist = pdist(X)
        morph_dist = pdist(StandardScaler().fit_transform(y))
    else:
        rng = np.random.default_rng(175)
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
        coord_dist = np.linalg.norm(X[i] - X[j], axis=1)
        morph_scaled = StandardScaler().fit_transform(y)
        morph_dist = np.linalg.norm(morph_scaled[i] - morph_scaled[j], axis=1)
    return pd.DataFrame([{"pearson_r": float(pearsonr(coord_dist, morph_dist).statistic), "spearman_r": float(spearmanr(coord_dist, morph_dist).statistic), "n_pairs": int(len(coord_dist))}])


def conditional_prediction(data: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    family = onehot(data["model_id"])
    specs = {
        "coordinates": coords,
        "family": family,
        "coordinates_plus_family": pd.concat([coords.add_prefix("coord__"), family], axis=1),
    }
    rows = []
    for spec, X in specs.items():
        r2, mae = grouped_score(data, X, TARGET)
        rows.append({"model_spec": spec, "grouped_cv_r2": r2, "grouped_cv_mae": mae, "n_features": X.shape[1]})
    out = pd.DataFrame(rows)
    coord = float(out.loc[out["model_spec"] == "coordinates", "grouped_cv_r2"].iloc[0])
    out["delta_vs_coordinates"] = out["grouped_cv_r2"] - coord
    return out


def local_family_effect(data: pd.DataFrame, neigh: pd.DataFrame) -> pd.DataFrame:
    merged = data[["run_id", TARGET]].merge(neigh, on="run_id")
    rows = []
    for bucket, group in merged.groupby(pd.qcut(merged["family_mixing"].rank(method="first"), 5, labels=False)):
        rows.append({"mixing_quintile": int(bucket) + 1, "mean_family_mixing": float(group["family_mixing"].mean()), "mean_local_cliffiness_delta": float(group[f"local_delta_{TARGET}"].mean()), "mean_random_cliffiness_delta": float(group[f"random_delta_{TARGET}"].mean()), "n_runs": int(len(group))})
    return pd.DataFrame(rows)


def decision(conditional: pd.DataFrame, distance: pd.DataFrame, neigh: pd.DataFrame, overlap: pd.DataFrame) -> tuple[str, str]:
    coord = float(conditional.loc[conditional["model_spec"] == "coordinates", "grouped_cv_r2"].iloc[0])
    family_gain = float(conditional.loc[conditional["model_spec"] == "coordinates_plus_family", "delta_vs_coordinates"].iloc[0])
    smooth_ratio = float(neigh[f"local_delta_{TARGET}"].mean() / max(1e-12, neigh[f"random_delta_{TARGET}"].mean()))
    dist_r = float(distance["spearman_r"].iloc[0])
    mixing = float(overlap["other_family_fraction"].mean())
    if coord >= 0.85 and family_gain <= 0.05 and smooth_ratio <= 0.35 and dist_r >= 0.65 and mixing >= 0.30:
        return "strong_support", "Families behave mostly as occupancy distributions in a shared coordinate manifold."
    if coord >= 0.75 and smooth_ratio <= 0.50 and dist_r >= 0.45:
        return "moderate_support", "Coordinates organize morphology locally, but family occupancy remains non-negligible."
    return "failure", "Family identity remains necessary or coordinate distance does not sufficiently govern morphology."


def plot_occupancy(data: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for family, group in data.groupby("model_id"):
        ax.scatter(group["coord_umap1"], group["coord_umap2"], s=12, alpha=0.55, label=family)
    ax.set_title("Family Occupancy in Accessibility Coordinates")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_distance(distance_points: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(distance_points["coordinate_distance"], distance_points["morphology_distance"], s=4, alpha=0.15)
    ax.set_xlabel("coordinate distance"); ax.set_ylabel("morphology distance")
    ax.set_title("Coordinate Distance vs Morphology Distance")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def sampled_distance_points(data: pd.DataFrame, coords: pd.DataFrame, n_pairs: int = 30000) -> pd.DataFrame:
    rng = np.random.default_rng(177)
    X = StandardScaler().fit_transform(coords)
    y = StandardScaler().fit_transform(clean(data[[TARGET, SURVIVAL, PERSISTENCE]]))
    n = len(data)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    return pd.DataFrame({"coordinate_distance": np.linalg.norm(X[i] - X[j], axis=1), "morphology_distance": np.linalg.norm(y[i] - y[j], axis=1)})


def write_report(positive_scores: Path = DEFAULT_POSITIVE_SCORES, output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data, coords = build_coordinates(load_positive_scores(positive_scores))
    neigh = neighborhood_consistency(data, coords)
    overlap = family_overlap(data, coords)
    dist = distance_correlation(data, coords)
    cond = conditional_prediction(data, coords)
    local = local_family_effect(data, neigh)
    outcome, text = decision(cond, dist, neigh, overlap)
    family_gain = float(cond.loc[cond["model_spec"] == "coordinates_plus_family", "delta_vs_coordinates"].iloc[0])
    summary = {"outcome": outcome, "outcome_text": text, "coordinate_cliffiness_r2": float(cond.loc[cond["model_spec"] == "coordinates", "grouped_cv_r2"].iloc[0]), "family_gain_after_coordinates": family_gain, "distance_spearman_r": float(dist["spearman_r"].iloc[0]), "mean_local_cliffiness_delta": float(neigh[f"local_delta_{TARGET}"].mean()), "mean_random_cliffiness_delta": float(neigh[f"random_delta_{TARGET}"].mean()), "mean_other_family_overlap": float(overlap["other_family_fraction"].mean())}
    report = output_dir / "accessibility_occupancy_hypothesis.md"
    summary_path = output_dir / "accessibility_occupancy_summary.json"
    neigh_path = output_dir / "accessibility_occupancy_neighbors.csv"
    overlap_path = output_dir / "accessibility_occupancy_family_overlap.csv"
    dist_path = output_dir / "accessibility_occupancy_distance_correlation.csv"
    cond_path = output_dir / "accessibility_occupancy_conditional_prediction.csv"
    local_path = output_dir / "accessibility_occupancy_local_family_mixing.csv"
    scores_path = output_dir / "accessibility_occupancy_coordinate_scores.csv"
    data.to_csv(scores_path, index=False); neigh.to_csv(neigh_path, index=False); overlap.to_csv(overlap_path, index=False); dist.to_csv(dist_path, index=False); cond.to_csv(cond_path, index=False); local.to_csv(local_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text("\n".join(["# Accessibility Occupancy Hypothesis", "", "## Executive Answer", text, "", "## Conditional Prediction", _markdown_table(cond), "", "## Family Overlap", _markdown_table(overlap), "", "## Distance Correlation", _markdown_table(dist), "", "## Local Family Mixing", _markdown_table(local)]) + "\n", encoding="utf-8")
    distance_points = sampled_distance_points(data, coords)
    return (
        report,
        summary_path,
        neigh_path,
        overlap_path,
        dist_path,
        cond_path,
        local_path,
        scores_path,
        plot_occupancy(data, output_dir / "accessibility_occupancy_family_map.png"),
        plot_distance(distance_points, output_dir / "accessibility_occupancy_distance_vs_morphology.png"),
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
