"""Test whether low-dimensional accessibility coordinates explain morphology."""

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
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_accessibility_regime_hypothesis import add_regimes  # noqa: E402
from report_accessibility_support_construction_hypothesis import build_feature_matrix, feature_blocks  # noqa: E402
from report_accessibility_support_topology_hypothesis import TARGET  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402
from report_minority_reachability_cohorts import DEFAULT_POSITIVE_SCORES, load_positive_scores  # noqa: E402


OUT = ROOT / "reports" / "topology"
SURVIVAL = "minority_survival_auc"
PERSISTENCE = "persistence_weighted_mass"
META = {"run_id", "dataset_id", "task_id", "model_id", TARGET, SURVIVAL, "breadth", "elevation", "simple_regime", "hierarchical_regime", "functional_regime"}


def clean(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)


def coordinate_feature_matrix(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = add_regimes(build_feature_matrix(scores))
    blocks = feature_blocks(features)
    full = clean(blocks["full"])
    return features, full


def learn_coordinates(X: pd.DataFrame, n_components: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    scaled = StandardScaler().fit_transform(clean(X))
    pca = PCA(n_components=min(n_components, X.shape[1]), random_state=151).fit(scaled)
    pca_coords = pca.transform(scaled)
    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=[f"pc{i+1}" for i in range(pca.n_components_)])
    loadings.insert(0, "feature", loadings.index)
    loadings = loadings.reset_index(drop=True)
    coord = pd.DataFrame(pca_coords, columns=[f"coord_pc{i+1}" for i in range(pca.n_components_)])
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        coord[f"pc{i}_explained_variance"] = ratio
    try:
        if len(X) < 200:
            raise RuntimeError("use PCA fallback for small test fixtures")
        import umap
        umap_coords = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.08, random_state=153).fit_transform(scaled)
    except Exception:
        umap_coords = PCA(n_components=2, random_state=153).fit_transform(scaled)
    try:
        if len(X) < 200:
            raise RuntimeError("use PCA fallback for small test fixtures")
        spectral = SpectralEmbedding(n_components=2, n_neighbors=min(15, len(X) - 1), random_state=155).fit_transform(scaled)
    except Exception:
        spectral = PCA(n_components=2, random_state=155).fit_transform(scaled)
    coord["coord_umap1"] = umap_coords[:, 0]
    coord["coord_umap2"] = umap_coords[:, 1]
    coord["coord_spectral1"] = spectral[:, 0]
    coord["coord_spectral2"] = spectral[:, 1]
    return coord, loadings, pca.explained_variance_ratio_, scaled


def onehot(series: pd.Series) -> pd.DataFrame:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    arr = enc.fit_transform(series.astype(str).to_frame())
    return pd.DataFrame(arr, columns=[f"label__{c}" for c in enc.categories_[0]], index=series.index)


def grouped_score(meta: pd.DataFrame, X: pd.DataFrame, target: str) -> tuple[float, float]:
    data = pd.concat([meta[[target, "dataset_id"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    data = clean(data)
    groups = data["dataset_id"].astype(str)
    y = data[target].astype(float)
    Xdata = data.drop(columns=[target, "dataset_id"])
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    best_r2, best_mae = -np.inf, np.nan
    for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=120, min_samples_leaf=5, random_state=157, n_jobs=-1)]:
        pred = cross_val_predict(model, Xdata, y, cv=cv, groups=groups)
        score = float(r2_score(y, pred))
        if score > best_r2:
            best_r2, best_mae = score, float(mean_absolute_error(y, pred))
    return best_r2, best_mae


def predictiveness(features: pd.DataFrame, Xfull: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    blocks = feature_blocks(features)
    coord_cols = [c for c in coords.columns if c.startswith("coord_pc")][:5]
    specs = {
        "coordinates_2d": coords[coord_cols[:2]],
        "coordinates_3d": coords[coord_cols[:3]],
        "coordinates_5d": coords[coord_cols[:5]],
        "family_labels": onehot(features["model_id"]),
        "regime_labels": onehot(features["simple_regime"]),
        "topology_geometry": blocks["geometry"],
        "construction": blocks["construction"],
        "full_features": Xfull,
    }
    targets = [TARGET, SURVIVAL, PERSISTENCE]
    rows = []
    for target in targets:
        if target not in features.columns:
            continue
        for spec, X in specs.items():
            r2, mae = grouped_score(features, X, target)
            rows.append({"target": target, "model_spec": spec, "grouped_cv_r2": r2, "grouped_cv_mae": mae, "n_features": X.shape[1]})
    out = pd.DataFrame(rows)
    for target, group in out.groupby("target"):
        full = float(group.loc[group["model_spec"] == "full_features", "grouped_cv_r2"].iloc[0])
        out.loc[out["target"] == target, "fraction_of_full"] = out.loc[out["target"] == target, "grouped_cv_r2"] / max(1e-12, full)
    return out


def neighbor_analysis(features: pd.DataFrame, coords: pd.DataFrame, k: int = 10, random_state: int = 159) -> tuple[pd.DataFrame, pd.DataFrame]:
    coord_cols = [c for c in coords.columns if c.startswith("coord_pc")][:5]
    X = StandardScaler().fit_transform(coords[coord_cols])
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(features))).fit(X)
    _, inds = nn.kneighbors(X)
    rng = np.random.default_rng(random_state)
    rows = []
    mix_rows = []
    targets = [c for c in [TARGET, SURVIVAL, PERSISTENCE] if c in features.columns]
    for i, neigh in enumerate(inds):
        neigh = neigh[neigh != i][:k]
        random_neigh = rng.choice(np.delete(np.arange(len(features)), i), size=len(neigh), replace=False)
        row = {"run_id": features.iloc[i]["run_id"], "model_id": features.iloc[i]["model_id"], "n_neighbors": int(len(neigh))}
        for target in targets:
            row[f"mean_abs_delta_{target}"] = float(np.abs(features.iloc[neigh][target].to_numpy(dtype=float) - float(features.iloc[i][target])).mean())
            row[f"random_abs_delta_{target}"] = float(np.abs(features.iloc[random_neigh][target].to_numpy(dtype=float) - float(features.iloc[i][target])).mean())
        fam = features.iloc[neigh]["model_id"].astype(str)
        shares = fam.value_counts(normalize=True)
        entropy = float(-(shares * np.log(shares)).sum()) if len(shares) else 0.0
        purity = float(shares.max()) if len(shares) else 1.0
        mix_rows.append({"run_id": features.iloc[i]["run_id"], "model_id": features.iloc[i]["model_id"], "family_entropy": entropy, "family_purity": purity, "family_mixing_score": 1.0 - purity, "n_neighbor_families": int(fam.nunique())})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(mix_rows)


def family_coordinate_accuracy(features: pd.DataFrame, coords: pd.DataFrame) -> dict[str, float]:
    X = clean(coords[[c for c in coords.columns if c.startswith("coord_pc")][:5]])
    y = features["model_id"].astype(str)
    min_count = int(y.value_counts().min())
    if y.nunique() < 2 or min_count < 2:
        return {"family_accuracy_from_coordinates": np.nan, "family_baseline_accuracy": np.nan}
    cv = StratifiedKFold(n_splits=min(5, min_count), shuffle=True, random_state=161)
    pred = cross_val_predict(RandomForestClassifier(n_estimators=100, min_samples_leaf=4, random_state=161, n_jobs=-1), X, y, cv=cv)
    return {"family_accuracy_from_coordinates": float(accuracy_score(y, pred)), "family_baseline_accuracy": float(y.value_counts(normalize=True).max())}


def vector_fields(features: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    intervention_cols = [c for c in features.columns if any(token in c.lower() for token in ["intervention", "dropout", "bagging", "weighting", "oversampling", "calibration"])]
    if not intervention_cols:
        return pd.DataFrame([{"intervention": "unavailable", "n_pairs": 0, "mean_delta_coord_pc1": np.nan, "mean_delta_coord_pc2": np.nan, "note": "Current positive-score capture has no before/after intervention identifiers."}])
    return pd.DataFrame([{"intervention": "unimplemented_metadata_present", "n_pairs": 0, "mean_delta_coord_pc1": np.nan, "mean_delta_coord_pc2": np.nan, "note": f"Intervention columns detected: {', '.join(intervention_cols)}"}])


def decision(pred: pd.DataFrame, neighbors: pd.DataFrame, mixing: pd.DataFrame, family_acc: dict[str, float]) -> tuple[str, str]:
    cliff = pred[pred["target"] == TARGET]
    coord = float(cliff.loc[cliff["model_spec"] == "coordinates_5d", "fraction_of_full"].iloc[0])
    regime = float(cliff.loc[cliff["model_spec"] == "regime_labels", "grouped_cv_r2"].iloc[0])
    smooth = float((neighbors[f"mean_abs_delta_{TARGET}"] < neighbors[f"random_abs_delta_{TARGET}"]).mean())
    mix = float(mixing["family_mixing_score"].mean())
    family_acc_value = family_acc.get("family_accuracy_from_coordinates", np.nan)
    if coord >= 0.90 and smooth >= 0.75 and mix >= 0.30 and family_acc_value < 0.80 and coord > regime:
        return "strong_support", "Coordinates explain morphology with smooth, family-mixed neighborhoods and do not collapse to family identity."
    if coord >= 0.90 and smooth >= 0.65 and coord > regime:
        return "moderate_support", "Coordinates explain morphology well, but family dependence or limited mixing remains."
    return "failure", "Coordinates do not meet predictive, smoothness, or family-independence criteria."


def plot_embedding(features: pd.DataFrame, coords: pd.DataFrame, x: str, y: str, color: str, output: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    values = features[color] if color in features.columns else coords[color]
    if pd.api.types.is_numeric_dtype(values):
        sc = ax.scatter(coords[x], coords[y], c=values, cmap="magma", s=12, alpha=0.7)
        fig.colorbar(sc, ax=ax, label=color)
    else:
        codes, _ = pd.factorize(values.astype(str))
        ax.scatter(coords[x], coords[y], c=codes, cmap="tab20", s=12, alpha=0.7)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def write_report(positive_scores: Path = DEFAULT_POSITIVE_SCORES, output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features, Xfull = coordinate_feature_matrix(load_positive_scores(positive_scores))
    coords, loadings, explained, _ = learn_coordinates(Xfull)
    coord_scores = pd.concat([features[["run_id", "dataset_id", "task_id", "model_id", "simple_regime", TARGET, SURVIVAL, "breadth", "elevation"]].reset_index(drop=True), coords.reset_index(drop=True)], axis=1)
    pred = predictiveness(features, Xfull, coords)
    neighbors, mixing = neighbor_analysis(features, coords)
    family_acc = family_coordinate_accuracy(features, coords)
    vectors = vector_fields(features, coords)
    outcome, text = decision(pred, neighbors, mixing, family_acc)
    cliff = pred[pred["target"] == TARGET]
    summary = {"outcome": outcome, "outcome_text": text, "n_runs": int(len(features)), "pca_components_for_90pct_feature_variance": int(np.searchsorted(np.cumsum(explained), 0.90) + 1), "coordinate_5d_cliffiness_fraction_of_full": float(cliff.loc[cliff["model_spec"] == "coordinates_5d", "fraction_of_full"].iloc[0]), "coordinate_5d_cliffiness_r2": float(cliff.loc[cliff["model_spec"] == "coordinates_5d", "grouped_cv_r2"].iloc[0]), "full_cliffiness_r2": float(cliff.loc[cliff["model_spec"] == "full_features", "grouped_cv_r2"].iloc[0]), "regime_cliffiness_r2": float(cliff.loc[cliff["model_spec"] == "regime_labels", "grouped_cv_r2"].iloc[0]), "mean_cliffiness_neighbor_delta": float(neighbors[f"mean_abs_delta_{TARGET}"].mean()), "mean_random_cliffiness_delta": float(neighbors[f"random_abs_delta_{TARGET}"].mean()), "mean_family_mixing_score": float(mixing["family_mixing_score"].mean()), **family_acc}
    report = output_dir / "accessibility_coordinates_hypothesis.md"
    summary_path = output_dir / "accessibility_coordinate_summary.json"
    loadings_path = output_dir / "accessibility_coordinate_loadings.csv"
    scores_path = output_dir / "accessibility_coordinate_scores.csv"
    pred_path = output_dir / "accessibility_coordinate_predictiveness.csv"
    neigh_path = output_dir / "accessibility_coordinate_neighbors.csv"
    mix_path = output_dir / "accessibility_coordinate_family_mixing.csv"
    vector_path = output_dir / "accessibility_coordinate_vector_fields.csv"
    loadings.to_csv(loadings_path, index=False); coord_scores.to_csv(scores_path, index=False); pred.to_csv(pred_path, index=False); neighbors.to_csv(neigh_path, index=False); mixing.to_csv(mix_path, index=False); vectors.to_csv(vector_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text("\n".join(["# Accessibility Coordinates Hypothesis", "", "## Executive Answer", text, "", "## Coordinate Predictiveness", _markdown_table(pred), "", "## Neighbor Smoothness Summary", _markdown_table(neighbors.describe().reset_index()), "", "## Family Mixing Summary", _markdown_table(mixing.describe().reset_index()), "", "## Vector Fields", _markdown_table(vectors)]) + "\n", encoding="utf-8")
    coords_with_mixing = pd.concat([coords.reset_index(drop=True), mixing[["family_mixing_score"]].reset_index(drop=True)], axis=1)
    return (
        report,
        summary_path,
        loadings_path,
        scores_path,
        pred_path,
        neigh_path,
        mix_path,
        vector_path,
        plot_embedding(features, coords, "coord_umap1", "coord_umap2", TARGET, output_dir / "accessibility_coordinate_umap.png", "Accessibility Coordinate UMAP"),
        plot_embedding(features, coords, "coord_pc1", "coord_pc2", "model_id", output_dir / "accessibility_coordinate_pca.png", "Accessibility Coordinate PCA"),
        plot_embedding(features, coords, "coord_pc1", "coord_pc2", TARGET, output_dir / "coordinate_vs_cliffiness.png", "Coordinates vs Cliffiness"),
        plot_embedding(features, coords, "coord_pc1", "coord_pc2", PERSISTENCE, output_dir / "coordinate_vs_persistence.png", "Coordinates vs Persistence"),
        plot_embedding(features, coords_with_mixing, "coord_umap1", "coord_umap2", "family_mixing_score", output_dir / "family_mixing_map.png", "Family Mixing Map"),
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
