"""Test whether accessibility morphology is primarily regime organized."""

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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, normalized_mutual_info_score, r2_score, silhouette_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_accessibility_support_construction_hypothesis import build_feature_matrix, feature_blocks  # noqa: E402
from report_accessibility_support_topology_hypothesis import CAPACITY_COLS, TARGET  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402
from report_minority_reachability_cohorts import DEFAULT_POSITIVE_SCORES, load_positive_scores  # noqa: E402


OUT = ROOT / "reports" / "topology"
REGIME_MAP = {
    "cart": "quantized",
    "random_forest": "quantized",
    "logistic_regression": "continuous",
    "lightgbm": "continuous",
    "xgboost": "continuous",
    "hddt": "broad",
    "bagged_hddt": "broad",
}
CONCENTRATED_FAMILIES = {"xgboost", "lightgbm"}


def add_regimes(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["simple_regime"] = out["model_id"].astype(str).map(REGIME_MAP).fillna("other")
    out["hierarchical_regime"] = out["model_id"].astype(str)
    out["functional_regime"] = out["simple_regime"]
    out.loc[out["model_id"].astype(str).isin(CONCENTRATED_FAMILIES), "functional_regime"] = "concentrated"
    return out


def clean(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)


def onehot(series: pd.Series) -> pd.DataFrame:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    arr = enc.fit_transform(series.astype(str).to_frame())
    return pd.DataFrame(arr, columns=[f"regime__{c}" for c in enc.categories_[0]], index=series.index)


def grouped_score(meta: pd.DataFrame, X: pd.DataFrame) -> tuple[float, float]:
    data = pd.concat([meta[[TARGET, "dataset_id"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    data = clean(data)
    groups = data["dataset_id"].astype(str)
    y = data[TARGET].astype(float)
    Xdata = data.drop(columns=[TARGET, "dataset_id"])
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    best_r2, best_mae = -np.inf, np.nan
    for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=220, min_samples_leaf=5, random_state=141, n_jobs=-1)]:
        pred = cross_val_predict(model, Xdata, y, cv=cv, groups=groups)
        score = float(r2_score(y, pred))
        if score > best_r2:
            best_r2, best_mae = score, float(mean_absolute_error(y, pred))
    return best_r2, best_mae


def model_scores(features: pd.DataFrame) -> pd.DataFrame:
    blocks = feature_blocks(features)
    regime = onehot(features["simple_regime"])
    hierarchy = onehot(features["hierarchical_regime"])
    functional = onehot(features["functional_regime"])
    specs = {
        "simple_regime": regime,
        "hierarchical_regime": hierarchy,
        "functional_regime": functional,
        "capacity": blocks["capacity"],
        "topology_geometry": blocks["geometry"],
        "construction": blocks["construction"],
        "regime_plus_topology": pd.concat([regime.add_prefix("r__"), blocks["geometry"].add_prefix("g__")], axis=1),
        "regime_plus_construction": pd.concat([regime.add_prefix("r__"), blocks["construction"].add_prefix("c__")], axis=1),
        "regime_plus_capacity": pd.concat([regime.add_prefix("r__"), blocks["capacity"].add_prefix("cap__")], axis=1),
        "full": pd.concat([regime.add_prefix("r__"), blocks["full"].add_prefix("f__")], axis=1),
    }
    rows = []
    for spec, X in specs.items():
        r2, mae = grouped_score(features, X)
        rows.append({"model_spec": spec, "grouped_cv_r2": r2, "grouped_cv_mae": mae, "n_features": X.shape[1]})
    return pd.DataFrame(rows)


def _score(scores: pd.DataFrame, spec: str) -> float:
    return float(scores.loc[scores["model_spec"] == spec, "grouped_cv_r2"].iloc[0])


def absorption_analysis(scores: pd.DataFrame) -> pd.DataFrame:
    regime = _score(scores, "simple_regime")
    topo = _score(scores, "topology_geometry")
    cons = _score(scores, "construction")
    reg_topo = _score(scores, "regime_plus_topology")
    reg_cons = _score(scores, "regime_plus_construction")
    topology_unique_before = max(0.0, topo)
    construction_unique_before = max(0.0, cons)
    topology_unique_after = max(0.0, reg_topo - regime)
    construction_unique_after = max(0.0, reg_cons - regime)
    return pd.DataFrame([
        {"feature_block": "topology", "standalone_r2": topo, "regime_plus_block_r2": reg_topo, "unique_after_regime": topology_unique_after, "absorbed_fraction": 1.0 - topology_unique_after / max(1e-12, topology_unique_before)},
        {"feature_block": "construction", "standalone_r2": cons, "regime_plus_block_r2": reg_cons, "unique_after_regime": construction_unique_after, "absorbed_fraction": 1.0 - construction_unique_after / max(1e-12, construction_unique_before)},
    ])


def variance_partition(scores: pd.DataFrame) -> pd.DataFrame:
    regime = max(0.0, _score(scores, "simple_regime"))
    capacity = max(0.0, _score(scores, "capacity"))
    topology = max(0.0, _score(scores, "topology_geometry"))
    construction = max(0.0, _score(scores, "construction"))
    full = max(0.0, _score(scores, "full"))
    return pd.DataFrame([
        {"component": "regime_effect", "variance_fraction": regime},
        {"component": "capacity_gain_over_regime", "variance_fraction": max(0.0, _score(scores, "regime_plus_capacity") - regime)},
        {"component": "topology_gain_over_regime", "variance_fraction": max(0.0, _score(scores, "regime_plus_topology") - regime)},
        {"component": "construction_gain_over_regime", "variance_fraction": max(0.0, _score(scores, "regime_plus_construction") - regime)},
        {"component": "standalone_capacity", "variance_fraction": capacity},
        {"component": "standalone_topology", "variance_fraction": topology},
        {"component": "standalone_construction", "variance_fraction": construction},
        {"component": "residual_full", "variance_fraction": max(0.0, 1.0 - full)},
    ])


def manifold_diagnostics(features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = clean(feature_blocks(features)["full"])
    scaled = StandardScaler().fit_transform(X)
    try:
        import umap
        emb = umap.UMAP(n_neighbors=30, min_dist=0.08, random_state=143).fit_transform(scaled)
    except Exception:
        emb = PCA(n_components=2, random_state=143).fit_transform(scaled)
    labels, _ = pd.factorize(features["simple_regime"].astype(str))
    families, _ = pd.factorize(features["model_id"].astype(str))
    km = KMeans(n_clusters=len(set(labels)), random_state=145, n_init=20).fit_predict(scaled)
    sil = float(silhouette_score(scaled, labels)) if len(set(labels)) > 1 else np.nan
    diag = pd.DataFrame([{"regime_silhouette": sil, "regime_kmeans_ari": float(adjusted_rand_score(labels, km)), "regime_kmeans_nmi": float(normalized_mutual_info_score(labels, km)), "family_vs_regime_ari": float(adjusted_rand_score(families, labels)), "family_vs_regime_nmi": float(normalized_mutual_info_score(families, labels))}])
    return diag, emb


def regime_residuals(features: pd.DataFrame) -> pd.DataFrame:
    X = onehot(features["simple_regime"])
    data = pd.concat([features[["run_id", "model_id", "simple_regime", TARGET, "dataset_id"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    groups = data["dataset_id"].astype(str)
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    pred = cross_val_predict(make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), data[X.columns], data[TARGET], cv=cv, groups=groups)
    out = data[["run_id", "model_id", "simple_regime", TARGET]].copy()
    out["regime_prediction"] = pred
    out["regime_residual"] = out[TARGET] - out["regime_prediction"]
    return out


def decision(scores: pd.DataFrame, absorption: pd.DataFrame) -> tuple[str, str]:
    regime_r2 = _score(scores, "simple_regime")
    topo_abs = float(absorption.loc[absorption["feature_block"] == "topology", "absorbed_fraction"].iloc[0])
    cons_abs = float(absorption.loc[absorption["feature_block"] == "construction", "absorbed_fraction"].iloc[0])
    if regime_r2 >= 0.70 and topo_abs >= 0.50 and cons_abs >= 0.50:
        return "strong_support", "Accessibility is primarily organized by allocator regime."
    if regime_r2 >= 0.50 and (topo_abs >= 0.25 or cons_abs >= 0.25):
        return "moderate_support", "Regimes are major organizers, but mechanisms retain substantial unique signal."
    if regime_r2 < 0.30 and topo_abs < 0.25 and cons_abs < 0.25:
        return "failure", "Regimes are mostly descriptive; topology/construction remain primary."
    return "mixed_support", "Regime membership explains meaningful variance but does not absorb mechanism features enough for the strong claim."


def plot_umap(features: pd.DataFrame, emb: np.ndarray, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    codes, labels = pd.factorize(features["simple_regime"].astype(str))
    ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap="tab10", s=12, alpha=0.7)
    ax.set_title("Accessibility Manifold by Regime")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_partition(partition: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(partition["component"], partition["variance_fraction"], color="slateblue")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("R2 / variance fraction")
    ax.set_title("Accessibility Variance Partition")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_regime_vs_cliffiness(features: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    features.boxplot(column=TARGET, by="simple_regime", ax=ax)
    ax.set_title("Cliffiness by Regime"); fig.suptitle("")
    ax.set_ylabel("cliffiness")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_residuals(residuals: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    residuals.boxplot(column="regime_residual", by="model_id", ax=ax)
    ax.set_title("Regime Model Residuals by Family"); fig.suptitle("")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def write_report(positive_scores: Path = DEFAULT_POSITIVE_SCORES, output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = add_regimes(build_feature_matrix(load_positive_scores(positive_scores)))
    scores = model_scores(features)
    absorption = absorption_analysis(scores)
    partition = variance_partition(scores)
    manifold, emb = manifold_diagnostics(features)
    residuals = regime_residuals(features)
    outcome, text = decision(scores, absorption)
    summary = {"outcome": outcome, "outcome_text": text, "n_runs": int(len(features)), "n_regimes": int(features["simple_regime"].nunique()), "regime_r2": _score(scores, "simple_regime"), "hierarchical_regime_r2": _score(scores, "hierarchical_regime"), "functional_regime_r2": _score(scores, "functional_regime"), "topology_absorbed_fraction": float(absorption.loc[absorption["feature_block"] == "topology", "absorbed_fraction"].iloc[0]), "construction_absorbed_fraction": float(absorption.loc[absorption["feature_block"] == "construction", "absorbed_fraction"].iloc[0]), **manifold.iloc[0].to_dict()}
    report = output_dir / "accessibility_regime_hypothesis.md"
    summary_path = output_dir / "accessibility_regime_summary.json"
    partition_path = output_dir / "regime_variance_partition.csv"
    scores_path = output_dir / "regime_model_scores.csv"
    absorption_path = output_dir / "regime_absorption_analysis.csv"
    manifold_path = output_dir / "regime_manifold_diagnostics.csv"
    residuals_path = output_dir / "regime_residuals.csv"
    features_path = output_dir / "regime_feature_matrix.csv"
    features.to_csv(features_path, index=False); scores.to_csv(scores_path, index=False); absorption.to_csv(absorption_path, index=False); partition.to_csv(partition_path, index=False); manifold.to_csv(manifold_path, index=False); residuals.to_csv(residuals_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    family_note = "Hierarchical family labels are included as a diagnostic, not as evidence for a family-independent regime mechanism."
    report.write_text("\n".join(["# Accessibility Regime Hypothesis", "", "## Executive Answer", text, family_note, "", "## Model Scores", _markdown_table(scores), "", "## Absorption Analysis", _markdown_table(absorption), "", "## Variance Partition", _markdown_table(partition), "", "## Manifold Diagnostics", _markdown_table(manifold)]) + "\n", encoding="utf-8")
    return (
        report,
        summary_path,
        partition_path,
        scores_path,
        absorption_path,
        manifold_path,
        residuals_path,
        features_path,
        plot_umap(features, emb, output_dir / "regime_umap.png"),
        plot_partition(partition, output_dir / "regime_variance_partition.png"),
        plot_regime_vs_cliffiness(features, output_dir / "regime_vs_cliffiness.png"),
        plot_residuals(residuals, output_dir / "regime_residuals.png"),
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
