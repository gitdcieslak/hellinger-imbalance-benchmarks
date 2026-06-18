"""Construct an unsupervised morphology atlas over allocator behavior."""

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
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, r2_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_frontier_morphology_geometry import attach_frontier_features, binwise_frontier  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "topology" / "classical_allocator_morphology.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "topology"
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"
BASE_FEATURES = ["breadth", "elevation"]
FRONTIER_FEATURES = [
    "vertical_slack",
    "normalized_position_between_envelopes",
    "local_frontier_width",
    "nearest_frontier_distance",
    "frontier_curvature",
]


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float) or isinstance(value, np.floating):
                values.append(format(float(value), floatfmt) if pd.notna(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_classical_atlas(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(input_csv)
    if "fit_failed" in df.columns:
        failed = df["fit_failed"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[~failed].copy()
    rename = {"model_id": "allocation_family"}
    df = df.rename(columns=rename)
    if "allocation_family" not in df.columns:
        df["allocation_family"] = "unknown"
    df["accessibility_breadth"] = pd.to_numeric(df["breadth"], errors="coerce")
    df["accessibility_elevation"] = pd.to_numeric(df["elevation"], errors="coerce")
    df["density_separability"] = df["accessibility_elevation"] / (df["accessibility_breadth"] + 1e-6)
    for col in ["breadth", "elevation", TARGET, SURVIVAL, "accessibility_breadth", "accessibility_elevation", "density_separability"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["breadth", "elevation", TARGET, SURVIVAL]).reset_index(drop=True)
    return df


def prepare_feature_space(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    frontier = binwise_frontier(df)
    enriched = attach_frontier_features(df, frontier)
    features = BASE_FEATURES + ["accessibility_breadth", "accessibility_elevation", "density_separability", *FRONTIER_FEATURES]
    enriched = enriched.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        enriched[feature] = pd.to_numeric(enriched[feature], errors="coerce")
        enriched[feature] = enriched[feature].fillna(float(enriched[feature].median()))
    X_raw = enriched[features].to_numpy(dtype=float)
    X_scaled = RobustScaler().fit_transform(X_raw)
    pca_full = PCA(random_state=7).fit(X_scaled)
    n_components = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.95) + 1)
    X_pca = PCA(n_components=n_components, random_state=7).fit_transform(X_scaled)
    notes = []
    try:
        import umap  # type: ignore

        X_umap = umap.UMAP(n_components=2, random_state=7, n_neighbors=30, min_dist=0.05).fit_transform(X_scaled)
        notes.append("UMAP available: used native umap.UMAP for X_umap_2d.")
    except Exception as exc:
        X_umap = PCA(n_components=2, random_state=7).fit_transform(X_scaled)
        notes.append(f"UMAP unavailable or failed; used PCA(2) fallback for X_umap_2d: {exc}")
    return enriched, features, X_raw, X_pca, X_umap, notes


def run_clustering(X_pca: np.ndarray, n_clusters: int = 4) -> tuple[pd.DataFrame, list[str]]:
    labels: dict[str, np.ndarray] = {}
    notes = []
    try:
        import hdbscan  # type: ignore

        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(15, len(X_pca) // 30), min_samples=8)
        labels["hdbscan"] = clusterer.fit_predict(X_pca)
        notes.append("HDBSCAN available: used native hdbscan.HDBSCAN.")
    except Exception as exc:
        labels["hdbscan"] = DBSCAN(eps=0.55, min_samples=8).fit_predict(X_pca)
        notes.append(f"HDBSCAN unavailable or failed; used DBSCAN fallback: {exc}")
    labels["gmm"] = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=7).fit_predict(X_pca)
    labels["kmeans"] = KMeans(n_clusters=n_clusters, n_init=20, random_state=7).fit_predict(X_pca)
    labels["spectral"] = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors", n_neighbors=min(20, len(X_pca) - 1), random_state=7).fit_predict(X_pca)
    return pd.DataFrame(labels), notes


def stability_report(labels: pd.DataFrame, X_pca: np.ndarray) -> pd.DataFrame:
    rows = []
    methods = list(labels.columns)
    for i, left in enumerate(methods):
        valid_left = labels[left].to_numpy()
        for right in methods[i + 1 :]:
            valid_right = labels[right].to_numpy()
            rows.append({"comparison": f"{left}_vs_{right}", "ari": float(adjusted_rand_score(valid_left, valid_right)), "nmi": float(normalized_mutual_info_score(valid_left, valid_right))})
    for method in methods:
        lab = labels[method].to_numpy()
        non_noise = lab >= 0
        if len(np.unique(lab[non_noise])) >= 2 and non_noise.sum() > 2:
            sil = float(silhouette_score(X_pca[non_noise], lab[non_noise]))
        else:
            sil = np.nan
        rows.append({"comparison": f"{method}_silhouette", "ari": np.nan, "nmi": sil})
    return pd.DataFrame(rows)


def bootstrap_stability(X_pca: np.ndarray, base_labels: np.ndarray, n_bootstrap: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    valid = base_labels >= 0
    n_clusters = max(2, len(np.unique(base_labels[valid]))) if valid.any() else 2
    for idx in range(n_bootstrap):
        sample_idx = rng.choice(np.arange(len(X_pca)), size=len(X_pca), replace=True)
        labels_sample = KMeans(n_clusters=n_clusters, n_init=10, random_state=idx).fit_predict(X_pca[sample_idx])
        rows.append({"bootstrap": idx, "ari_to_hdbscan_on_sample": float(adjusted_rand_score(base_labels[sample_idx], labels_sample))})
    return pd.DataFrame(rows)


def cluster_summary(df: pd.DataFrame, cluster_col: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for cluster_id, group in df.groupby(cluster_col):
        if int(cluster_id) < 0:
            continue
        cov = np.cov(group[["breadth", "elevation"]].to_numpy(dtype=float).T) if len(group) > 1 else np.zeros((2, 2))
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_rows": len(group),
                "dominant_family": group["allocation_family"].mode().iloc[0],
                "mean_breadth": float(group["breadth"].mean()),
                "mean_elevation": float(group["elevation"].mean()),
                "mean_vertical_slack": float(group["vertical_slack"].mean()),
                "mean_frontier_position": float(group["normalized_position_between_envelopes"].mean()),
                "mean_cliffiness": float(group[TARGET].mean()),
                "cliffiness_variance": float(group[TARGET].var(ddof=0)),
                "mean_survival_auc": float(group[SURVIVAL].mean()),
                "breadth_elevation_covariance": float(cov[0, 1]),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id")


def bucket(series: pd.Series, n: int = 3) -> pd.Series:
    return pd.qcut(series.rank(method="first"), q=n, labels=False).astype(int)


def separability_tests(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    data = df[df[cluster_col] >= 0].copy()
    if data.empty or data[cluster_col].nunique() < 2:
        return pd.DataFrame()
    X = pd.get_dummies(data[[cluster_col]].astype(str), drop_first=False)
    rows = []
    for target_name, target in [("cliffiness_bucket", bucket(data[TARGET])), ("survival_auc_bucket", bucket(data[SURVIVAL]))]:
        cv = StratifiedKFold(n_splits=min(5, target.value_counts().min()), shuffle=True, random_state=7)
        for model_name, model in [("logistic_regression", LogisticRegression(max_iter=2000)), ("random_forest", RandomForestClassifier(n_estimators=100, random_state=7))]:
            pred = cross_val_predict(model, X, target, cv=cv)
            baseline = float(target.value_counts(normalize=True).max())
            rows.append({"prediction_target": target_name, "model": model_name, "cv_accuracy": float(accuracy_score(target, pred)), "permutation_baseline": baseline, "accuracy_lift": float(accuracy_score(target, pred) - baseline)})
    return pd.DataFrame(rows)


def within_cluster_equations(df: pd.DataFrame, cluster_col: str) -> dict[str, dict[str, float]]:
    features = ["breadth", "elevation", "vertical_slack", "normalized_position_between_envelopes"]
    global_model = Ridge(alpha=1.0).fit(StandardScaler().fit_transform(df[features]), df[TARGET])
    global_r2 = float(r2_score(df[TARGET], global_model.predict(StandardScaler().fit_transform(df[features]))))
    scores: dict[str, dict[str, float]] = {"global_in_sample": {"ridge_r2": global_r2}}
    for cluster_id, group in df[df[cluster_col] >= 0].groupby(cluster_col):
        if len(group) < 8:
            continue
        X = StandardScaler().fit_transform(group[features])
        y = group[TARGET].to_numpy(dtype=float)
        ridge = Ridge(alpha=1.0).fit(X, y)
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=7).fit(X, y)
        rf = RandomForestRegressor(n_estimators=80, min_samples_leaf=4, random_state=7, n_jobs=1).fit(X, y)
        scores[str(int(cluster_id))] = {"ridge_r2": float(r2_score(y, ridge.predict(X))), "shallow_tree_r2": float(r2_score(y, tree.predict(X))), "small_rf_r2": float(r2_score(y, rf.predict(X))), "n_rows": float(len(group))}
    return scores


def hypothesis_tests(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    data = df[df[cluster_col] >= 0].copy()
    overall = float(data[TARGET].var(ddof=0))
    within = float(data.groupby(cluster_col)[TARGET].var(ddof=0).mean())
    means = data.groupby(cluster_col)[TARGET].mean()
    between = float(means.var(ddof=0))
    data["compact_residual"] = data[TARGET] - Ridge(alpha=1.0).fit(data[["breadth", "elevation"]], data[TARGET]).predict(data[["breadth", "elevation"]])
    residual_between = float(data.groupby(cluster_col)["compact_residual"].mean().var(ddof=0))
    corr_values = []
    for _, group in data.groupby(cluster_col):
        corr_values.append(group["vertical_slack"].corr(group[TARGET]) if len(group) > 2 else np.nan)
    frontier_corr = float(np.nanmean(corr_values)) if corr_values else np.nan
    return pd.DataFrame(
        [
            {"test": "between_cluster_cliffiness_variance", "value": between},
            {"test": "within_cluster_cliffiness_variance_mean", "value": within},
            {"test": "between_to_within_ratio", "value": between / max(within, 1e-9)},
            {"test": "compact_equation_residual_between_cluster_variance", "value": residual_between},
            {"test": "mean_within_cluster_frontier_slack_corr", "value": frontier_corr},
            {"test": "overall_cliffiness_variance", "value": overall},
        ]
    )


def decision_case(stability: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame) -> str:
    mean_ari = float(stability["ari"].dropna().mean()) if stability["ari"].notna().any() else 0.0
    ratio_row = tests[tests["test"] == "between_to_within_ratio"]
    ratio = float(ratio_row.iloc[0].value) if not ratio_row.empty else 0.0
    cliff_range = float(summary["mean_cliffiness"].max() - summary["mean_cliffiness"].min()) if not summary.empty else 0.0
    if mean_ari > 0.45 and ratio > 1.0 and cliff_range > 0.2:
        return "Case A — discrete regimes exist"
    if mean_ari < 0.20 and ratio < 0.5:
        return "Case B — fuzzy continuous manifold"
    return "Case C — hybrid structure"


def write_markdown(path: Path, notes: list[str], stability: pd.DataFrame, bootstrap: pd.DataFrame, summary: pd.DataFrame, separability: pd.DataFrame, tests: pd.DataFrame, case: str) -> Path:
    text = "\n".join(
        [
            "# Morphology Atlas Cluster Stability Report",
            "",
            "## Notes",
            *[f"- {note}" for note in notes],
            "",
            "## Cluster Stability",
            _markdown_table(stability),
            "",
            "## Bootstrap Stability",
            _markdown_table(bootstrap.describe().reset_index()),
            "",
            "## Regime Characterization",
            _markdown_table(summary),
            "",
            "## Regime Separability Tests",
            _markdown_table(separability),
            "",
            "## Hypothesis Tests",
            _markdown_table(tests),
            "",
            "## Decision",
            case,
        ]
    ) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def plot_phase(df: pd.DataFrame, cluster_col: str, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["breadth"], df["elevation"], c=df[cluster_col], cmap="tab10", s=22, alpha=0.75)
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Morphology Phase Diagram")
    fig.colorbar(sc, ax=ax, label="cluster_id")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_cliffiness(df: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["breadth"], df["elevation"], c=df[TARGET], cmap="magma", s=22, alpha=0.75)
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Cliffiness Overlay")
    fig.colorbar(sc, ax=ax, label="cliffiness")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_distribution(df: pd.DataFrame, cluster_col: str, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    clusters = sorted([c for c in df[cluster_col].unique() if c >= 0])
    ax.boxplot([df[df[cluster_col] == c][TARGET] for c in clusters], labels=[str(c) for c in clusters])
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("cliffiness")
    ax.set_title("Cluster Cliffiness Distribution")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def write_atlas(input_csv: Path, output_dir: Path) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_classical_atlas(input_csv)
    enriched, features, _, X_pca, X_umap, prep_notes = prepare_feature_space(df)
    labels, cluster_notes = run_clustering(X_pca)
    atlas = pd.concat([enriched.reset_index(drop=True), labels.add_suffix("_cluster")], axis=1)
    primary_col = "hdbscan_cluster"
    stability = stability_report(labels, X_pca)
    bootstrap = bootstrap_stability(X_pca, labels["hdbscan"].to_numpy())
    summary = cluster_summary(atlas, primary_col, features)
    separability = separability_tests(atlas, primary_col)
    equations = within_cluster_equations(atlas, primary_col)
    tests = hypothesis_tests(atlas, primary_col)
    case = decision_case(stability, summary, tests)
    atlas["umap_x"] = X_umap[:, 0]
    atlas["umap_y"] = X_umap[:, 1]
    outputs = []
    clusters_csv = output_dir / "morphology_atlas_clusters.csv"
    atlas.to_csv(clusters_csv, index=False)
    outputs.append(clusters_csv)
    summary_md = output_dir / "cluster_summary_table.md"
    summary_md.write_text("# Cluster Summary Table\n\n" + _markdown_table(summary) + "\n", encoding="utf-8")
    outputs.append(summary_md)
    stability_md = write_markdown(output_dir / "cluster_stability_report.md", [*prep_notes, *cluster_notes], stability, bootstrap, summary, separability, tests, case)
    outputs.append(stability_md)
    equations_json = output_dir / "within_cluster_equation_scores.json"
    equations_json.write_text(json.dumps(equations, indent=2, sort_keys=True), encoding="utf-8")
    outputs.append(equations_json)
    outputs.append(plot_phase(atlas, primary_col, output_dir / "regime_phase_diagram.png"))
    outputs.append(plot_cliffiness(atlas, output_dir / "regime_cliffiness_overlay.png"))
    outputs.append(plot_distribution(atlas, primary_col, output_dir / "cluster_cliffiness_distribution.png"))
    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for path in write_atlas(args.input, args.output_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
