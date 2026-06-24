"""Collapse morphology atlas clusters into interpretable macro-regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, davies_bouldin_score, r2_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATLAS = ROOT / "reports" / "topology" / "morphology_atlas_clusters.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "topology"
CLUSTER_COL = "hdbscan_cluster"
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"
CENTROID_FEATURES = ["mean_breadth", "mean_elevation", "mean_cliffiness", "mean_survival_auc", "mean_vertical_slack", "mean_frontier_position"]


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


def load_atlas(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if CLUSTER_COL not in df.columns:
        raise ValueError(f"missing required cluster column {CLUSTER_COL}")
    return df[df[CLUSTER_COL] >= 0].copy().reset_index(drop=True)


def cluster_centroids(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, group in df.groupby(CLUSTER_COL):
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_rows": len(group),
                "dominant_family": group["allocation_family"].mode().iloc[0] if "allocation_family" in group else "unknown",
                "mean_breadth": float(group["breadth"].mean()),
                "mean_elevation": float(group["elevation"].mean()),
                "mean_cliffiness": float(group[TARGET].mean()),
                "mean_survival_auc": float(group[SURVIVAL].mean()),
                "mean_vertical_slack": float(group["vertical_slack"].mean()),
                "mean_frontier_position": float(group["normalized_position_between_envelopes"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def similarity_matrices(centroids: pd.DataFrame) -> dict[str, pd.DataFrame]:
    X = centroids[CENTROID_FEATURES].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    euclidean = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(axis=2))
    cov = np.cov(Xs.T)
    inv = np.linalg.pinv(cov)
    diff = Xs[:, None, :] - Xs[None, :, :]
    mahal = np.sqrt(np.maximum(0.0, np.einsum("...i,ij,...j->...", diff, inv, diff)))
    cosine_distance = 1.0 - cosine_similarity(Xs)
    ids = centroids["cluster_id"].astype(str).tolist()
    return {
        "euclidean": pd.DataFrame(euclidean, index=ids, columns=ids),
        "mahalanobis": pd.DataFrame(mahal, index=ids, columns=ids),
        "cosine_distance": pd.DataFrame(cosine_distance, index=ids, columns=ids),
    }


def assign_macro_regimes(centroids: pd.DataFrame, k: int) -> pd.DataFrame:
    X = StandardScaler().fit_transform(centroids[CENTROID_FEATURES].to_numpy(dtype=float))
    labels = AgglomerativeClustering(n_clusters=int(k), linkage="ward").fit_predict(X)
    out = centroids.copy()
    out["regime_id"] = labels
    return out


def variance_explained(df: pd.DataFrame, label_col: str) -> float:
    overall = float(df[TARGET].var(ddof=0))
    if overall <= 0:
        return 0.0
    within = df.groupby(label_col)[TARGET].var(ddof=0).fillna(0.0)
    weights = df.groupby(label_col).size() / len(df)
    weighted_within = float((within * weights).sum())
    return float(1.0 - weighted_within / overall)


def evaluate_k_range(df: pd.DataFrame, centroids: pd.DataFrame, k_values=range(2, 9)) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows = []
    assignments = {}
    X = StandardScaler().fit_transform(centroids[CENTROID_FEATURES].to_numpy(dtype=float))
    max_valid_k = max(2, len(centroids) - 1)
    for k in [int(value) for value in k_values if 2 <= int(value) <= max_valid_k]:
        assigned = assign_macro_regimes(centroids, k)
        assignments[int(k)] = assigned
        labels = assigned["regime_id"].to_numpy()
        mapped = df.merge(assigned[["cluster_id", "regime_id"]], left_on=CLUSTER_COL, right_on="cluster_id", how="left")
        valid_metric = 1 < len(np.unique(labels)) < len(X)
        rows.append(
            {
                "k": int(k),
                "silhouette": float(silhouette_score(X, labels)) if valid_metric else np.nan,
                "davies_bouldin": float(davies_bouldin_score(X, labels)) if valid_metric else np.nan,
                "cliffiness_variance_explained": variance_explained(mapped, "regime_id"),
            }
        )
    return pd.DataFrame(rows), assignments


def bucket(series: pd.Series, n: int = 3) -> pd.Series:
    return pd.qcut(series.rank(method="first"), q=n, labels=False).astype(int)


def cv_accuracy(X: pd.DataFrame, y: pd.Series, model) -> float:
    min_count = int(y.value_counts().min())
    cv = StratifiedKFold(n_splits=max(2, min(5, min_count)), shuffle=True, random_state=7)
    pred = cross_val_predict(model, X, y, cv=cv)
    return float(accuracy_score(y, pred))


def cv_r2(X: pd.DataFrame, y: pd.Series, model, max_splits: int = 5) -> float:
    if len(y) < 8 or float(y.var(ddof=0)) <= 0.0:
        return float("nan")
    n_splits = max(2, min(max_splits, len(y) // 4))
    pred = cross_val_predict(model, X, y, cv=KFold(n_splits=n_splits, shuffle=True, random_state=7))
    return float(r2_score(y, pred))


def compression_curve(df: pd.DataFrame, assignments: dict[int, pd.DataFrame]) -> pd.DataFrame:
    y = bucket(df[TARGET])
    baseline = float(y.value_counts(normalize=True).max())
    cluster_acc = cv_accuracy(pd.get_dummies(df[[CLUSTER_COL]].astype(str)), y, LogisticRegression(max_iter=2000))
    rows = [{"representation": "cluster_id", "k": int(df[CLUSTER_COL].nunique()), "cv_accuracy": cluster_acc, "baseline": baseline, "predictive_loss_vs_cluster": 0.0}]
    for k, assignment in sorted(assignments.items(), reverse=True):
        mapped = df.merge(assignment[["cluster_id", "regime_id"]], left_on=CLUSTER_COL, right_on="cluster_id", how="left")
        acc = cv_accuracy(pd.get_dummies(mapped[["regime_id"]].astype(str)), y, LogisticRegression(max_iter=2000))
        rows.append({"representation": "regime_id", "k": k, "cv_accuracy": acc, "baseline": baseline, "predictive_loss_vs_cluster": float(cluster_acc - acc)})
    return pd.DataFrame(rows)


def name_regime(row: pd.Series) -> str:
    b = row.mean_breadth
    e = row.mean_elevation
    c = row.mean_cliffiness
    pos = row.mean_frontier_position
    if b < 0.45 and e < 0.20:
        return "Quantized floor"
    if b > 1.2 and e < 0.12 and c > 0.50:
        return "Broad-flat high-cliff basin"
    if b > 1.2 and e >= 0.18 and pos > 0.55:
        return "Elevated broad plateau"
    if b > 1.1 and 0.10 <= e < 0.25:
        return "Transitional broad ridge"
    if e >= 0.30:
        return "Elevated transition"
    return "Mixed morphology"


def regime_characterization(df: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    mapped = df.merge(assignment[["cluster_id", "regime_id"]], left_on=CLUSTER_COL, right_on="cluster_id", how="left")
    rows = []
    for regime_id, group in mapped.groupby("regime_id"):
        dominant = ", ".join(group["allocation_family"].value_counts().head(3).index.astype(str))
        row = {
            "regime_id": int(regime_id),
            "n": len(group),
            "dominant_families": dominant,
            "mean_breadth": float(group["breadth"].mean()),
            "mean_elevation": float(group["elevation"].mean()),
            "mean_cliffiness": float(group[TARGET].mean()),
            "mean_survival_auc": float(group[SURVIVAL].mean()),
            "mean_frontier_position": float(group["normalized_position_between_envelopes"].mean()),
        }
        row["interpretation"] = name_regime(pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_cliffiness").reset_index(drop=True)


def predictive_power(df: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    mapped = df.merge(assignment[["cluster_id", "regime_id"]], left_on=CLUSTER_COL, right_on="cluster_id", how="left")
    rows = []
    for target_name, y in [("cliffiness_bucket", bucket(mapped[TARGET])), ("survival_auc_bucket", bucket(mapped[SURVIVAL]))]:
        feature_sets = {
            "Cluster ID": pd.get_dummies(mapped[[CLUSTER_COL]].astype(str)),
            "Regime ID": pd.get_dummies(mapped[["regime_id"]].astype(str)),
            "Morphology coordinates": mapped[["breadth", "elevation"]],
            "Regime + coordinates": pd.concat([pd.get_dummies(mapped[["regime_id"]].astype(str)), mapped[["breadth", "elevation"]]], axis=1),
        }
        for name, X in feature_sets.items():
            model = RandomForestClassifier(n_estimators=120, random_state=7) if "coordinates" in name else LogisticRegression(max_iter=2000)
            acc = cv_accuracy(X, y, model)
            rows.append({"prediction_target": target_name, "feature_set": name, "cv_accuracy": acc})
    return pd.DataFrame(rows)


def within_regime_equations(df: pd.DataFrame, assignment: pd.DataFrame) -> dict[str, dict[str, float]]:
    mapped = df.merge(assignment[["cluster_id", "regime_id"]], left_on=CLUSTER_COL, right_on="cluster_id", how="left")
    X_global = mapped[["breadth", "elevation"]]
    y_global = mapped[TARGET]
    global_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    scores: dict[str, dict[str, float]] = {
        "global": {
            "n": float(len(mapped)),
            "ridge_r2": float(r2_score(y_global, global_model.fit(X_global, y_global).predict(X_global))),
            "ridge_cv_r2": cv_r2(X_global, y_global, make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        }
    }
    for regime_id, group in mapped.groupby("regime_id"):
        if len(group) < 8:
            continue
        X = group[["breadth", "elevation"]]
        y = group[TARGET]
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=7).fit(X, y)
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=4, random_state=7, n_jobs=1).fit(X, y)
        scores[str(int(regime_id))] = {
            "n": float(len(group)),
            "ridge_r2": float(r2_score(y, ridge.predict(X))),
            "ridge_cv_r2": cv_r2(X, y, make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
            "shallow_tree_r2": float(r2_score(y, tree.predict(X))),
            "shallow_tree_cv_r2": cv_r2(X, y, DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=7)),
            "small_rf_r2": float(r2_score(y, rf.predict(X))),
            "small_rf_cv_r2": cv_r2(X, y, RandomForestRegressor(n_estimators=100, min_samples_leaf=4, random_state=7, n_jobs=1)),
        }
    return scores


def equation_summary_table(scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for regime_id, values in scores.items():
        row = {"regime_id": regime_id}
        row.update(values)
        rows.append(row)
    columns = ["regime_id", "n", "ridge_r2", "ridge_cv_r2", "shallow_tree_r2", "shallow_tree_cv_r2", "small_rf_r2", "small_rf_cv_r2"]
    return pd.DataFrame(rows).reindex(columns=columns)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def transition_graph(df: pd.DataFrame, assignment: pd.DataFrame, n_neighbors: int = 8) -> pd.DataFrame:
    mapped = df.merge(assignment[["cluster_id", "regime_id"]], left_on=CLUSTER_COL, right_on="cluster_id", how="left")
    coords = mapped[["breadth", "elevation"]].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(mapped))).fit(coords)
    _, indices = nn.kneighbors(coords)
    edges: dict[tuple[int, int], int] = {}
    regimes = mapped["regime_id"].to_numpy(dtype=int)
    for i, neigh in enumerate(indices):
        src = int(regimes[i])
        for j in neigh[1:]:
            dst = int(regimes[j])
            if src != dst:
                edge = tuple(sorted((src, dst)))
                edges[edge] = edges.get(edge, 0) + 1
    if not edges:
        return pd.DataFrame(columns=["regime_a", "regime_b", "edge_count"])
    return pd.DataFrame([{"regime_a": a, "regime_b": b, "edge_count": count} for (a, b), count in sorted(edges.items())]).sort_values("edge_count", ascending=False)


def choose_k(evaluation: pd.DataFrame, compression: pd.DataFrame) -> int:
    candidates = evaluation[(evaluation["k"] >= 3) & (evaluation["k"] <= 6)].copy()
    merged = candidates.merge(compression[compression["representation"] == "regime_id"][["k", "predictive_loss_vs_cluster"]], on="k", how="left")
    merged["score"] = merged["silhouette"].fillna(0) + merged["cliffiness_variance_explained"].fillna(0) - merged["predictive_loss_vs_cluster"].fillna(1)
    return int(merged.sort_values("score", ascending=False).iloc[0].k)


def plot_heatmap(matrix: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title("Cluster Distance Heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_dendrogram(centroids: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage

        X = StandardScaler().fit_transform(centroids[CENTROID_FEATURES])
        dendrogram(linkage(X, method="ward"), labels=centroids["cluster_id"].astype(str).tolist(), ax=ax)
    except Exception as exc:
        ax.text(0.5, 0.5, f"scipy dendrogram unavailable: {exc}", ha="center", va="center")
    ax.set_title("Cluster Dendrogram")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_compression(compression: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    data = compression.sort_values("k")
    ax.plot(data["k"], data["cv_accuracy"], marker="o")
    ax.set_xlabel("regime count")
    ax.set_ylabel("cliffiness bucket CV accuracy")
    ax.set_title("Regime Compression Curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def write_report(output_dir: Path, atlas_csv: Path = DEFAULT_ATLAS) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_atlas(atlas_csv)
    centroids = cluster_centroids(df)
    matrices = similarity_matrices(centroids)
    evaluation, assignments = evaluate_k_range(df, centroids)
    compression = compression_curve(df, assignments)
    k = choose_k(evaluation, compression)
    assignment = assignments[k]
    characterization = regime_characterization(df, assignment)
    predictive = predictive_power(df, assignment)
    equations = within_regime_equations(df, assignment)
    equation_summary = equation_summary_table(equations)
    graph = transition_graph(df, assignment)
    report = output_dir / "morphology_regime_consolidation.md"
    text = "\n".join([
        "# Morphology Regime Consolidation",
        "",
        f"Selected macro-regime count: {k}",
        "",
        "## Hierarchical Regime Discovery",
        _markdown_table(evaluation),
        "",
        "## Regime Compression Curve",
        _markdown_table(compression),
        "",
        "## Regime Characterization",
        _markdown_table(characterization),
        "",
        "## Regime Predictive Power",
        _markdown_table(predictive),
        "",
        "## Cross-Validated Local Accessibility Laws",
        _markdown_table(equation_summary),
        "",
        "## Transition Graph",
        _markdown_table(graph.head(20)),
        "",
        "## Interpretation",
        "The selected macro-regimes summarize the 14 atlas clusters into a smaller operational vocabulary. Compare the compression curve against the cluster baseline to determine how much cliffiness information survives consolidation.",
    ]) + "\n"
    report.write_text(text, encoding="utf-8")
    equations_json = output_dir / "within_regime_equation_scores.json"
    equations_json.write_text(json.dumps(json_safe(equations), indent=2, sort_keys=True), encoding="utf-8")
    return (
        report,
        equations_json,
        plot_heatmap(matrices["euclidean"], output_dir / "cluster_distance_heatmap.png"),
        plot_dendrogram(centroids, output_dir / "cluster_dendrogram.png"),
        plot_compression(compression[compression["representation"].eq("regime_id")], output_dir / "regime_compression_curve.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-csv", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for path in write_report(args.output_dir, atlas_csv=args.atlas_csv):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
