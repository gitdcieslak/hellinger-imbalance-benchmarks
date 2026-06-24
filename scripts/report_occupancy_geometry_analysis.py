"""Analyze whether occupancy geometry explains morphology, regimes, and accessibility."""

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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score
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
MORPHOLOGY_COLS = ["breadth", "elevation"]
DATASET_NUMERIC_COLS = ["positive_fraction", "n_rows", "positive_count", "n_features_processed", "auroc", "average_precision", "brier_score"]


def load_data(path: Path = HDDT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "fit_failed" in df.columns:
        df = df[~df["fit_failed"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    required = ["dataset_name", "model_id", "regime_id", TARGET, SURVIVAL, *MORPHOLOGY_COLS]
    return df.dropna(subset=required).reset_index(drop=True)


def available(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns and df[c].notna().any()]


def region_label(breadth: float, elevation: float) -> str:
    if elevation >= 0.80 and breadth < 0.70:
        return "high_elevation_low_breadth"
    if elevation >= 0.80 and breadth >= 0.70:
        return "high_elevation_moderate_breadth"
    if 0.40 <= elevation < 0.80:
        return "mid_elevation"
    if elevation < 0.40 and breadth >= 0.70:
        return "low_elevation_high_breadth"
    return "low_elevation_low_breadth"


def feature_matrix(df: pd.DataFrame, numeric: list[str] | None = None, categorical: list[str] | None = None) -> pd.DataFrame:
    pieces = []
    if numeric:
        cols = available(numeric, df)
        if cols:
            pieces.append(df[cols].astype(float).reset_index(drop=True))
    if categorical:
        for col in categorical:
            if col in df.columns:
                pieces.append(pd.get_dummies(df[col].astype(str), prefix=col).reset_index(drop=True))
    if not pieces:
        return pd.DataFrame(index=df.index)
    return pd.concat(pieces, axis=1)


def grouped_regression_r2(df: pd.DataFrame, X: pd.DataFrame, target: str, group_col: str = "dataset_name") -> tuple[float, bool, str]:
    if X.shape[1] == 0:
        return np.nan, False, "no_features"
    data = pd.concat([df[[target, group_col]].reset_index(drop=True), X.reset_index(drop=True)], axis=1).dropna()
    if len(data) < 20:
        return np.nan, False, "too_few_rows"
    groups = data[group_col].astype(str)
    if groups.nunique() < 2:
        return np.nan, False, "too_few_groups"
    if data[target].var(ddof=0) <= 1e-12:
        return np.nan, False, "constant_target"
    try:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
        pred = cross_val_predict(make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), data.drop(columns=[target, group_col]), data[target], cv=cv, groups=groups)
        return float(r2_score(data[target], pred)), True, ""
    except Exception as exc:
        return np.nan, False, type(exc).__name__


def grouped_classification(df: pd.DataFrame, X: pd.DataFrame, target: str = "regime_id", group_col: str = "dataset_name") -> tuple[pd.DataFrame, np.ndarray]:
    data = pd.concat([df[[target, group_col]].reset_index(drop=True), X.reset_index(drop=True)], axis=1).dropna()
    y = data[target].astype(str)
    groups = data[group_col].astype(str)
    rows = []
    predictions = {}
    if X.shape[1] == 0 or groups.nunique() < 2 or y.nunique() < 2:
        return pd.DataFrame([{"classifier": "unavailable", "grouped_cv_accuracy": np.nan, "macro_f1": np.nan, "valid": False, "skip_reason": "insufficient_data"}]), np.empty((0, 0), dtype=int)
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    models = {
        "multinomial_logistic_regression": make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000)),
        "random_forest_classifier": RandomForestClassifier(n_estimators=300, min_samples_leaf=4, random_state=0, n_jobs=-1),
    }
    Xdata = data.drop(columns=[target, group_col])
    for name, model in models.items():
        try:
            pred = cross_val_predict(model, Xdata, y, cv=cv, groups=groups)
            predictions[name] = pred
            rows.append({"classifier": name, "grouped_cv_accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")), "valid": True, "skip_reason": ""})
        except Exception as exc:
            rows.append({"classifier": name, "grouped_cv_accuracy": np.nan, "macro_f1": np.nan, "valid": False, "skip_reason": type(exc).__name__})
    best_name = max((r for r in rows if r["valid"]), key=lambda r: r["grouped_cv_accuracy"], default=None)
    labels = sorted(y.unique())
    cm = confusion_matrix(y, predictions[best_name["classifier"]], labels=labels) if best_name else np.empty((0, 0), dtype=int)
    return pd.DataFrame(rows), cm


def occupancy_to_morphology(df: pd.DataFrame) -> pd.DataFrame:
    X_occ = feature_matrix(df, OCCUPANCY_COLS)
    rows = []
    for target in ["breadth", "elevation"]:
        score, valid, reason = grouped_regression_r2(df, X_occ, target)
        rows.append({"target": target, "feature_set": "occupancy", "grouped_cv_r2": score, "valid": valid, "skip_reason": reason})
    breadth = rows[0]["grouped_cv_r2"]
    elevation = rows[1]["grouped_cv_r2"]
    joint = float(np.nanmean([breadth, elevation])) if np.isfinite([breadth, elevation]).any() else np.nan
    rows.append({"target": "joint_morphology_position", "feature_set": "occupancy", "grouped_cv_r2": joint, "valid": bool(np.isfinite(joint)), "skip_reason": "" if np.isfinite(joint) else "no_valid_scores"})
    return pd.DataFrame(rows)


def morphology_redundancy_scores(df: pd.DataFrame) -> pd.DataFrame:
    occ = available(OCCUPANCY_COLS, df)
    specs = {
        "morphology_only": feature_matrix(df, MORPHOLOGY_COLS),
        "occupancy_only": feature_matrix(df, occ),
        "morphology_plus_occupancy": feature_matrix(df, MORPHOLOGY_COLS + occ),
    }
    rows = []
    for target in [TARGET, SURVIVAL]:
        scores = {}
        for name, X in specs.items():
            score, valid, reason = grouped_regression_r2(df, X, target)
            scores[name] = score
            rows.append({"target": target, "model_spec": name, "grouped_cv_r2": score, "valid": valid, "skip_reason": reason})
        rows.append({"target": target, "model_spec": "delta_occupancy_after_morphology", "grouped_cv_r2": scores["morphology_plus_occupancy"] - scores["morphology_only"], "valid": True, "skip_reason": ""})
        rows.append({"target": target, "model_spec": "delta_morphology_after_occupancy", "grouped_cv_r2": scores["morphology_plus_occupancy"] - scores["occupancy_only"], "valid": True, "skip_reason": ""})
    return pd.DataFrame(rows)


def occupancy_manifold(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    occ = available(OCCUPANCY_COLS, df)
    data = df.dropna(subset=occ).copy()
    X = StandardScaler().fit_transform(data[occ].astype(float))
    try:
        import umap

        emb = umap.UMAP(n_neighbors=25, min_dist=0.05, random_state=0).fit_transform(X)
    except Exception:
        from sklearn.decomposition import PCA

        emb = PCA(n_components=2, random_state=0).fit_transform(X)
    try:
        import hdbscan

        labels = hdbscan.HDBSCAN(min_cluster_size=45, min_samples=10).fit_predict(X)
    except Exception:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=0.8, min_samples=10).fit_predict(X)
    data["occupancy_umap_x"] = emb[:, 0]
    data["occupancy_umap_y"] = emb[:, 1]
    data["occupancy_cluster"] = labels
    data["morphology_region"] = [region_label(b, e) for b, e in zip(data["breadth"], data["elevation"])]
    summary = cluster_summary(data, occ)
    return data, summary


def archetype_name(row: pd.Series) -> str:
    unique_ratio = row.get("mean_positive_unique_score_ratio", np.nan)
    top_mass = row.get("mean_positive_top_bin_mass", np.nan)
    iqr = row.get("mean_positive_score_iqr", np.nan)
    entropy = row.get("mean_positive_histogram_entropy", np.nan)
    if np.isfinite(top_mass) and top_mass > 0.93 and np.isfinite(unique_ratio) and unique_ratio < 0.15:
        return "Monopolized quantized allocator"
    if np.isfinite(iqr) and iqr < 0.02:
        return "Compressed allocator"
    if np.isfinite(unique_ratio) and unique_ratio > 0.55 and np.isfinite(entropy) and entropy > 1.0:
        return "Diffuse allocator"
    if np.isfinite(entropy) and entropy > 0.8:
        return "Layered allocator"
    return "Quantized allocator"


def cluster_summary(data: pd.DataFrame, occ: list[str]) -> pd.DataFrame:
    valid = data[data["occupancy_cluster"] >= 0].copy()
    if valid.empty:
        return pd.DataFrame()
    agg = {
        "n_runs": (TARGET, "size"),
        "mean_breadth": ("breadth", "mean"),
        "mean_elevation": ("elevation", "mean"),
        "mean_survival_auc": (SURVIVAL, "mean"),
        "mean_cliffiness": (TARGET, "mean"),
        "dominant_models": ("model_id", lambda s: ", ".join(s.value_counts().head(5).index.astype(str))),
        "dominant_datasets": ("dataset_name", lambda s: ", ".join(s.value_counts().head(6).index.astype(str))),
        "dominant_regime": ("regime_id", lambda s: str(s.value_counts().idxmax())),
        "dominant_region": ("morphology_region", lambda s: str(s.value_counts().idxmax())),
    }
    for col in occ:
        agg[f"mean_{col}"] = (col, "mean")
    summary = valid.groupby("occupancy_cluster", as_index=False).agg(**agg)
    summary["archetype"] = summary.apply(archetype_name, axis=1)
    return summary.sort_values("n_runs", ascending=False).reset_index(drop=True)


def variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    occ = available(OCCUPANCY_COLS, df)
    dataset_numeric = available(DATASET_NUMERIC_COLS, df)
    specs = {
        "occupancy_only": feature_matrix(df, occ),
        "occupancy_plus_model": feature_matrix(df, occ, ["model_id"]),
        "occupancy_plus_dataset_identity": feature_matrix(df, occ, ["dataset_name"]),
        "occupancy_plus_dataset_numeric": feature_matrix(df, occ + dataset_numeric),
        "occupancy_plus_morphology": feature_matrix(df, occ + MORPHOLOGY_COLS),
        "all_available": feature_matrix(df, occ + MORPHOLOGY_COLS + dataset_numeric, ["model_id"]),
        "morphology_only": feature_matrix(df, MORPHOLOGY_COLS),
        "model_only": feature_matrix(df, categorical=["model_id"]),
        "dataset_identity_only": feature_matrix(df, categorical=["dataset_name"]),
    }
    rows = []
    for target in [SURVIVAL, TARGET]:
        occ_score = np.nan
        for name, X in specs.items():
            score, valid, reason = grouped_regression_r2(df, X, target)
            if name == "occupancy_only":
                occ_score = score
            rows.append({"target": target, "model_spec": name, "grouped_cv_r2": score, "delta_vs_occupancy": score - occ_score if np.isfinite(score) and np.isfinite(occ_score) else np.nan, "n_features": int(X.shape[1]), "valid": valid, "skip_reason": reason})
    return pd.DataFrame(rows)


def outcome_decision(morphology_scores: pd.DataFrame, regime_scores: pd.DataFrame, redundancy: pd.DataFrame, variance: pd.DataFrame) -> tuple[str, str]:
    joint = score_lookup(morphology_scores, "joint_morphology_position", "grouped_cv_r2")
    regime_acc = float(regime_scores["grouped_cv_accuracy"].max()) if not regime_scores.empty else np.nan
    cliff_occ = lookup_target_spec(redundancy, TARGET, "occupancy_only")
    cliff_morph = lookup_target_spec(redundancy, TARGET, "morphology_only")
    cliff_both = lookup_target_spec(redundancy, TARGET, "morphology_plus_occupancy")
    morph_adds = cliff_both - cliff_occ if np.isfinite(cliff_both) and np.isfinite(cliff_occ) else np.nan
    occ_adds = cliff_both - cliff_morph if np.isfinite(cliff_both) and np.isfinite(cliff_morph) else np.nan
    if joint > 0.60 and regime_acc > 0.80 and cliff_occ > cliff_morph and morph_adds < 0.05:
        return "Outcome A", "Occupancy is fundamental: it predicts morphology, regimes, and cliffiness, while morphology adds little after occupancy."
    if np.isfinite(occ_adds) and np.isfinite(morph_adds) and occ_adds > 0.05 and morph_adds > 0.05:
        return "Outcome B", "Occupancy and morphology are complementary: both add unique accessibility signal."
    return "Outcome C", "Occupancy is descriptive or downstream under the current evidence."


def score_lookup(df: pd.DataFrame, key: str, value_col: str) -> float:
    for col in ["target", "model_spec", "classifier"]:
        if col in df.columns:
            match = df[df[col] == key]
            if not match.empty:
                return float(match[value_col].iloc[0])
    return np.nan


def lookup_target_spec(df: pd.DataFrame, target: str, spec: str) -> float:
    match = df[(df["target"] == target) & (df["model_spec"] == spec)]
    return float(match["grouped_cv_r2"].iloc[0]) if not match.empty else np.nan


def plot_occupancy_umap(data: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(data["occupancy_umap_x"], data["occupancy_umap_y"], c=data[TARGET], cmap="magma", s=12, alpha=0.75)
    ax.set_title("Occupancy UMAP Colored by Cliffiness")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.colorbar(sc, ax=ax, label="cliffiness")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_occupancy_clusters(data: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title in [(axes[0], "occupancy_cluster", "Occupancy Clusters"), (axes[1], "regime_id", "Projected Regimes"), (axes[2], "model_id", "Model Families")]:
        codes, uniques = pd.factorize(data[col].astype(str))
        sc = ax.scatter(data["occupancy_umap_x"], data["occupancy_umap_y"], c=codes, cmap="tab20", s=10, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_occupancy_vs_morphology(df: pd.DataFrame, output: Path) -> Path:
    metrics = ["positive_effective_score_bins", "positive_unique_score_ratio", "positive_histogram_entropy", "positive_score_iqr"]
    fig, axes = plt.subplots(2, len(metrics), figsize=(4 * len(metrics), 7))
    for j, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        axes[0, j].scatter(df[metric], df["breadth"], s=10, alpha=0.4)
        axes[0, j].set_title(f"breadth vs {metric}")
        axes[1, j].scatter(df[metric], df["elevation"], s=10, alpha=0.4)
        axes[1, j].set_title(f"elevation vs {metric}")
        axes[1, j].set_xlabel(metric)
    axes[0, 0].set_ylabel("breadth")
    axes[1, 0].set_ylabel("elevation")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_occupancy_heatmaps(df: pd.DataFrame, output: Path) -> Path:
    metrics = ["positive_effective_score_bins", "positive_unique_score_ratio", "positive_histogram_entropy", "positive_score_iqr"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, metric in zip(axes.ravel(), metrics):
        sc = ax.scatter(df["breadth"], df["elevation"], c=df[metric], cmap="viridis", s=10, alpha=0.65)
        ax.set_title(metric)
        ax.set_xlabel("breadth")
        ax.set_ylabel("elevation")
        fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_bar(df: pd.DataFrame, x: str, y: str, output: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    view = df.dropna(subset=[y]).copy()
    ax.bar(view[x].astype(str), view[y], color="steelblue")
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_archetypes(summary: pd.DataFrame, output: Path) -> Path:
    lines = ["# Occupancy Archetypes", ""]
    if summary.empty:
        lines.append("No non-noise occupancy clusters were found.")
    for _, row in summary.iterrows():
        lines.extend(
            [
                f"## Cluster {int(row['occupancy_cluster'])}: {row['archetype']}",
                "",
                f"Runs: {int(row['n_runs'])}",
                "",
                f"Morphology: breadth={row['mean_breadth']:.4f}, elevation={row['mean_elevation']:.4f}",
                "",
                f"Accessibility: survival_auc={row['mean_survival_auc']:.4f}, cliffiness={row['mean_cliffiness']:.4f}",
                "",
                f"Dominant models: {row['dominant_models']}",
                "",
                f"Dominant datasets: {row['dominant_datasets']}",
                "",
                f"Signature: unique_ratio={row.get('mean_positive_unique_score_ratio', np.nan):.4f}, top_bin_mass={row.get('mean_positive_top_bin_mass', np.nan):.4f}, score_iqr={row.get('mean_positive_score_iqr', np.nan):.4f}, entropy={row.get('mean_positive_histogram_entropy', np.nan):.4f}",
                "",
            ]
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_report(output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df["morphology_region"] = [region_label(b, e) for b, e in zip(df["breadth"], df["elevation"])]
    morphology_scores = occupancy_to_morphology(df)
    X_occ = feature_matrix(df, OCCUPANCY_COLS)
    regime_scores, regime_cm = grouped_classification(df, X_occ)
    redundancy = morphology_redundancy_scores(df)
    manifold, cluster_stats = occupancy_manifold(df)
    decomposition = variance_decomposition(df)
    outcome, outcome_text = outcome_decision(morphology_scores, regime_scores, redundancy, decomposition)
    cluster_csv = output_dir / "occupancy_cluster_summary.csv"
    cluster_stats.to_csv(cluster_csv, index=False)
    decomp_csv = output_dir / "occupancy_variance_decomposition.csv"
    decomposition.to_csv(decomp_csv, index=False)
    archetypes_md = write_archetypes(cluster_stats, output_dir / "occupancy_archetypes.md")
    summary = {
        "n_rows": int(len(df)),
        "available_occupancy_metrics": available(OCCUPANCY_COLS, df),
        "occupancy_to_breadth_r2": lookup_target_spec_like(morphology_scores, "breadth"),
        "occupancy_to_elevation_r2": lookup_target_spec_like(morphology_scores, "elevation"),
        "occupancy_to_joint_morphology_r2": lookup_target_spec_like(morphology_scores, "joint_morphology_position"),
        "best_regime_accuracy": float(regime_scores["grouped_cv_accuracy"].max()),
        "best_regime_macro_f1": float(regime_scores["macro_f1"].max()),
        "cliffiness_morphology_r2": lookup_target_spec(redundancy, TARGET, "morphology_only"),
        "cliffiness_occupancy_r2": lookup_target_spec(redundancy, TARGET, "occupancy_only"),
        "cliffiness_morphology_plus_occupancy_r2": lookup_target_spec(redundancy, TARGET, "morphology_plus_occupancy"),
        "survival_morphology_r2": lookup_target_spec(redundancy, SURVIVAL, "morphology_only"),
        "survival_occupancy_r2": lookup_target_spec(redundancy, SURVIVAL, "occupancy_only"),
        "survival_morphology_plus_occupancy_r2": lookup_target_spec(redundancy, SURVIVAL, "morphology_plus_occupancy"),
        "occupancy_cluster_count": int(cluster_stats["occupancy_cluster"].nunique()) if not cluster_stats.empty else 0,
        "outcome": outcome,
        "outcome_text": outcome_text,
    }
    summary_json = output_dir / "occupancy_geometry_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    cm_table = pd.DataFrame(regime_cm) if regime_cm.size else pd.DataFrame()
    report = output_dir / "occupancy_geometry_analysis.md"
    report.write_text(
        "\n".join(
            [
                "# Occupancy Geometry Analysis",
                "",
                "## Executive Answer",
                executive_answer(summary),
                "",
                "## Q1: Can Occupancy Reconstruct Morphology?",
                _markdown_table(morphology_scores),
                "",
                "Caveat: the near-perfect reconstruction should be read as evidence that the current morphology coordinates are projections of the recorded occupancy statistics. It is not an independent causal identification result, because several occupancy columns are operationally close to the breadth/elevation definitions.",
                "",
                "## Q2: Can Occupancy Reconstruct Regime Assignments?",
                _markdown_table(regime_scores),
                "",
                "Best-classifier confusion matrix:",
                _markdown_table(cm_table),
                "",
                "## Q3: Is Morphology Redundant Once Occupancy Is Known?",
                _markdown_table(redundancy),
                "",
                "## Q4: Occupancy Manifold Discovery",
                _markdown_table(cluster_stats),
                "",
                "## Q5: Occupancy to Morphology Mapping",
                "See `occupancy_vs_morphology.png` and `occupancy_heatmaps.png`.",
                "",
                "## Q6: Occupancy Mechanism Decomposition",
                _markdown_table(decomposition),
                "",
                "## Q7: Occupancy Archetypes",
                "See `occupancy_archetypes.md`.",
                "",
                "## Decision",
                f"{outcome}: {outcome_text}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outputs = [
        report,
        plot_occupancy_umap(manifold, output_dir / "occupancy_umap.png"),
        plot_occupancy_clusters(manifold, output_dir / "occupancy_clusters.png"),
        plot_occupancy_vs_morphology(df, output_dir / "occupancy_vs_morphology.png"),
        plot_occupancy_heatmaps(df, output_dir / "occupancy_heatmaps.png"),
        plot_bar(regime_scores, "classifier", "grouped_cv_accuracy", output_dir / "regime_from_occupancy.png", "Regime Prediction from Occupancy"),
        plot_bar(morphology_scores, "target", "grouped_cv_r2", output_dir / "morphology_from_occupancy.png", "Morphology Prediction from Occupancy"),
        decomp_csv,
        cluster_csv,
        archetypes_md,
        summary_json,
    ]
    return tuple(outputs)


def lookup_target_spec_like(df: pd.DataFrame, target: str) -> float:
    match = df[df["target"] == target]
    return float(match["grouped_cv_r2"].iloc[0]) if not match.empty else np.nan


def executive_answer(summary: dict[str, object]) -> str:
    return (
        f"Occupancy-to-joint-morphology R2 is {summary['occupancy_to_joint_morphology_r2']:.4f}; "
        f"best regime accuracy from occupancy is {summary['best_regime_accuracy']:.4f}; "
        f"cliffiness R2 is morphology={summary['cliffiness_morphology_r2']:.4f}, occupancy={summary['cliffiness_occupancy_r2']:.4f}, both={summary['cliffiness_morphology_plus_occupancy_r2']:.4f}. "
        f"Decision: {summary['outcome_text']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
