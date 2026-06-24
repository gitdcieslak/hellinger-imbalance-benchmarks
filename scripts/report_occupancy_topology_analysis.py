"""Analyze whether occupancy-topology proxies explain residual cliffiness."""

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
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
TOPOLOGY_BASE_COLS = [
    "minority_survival_max_drop",
    "minority_survival_effective_drop_count",
    "positive_score_iqr",
    "positive_score_q10_q90_width",
    "positive_score_gini_or_concentration_index",
    "positive_effective_score_bins",
    "positive_histogram_entropy",
    "positive_unique_score_ratio",
    "positive_top_bin_mass",
    "positive_max_bin_mass",
]


def load_data(path: Path = HDDT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "fit_failed" in df.columns:
        df = df[~df["fit_failed"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    required = ["dataset_name", "model_id", "regime_id", "breadth", "elevation", TARGET, SURVIVAL]
    return df.dropna(subset=required).reset_index(drop=True)


def available(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns and df[c].notna().any()]


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


def build_topology_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    unique_ratio = df.get("positive_unique_score_ratio", pd.Series(np.nan, index=df.index)).astype(float)
    entropy = df.get("positive_histogram_entropy", pd.Series(np.nan, index=df.index)).astype(float)
    top_mass = df.get("positive_top_bin_mass", pd.Series(np.nan, index=df.index)).astype(float)
    max_mass = df.get("positive_max_bin_mass", pd.Series(np.nan, index=df.index)).astype(float)
    iqr = df.get("positive_score_iqr", pd.Series(np.nan, index=df.index)).astype(float)
    q_width = df.get("positive_score_q10_q90_width", pd.Series(np.nan, index=df.index)).astype(float)
    gini = df.get("positive_score_gini_or_concentration_index", pd.Series(np.nan, index=df.index)).astype(float)
    eff_bins = df.get("positive_effective_score_bins", pd.Series(np.nan, index=df.index)).astype(float)
    max_drop = df.get("minority_survival_max_drop", pd.Series(np.nan, index=df.index)).astype(float)
    eff_drops = df.get("minority_survival_effective_drop_count", pd.Series(np.nan, index=df.index)).astype(float)

    out["n_unique_posteriors_proxy"] = unique_ratio
    out["alphabet_entropy"] = entropy
    out["alphabet_concentration"] = top_mass
    out["alphabet_gini"] = gini
    out["mean_gap_proxy"] = q_width / eff_bins.replace(0, np.nan)
    out["median_gap_proxy"] = iqr / eff_bins.replace(0, np.nan)
    out["max_gap_proxy"] = q_width
    out["gap_entropy_proxy"] = entropy
    out["gap_gini_proxy"] = gini
    out["largest_drop"] = max_drop
    out["top_5_drop_mass_proxy"] = np.minimum(1.0, 5.0 * max_drop)
    out["drop_entropy_proxy"] = np.log1p(eff_drops)
    out["drop_gini_proxy"] = 1.0 / eff_drops.replace(0, np.nan)
    out["effective_drop_count"] = eff_drops
    out["peak_slope_proxy"] = max_drop
    out["mean_slope_proxy"] = 1.0 / eff_drops.replace(0, np.nan)
    out["slope_entropy_proxy"] = np.log1p(eff_drops)
    out["curvature_entropy_proxy"] = entropy * (1.0 - max_mass.clip(0, 1))
    return out.replace([np.inf, -np.inf], np.nan)


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


def grouped_cv_r2(df: pd.DataFrame, X: pd.DataFrame, target: str = TARGET, group_col: str = "dataset_name") -> tuple[float, bool, str]:
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


def variance_decomposition(df: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([df.reset_index(drop=True), topology.add_prefix("topology__").reset_index(drop=True)], axis=1)
    occ = available(OCCUPANCY_COLS, data)
    topo = [c for c in data.columns if c.startswith("topology__") and data[c].notna().any()]
    specs = {
        "occupancy_only": feature_matrix(data, occ),
        "topology_only": feature_matrix(data, topo),
        "occupancy_plus_topology": feature_matrix(data, occ + topo),
        "occupancy_plus_model": feature_matrix(data, occ, ["model_id"]),
        "occupancy_plus_topology_plus_model": feature_matrix(data, occ + topo, ["model_id"]),
        "all_with_morphology": feature_matrix(data, occ + topo + ["breadth", "elevation"], ["model_id"]),
    }
    rows = []
    scores = {}
    for name, X in specs.items():
        score, valid, reason = grouped_cv_r2(data, X)
        scores[name] = score
        rows.append({"model_spec": name, "grouped_cv_r2": score, "n_features": int(X.shape[1]), "valid": valid, "skip_reason": reason})
    baseline = scores.get("occupancy_only", np.nan)
    for row in rows:
        row["delta_vs_occupancy"] = row["grouped_cv_r2"] - baseline if np.isfinite(row["grouped_cv_r2"]) and np.isfinite(baseline) else np.nan
    return pd.DataFrame(rows)


def mediation_summary(decomp: pd.DataFrame) -> dict[str, float | str]:
    occ = score_for(decomp, "occupancy_only")
    occ_model = score_for(decomp, "occupancy_plus_model")
    occ_top = score_for(decomp, "occupancy_plus_topology")
    occ_top_model = score_for(decomp, "occupancy_plus_topology_plus_model")
    model_contribution_without_topology = occ_model - occ if np.isfinite(occ_model) and np.isfinite(occ) else np.nan
    model_contribution_after_topology = occ_top_model - occ_top if np.isfinite(occ_top_model) and np.isfinite(occ_top) else np.nan
    reduction = 1.0 - (model_contribution_after_topology / model_contribution_without_topology) if model_contribution_without_topology and np.isfinite(model_contribution_after_topology) else np.nan
    return {
        "occupancy_r2": occ,
        "occupancy_plus_model_r2": occ_model,
        "occupancy_plus_topology_r2": occ_top,
        "occupancy_plus_topology_plus_model_r2": occ_top_model,
        "direct_model_effect_without_topology": model_contribution_without_topology,
        "residual_model_effect_after_topology": model_contribution_after_topology,
        "topology_mediated_fraction_of_model_effect": reduction,
        "topology_absorbs_model_signal": bool(np.isfinite(reduction) and reduction > 0.50),
    }


def score_for(df: pd.DataFrame, model_spec: str) -> float:
    match = df[df["model_spec"] == model_spec]
    return float(match["grouped_cv_r2"].iloc[0]) if not match.empty else np.nan


def permutation_importance_table(df: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([df[[TARGET]].reset_index(drop=True), topology.reset_index(drop=True)], axis=1).dropna()
    if len(data) < 50 or topology.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=6, random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)
    imp = permutation_importance(model, X_test, y_test, n_repeats=12, random_state=0, n_jobs=-1)
    out = pd.DataFrame({"feature": X.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std})
    return out.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def cliffy_structure_contrasts(df: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([df.reset_index(drop=True), topology.reset_index(drop=True)], axis=1)
    data["morphology_region"] = morphology_region(data)
    rows = []
    for region, group in data.groupby("morphology_region"):
        high = group[group[TARGET] > 0.8]
        low = group[group[TARGET] < 0.2]
        if high.empty or low.empty:
            continue
        for col in topology.columns:
            rows.append({"morphology_region": region, "feature": col, "high_cliff_mean": float(high[col].mean()), "low_cliff_mean": float(low[col].mean()), "delta_high_minus_low": float(high[col].mean() - low[col].mean()), "n_high": int(len(high)), "n_low": int(len(low))})
    return pd.DataFrame(rows).sort_values(["morphology_region", "delta_high_minus_low"], ascending=[True, False])


def topology_manifold(df: pd.DataFrame, topology: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.concat([df.reset_index(drop=True), topology.reset_index(drop=True)], axis=1).dropna(subset=topology.columns).copy()
    X = StandardScaler().fit_transform(data[topology.columns].astype(float))
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
    data["topology_umap_x"] = emb[:, 0]
    data["topology_umap_y"] = emb[:, 1]
    data["topology_cluster"] = labels
    data["morphology_region"] = morphology_region(data)
    clusters = data[data["topology_cluster"] >= 0].groupby("topology_cluster", as_index=False).agg(
        n_runs=(TARGET, "size"),
        mean_cliffiness=(TARGET, "mean"),
        mean_survival_auc=(SURVIVAL, "mean"),
        mean_largest_drop=("largest_drop", "mean"),
        mean_effective_drop_count=("effective_drop_count", "mean"),
        mean_max_gap_proxy=("max_gap_proxy", "mean"),
        dominant_models=("model_id", lambda s: ", ".join(s.value_counts().head(5).index.astype(str))),
        dominant_regime=("regime_id", lambda s: str(s.value_counts().idxmax())),
        dominant_region=("morphology_region", lambda s: str(s.value_counts().idxmax())),
    )
    return data, clusters


def classify_outcome(decomp: pd.DataFrame, mediation: dict[str, float | str]) -> tuple[str, str]:
    occ = score_for(decomp, "occupancy_only")
    occ_top = score_for(decomp, "occupancy_plus_topology")
    occ_top_model = score_for(decomp, "occupancy_plus_topology_plus_model")
    delta_top = occ_top - occ if np.isfinite(occ_top) and np.isfinite(occ) else np.nan
    model_after_top = occ_top_model - occ_top if np.isfinite(occ_top_model) and np.isfinite(occ_top) else np.nan
    if np.isfinite(delta_top) and delta_top > 0.10 and bool(mediation.get("topology_absorbs_model_signal")):
        return "Outcome A", "Topology absorbs much of the model-family signal: Model -> Occupancy Topology -> Cliffiness."
    if np.isfinite(model_after_top) and model_after_top > 0.10:
        return "Outcome B", "Model-family signal remains after topology; algorithm-specific threshold behavior is not fully captured."
    if np.isfinite(delta_top) and delta_top > 0.10 and occ_top > score_for(decomp, "occupancy_plus_model"):
        return "Outcome C", "Reachability/topology geometry dominates conventional occupancy metrics."
    return "Outcome B", "Topology adds limited signal or model-family effects remain substantial."


def plot_umap(data: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(data["topology_umap_x"], data["topology_umap_y"], c=data[TARGET], cmap="magma", s=12, alpha=0.75)
    ax.set_title("Occupancy Topology UMAP")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.colorbar(sc, ax=ax, label="cliffiness")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_clusters(data: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title in [(axes[0], "topology_cluster", "Topology Clusters"), (axes[1], "model_id", "Model Family"), (axes[2], "regime_id", "Regime")]:
        codes, _ = pd.factorize(data[col].astype(str))
        ax.scatter(data["topology_umap_x"], data["topology_umap_y"], c=codes, cmap="tab20", s=10, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_vs_cliffiness(df: pd.DataFrame, topology: pd.DataFrame, output: Path) -> Path:
    cols = ["largest_drop", "effective_drop_count", "max_gap_proxy", "alphabet_entropy"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.ravel(), cols):
        ax.scatter(topology[col], df[TARGET], s=10, alpha=0.45)
        ax.set_xlabel(col)
        ax.set_ylabel("cliffiness")
    fig.suptitle("Topology Features vs Cliffiness")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_jump_spectra(df: pd.DataFrame, topology: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    low = topology[df[TARGET] < 0.2]
    high = topology[df[TARGET] > 0.8]
    bins = np.linspace(0, np.nanmax(topology["largest_drop"]), 25)
    ax.hist(low["largest_drop"], bins=bins, alpha=0.6, label="cliffiness < 0.2")
    ax.hist(high["largest_drop"], bins=bins, alpha=0.6, label="cliffiness > 0.8")
    ax.set_xlabel("largest reachability drop proxy")
    ax.set_ylabel("runs")
    ax.set_title("Reachability Jump Spectra")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_importance(importance: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(12).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], color="steelblue")
    ax.set_xlabel("permutation importance")
    ax.set_title("Topology Feature Importance")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_report(output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data()
    topology = build_topology_features(df)
    decomp = variance_decomposition(df, topology)
    mediation = mediation_summary(decomp)
    importance = permutation_importance_table(df, topology)
    contrasts = cliffy_structure_contrasts(df, topology)
    manifold, clusters = topology_manifold(df, topology)
    outcome, outcome_text = classify_outcome(decomp, mediation)
    summary = {
        "n_rows": int(len(df)),
        "available_topology_proxy_features": list(topology.columns),
        "occupancy_only_r2": score_for(decomp, "occupancy_only"),
        "topology_only_r2": score_for(decomp, "topology_only"),
        "occupancy_plus_topology_r2": score_for(decomp, "occupancy_plus_topology"),
        "occupancy_plus_model_r2": score_for(decomp, "occupancy_plus_model"),
        "occupancy_plus_topology_plus_model_r2": score_for(decomp, "occupancy_plus_topology_plus_model"),
        "delta_topology_after_occupancy": score_for(decomp, "occupancy_plus_topology") - score_for(decomp, "occupancy_only"),
        "topology_cluster_count": int(clusters["topology_cluster"].nunique()) if not clusters.empty else 0,
        "mediation": mediation,
        "outcome": outcome,
        "outcome_text": outcome_text,
        "caveat": "Topology features are proxies derived from aggregate HDDT outputs; raw posterior arrays and full reachability curves were unavailable. The strongest reachability features, including largest_drop and effective_drop_count, are derived from the same survival-curve family as cliffiness, so their high R2 should be interpreted as evidence that reachability topology is the target object, not as leakage-free external prediction.",
    }
    summary_path = output_dir / "occupancy_topology_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    decomp_path = output_dir / "topology_variance_decomposition.csv"
    decomp.to_csv(decomp_path, index=False)
    clusters_path = output_dir / "occupancy_topology_clusters.csv"
    clusters.to_csv(clusters_path, index=False)
    report = output_dir / "occupancy_topology_analysis.md"
    report.write_text(
        "\n".join(
            [
                "# Occupancy Topology Analysis",
                "",
                "## Executive Answer",
                executive_answer(summary),
                "",
                "## Feature Availability Caveat",
                summary["caveat"],
                "",
                "## Q1: Does Occupancy Topology Explain Residual Cliffiness?",
                _markdown_table(decomp),
                "",
                "Permutation importance over topology proxy features:",
                _markdown_table(importance.head(20)),
                "",
                "## Q2: Are Model-Family Effects Mediated by Topology?",
                _markdown_table(pd.DataFrame([mediation])),
                "",
                "## Q3: What Topological Structures Correspond to Cliffy Allocators?",
                _markdown_table(contrasts.head(40)),
                "",
                "## Occupancy Topology Atlas",
                _markdown_table(clusters),
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
        plot_umap(manifold, output_dir / "occupancy_topology_umap.png"),
        plot_clusters(manifold, output_dir / "occupancy_topology_clusters.png"),
        plot_vs_cliffiness(df, topology, output_dir / "occupancy_topology_vs_cliffiness.png"),
        plot_jump_spectra(df, topology, output_dir / "reachability_jump_spectra.png"),
        plot_importance(importance, output_dir / "topology_feature_importance.png"),
        summary_path,
        clusters_path,
        decomp_path,
    ]
    return tuple(outputs)


def executive_answer(summary: dict[str, object]) -> str:
    return (
        f"Topology proxies add delta R2={summary['delta_topology_after_occupancy']:.4f} after conventional occupancy metrics. "
        f"Occupancy+model R2={summary['occupancy_plus_model_r2']:.4f}; occupancy+topology+model R2={summary['occupancy_plus_topology_plus_model_r2']:.4f}. "
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
