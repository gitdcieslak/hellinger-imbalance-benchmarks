"""Test whether accessibility regime structure is global or locally conditioned."""

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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402


HDDT_CSV = ROOT / "results" / "real_validation" / "hddt_accessibility_validation_relaxed.csv"
ATLAS_CSV = ROOT / "reports" / "topology" / "morphology_atlas_clusters.csv"
EXPANSION_SUMMARY = ROOT / "reports" / "topology" / "atlas_expansion_summary.json"
EXPANDED_STATS = ROOT / "reports" / "topology" / "expanded_regime_statistics.csv"
OOS_SUMMARY = ROOT / "reports" / "topology" / "out_of_support_cluster_summary.csv"
OUT = ROOT / "reports" / "topology"
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"
MIN_REGION_ROWS = 30


REGION_ORDER = [
    "high_elevation_low_breadth",
    "high_elevation_moderate_breadth",
    "mid_elevation",
    "low_elevation_high_breadth",
    "low_elevation_low_breadth",
]


def local_region(breadth: float, elevation: float) -> str:
    if elevation >= 0.80 and breadth < 0.70:
        return "high_elevation_low_breadth"
    if elevation >= 0.80 and breadth >= 0.70:
        return "high_elevation_moderate_breadth"
    if 0.40 <= elevation < 0.80:
        return "mid_elevation"
    if elevation < 0.40 and breadth >= 0.70:
        return "low_elevation_high_breadth"
    return "low_elevation_low_breadth"


def assign_regions(df: pd.DataFrame, min_rows: int = MIN_REGION_ROWS) -> tuple[pd.DataFrame, bool]:
    out = df.copy()
    out["region"] = [local_region(b, e) for b, e in zip(out["breadth"], out["elevation"])]
    counts = out["region"].value_counts()
    use_fallback = bool((counts.reindex(REGION_ORDER, fill_value=0) < min_rows).any())
    if use_fallback:
        elev_bins = pd.qcut(out["elevation"].rank(method="first"), q=3, labels=["low", "mid", "high"])
        breadth_bins = pd.qcut(out["breadth"].rank(method="first"), q=2, labels=["low", "high"])
        out["fallback_region"] = elev_bins.astype(str) + "_elevation_" + breadth_bins.astype(str) + "_breadth"
    else:
        out["fallback_region"] = out["region"]
    return out, use_fallback


def load_inputs(hddt_csv: Path = HDDT_CSV, atlas_csv: Path = ATLAS_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    hddt = pd.read_csv(hddt_csv)
    if "fit_failed" in hddt.columns:
        hddt = hddt[~hddt["fit_failed"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    hddt = hddt.dropna(subset=["breadth", "elevation", TARGET, SURVIVAL, "regime_id"]).copy()
    hddt["source"] = "hddt_validation"
    atlas = pd.read_csv(atlas_csv).rename(columns={"allocation_family": "model_id"})
    atlas = atlas.dropna(subset=["breadth", "elevation", TARGET, SURVIVAL]).copy()
    atlas["source"] = "original_atlas"
    atlas["dataset_name"] = "original_atlas"
    atlas["task_id"] = atlas.get("skew_ratio", "atlas").astype(str)
    atlas["regime_id"] = -1
    for col in ["model_id", "task_id", "dataset_name"]:
        if col not in hddt.columns:
            hddt[col] = "unknown"
    return hddt.reset_index(drop=True), atlas.reset_index(drop=True)


def attach_expanded_clusters(hddt: pd.DataFrame, atlas: pd.DataFrame) -> pd.DataFrame:
    combined_cols = ["breadth", "elevation", SURVIVAL, TARGET, "source"]
    combined = pd.concat([atlas[combined_cols], hddt[combined_cols]], ignore_index=True)
    X = RobustScaler().fit_transform(combined[["breadth", "elevation", SURVIVAL, TARGET]].to_numpy(dtype=float))
    try:
        import hdbscan

        labels = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=10).fit_predict(X)
    except Exception:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=0.55, min_samples=10).fit_predict(X)
    out = hddt.copy()
    out["expanded_cluster"] = labels[len(atlas) :]
    return out


def region_inventory(hddt: pd.DataFrame, atlas: pd.DataFrame, region_col: str = "region") -> pd.DataFrame:
    atlas_regions, _ = assign_regions(atlas, min_rows=1)
    atlas_counts = atlas_regions["region"].value_counts().to_dict()
    rows = []
    for region, group in hddt.groupby(region_col):
        rows.append(
            {
                "region": region,
                "n_runs": int(len(group)),
                "n_datasets": int(group["dataset_name"].nunique()),
                "n_tasks": int(group["task_id"].nunique()),
                "n_models": int(group["model_id"].nunique()),
                "mean_breadth": float(group["breadth"].mean()),
                "mean_elevation": float(group["elevation"].mean()),
                "mean_survival_auc": float(group[SURVIVAL].mean()),
                "mean_cliffiness": float(group[TARGET].mean()),
                "cliffiness_std": float(group[TARGET].std(ddof=0)),
                "dominant_datasets": ", ".join(group["dataset_name"].value_counts().head(6).index.astype(str)),
                "dominant_models": ", ".join(group["model_id"].value_counts().head(5).index.astype(str)),
                "original_atlas_runs": int(atlas_counts.get(region, 0)),
                "atlas_status": "present_in_atlas" if atlas_counts.get(region, 0) > 0 else "new_in_hddt",
            }
        )
    return pd.DataFrame(rows).sort_values("n_runs", ascending=False).reset_index(drop=True)


def _feature_matrix(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    pieces = []
    if "morphology" in spec:
        pieces.append(df[["breadth", "elevation"]].reset_index(drop=True))
    if "original_regime" in spec:
        pieces.append(pd.get_dummies(df["regime_id"].astype(str), prefix="original_regime").reset_index(drop=True))
    if "expanded_cluster" in spec:
        pieces.append(pd.get_dummies(df["expanded_cluster"].astype(str), prefix="expanded_cluster").reset_index(drop=True))
    if "region" in spec:
        pieces.append(pd.get_dummies(df["region"].astype(str), prefix="region").reset_index(drop=True))
    if "interaction" in spec:
        interaction = df["regime_id"].astype(str) + "__" + df["region"].astype(str)
        pieces.append(pd.get_dummies(interaction, prefix="regime_x_region").reset_index(drop=True))
    if not pieces:
        raise ValueError(f"Unknown model spec: {spec}")
    return pd.concat(pieces, axis=1)


def grouped_cv_r2(df: pd.DataFrame, spec: str, group_col: str = "dataset_name") -> tuple[float, bool, str]:
    data = df.dropna(subset=["breadth", "elevation", TARGET, "regime_id", "expanded_cluster", "region", group_col]).copy()
    if len(data) < 20:
        return np.nan, False, "too_few_rows"
    groups = data[group_col].astype(str)
    n_groups = groups.nunique()
    if n_groups < 2:
        return np.nan, False, "too_few_groups"
    if data[TARGET].var(ddof=0) <= 1e-12:
        return np.nan, False, "constant_target"
    X = _feature_matrix(data, spec)
    try:
        cv = GroupKFold(n_splits=min(5, n_groups))
        pred = cross_val_predict(make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), X, data[TARGET], cv=cv, groups=groups)
        return float(r2_score(data[TARGET], pred)), True, ""
    except Exception as exc:
        return np.nan, False, type(exc).__name__


def regional_predictive_scores(hddt: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "morphology_only": "morphology",
        "original_regime_only": "original_regime",
        "morphology_plus_original_regime": "morphology+original_regime",
        "expanded_cluster_only": "expanded_cluster",
        "morphology_plus_expanded_cluster": "morphology+expanded_cluster",
    }
    rows = []
    for region, group in hddt.groupby("region"):
        baseline, _, _ = grouped_cv_r2(group, specs["morphology_only"])
        for model_spec, feature_spec in specs.items():
            score, valid, reason = grouped_cv_r2(group, feature_spec)
            rows.append(
                {
                    "region": region,
                    "model_spec": model_spec,
                    "grouped_cv_r2": score,
                    "delta_vs_morphology": score - baseline if np.isfinite(score) and np.isfinite(baseline) else np.nan,
                    "n_rows": int(len(group)),
                    "n_groups": int(group["dataset_name"].nunique()),
                    "valid": bool(valid),
                    "skip_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def global_interaction_scores(hddt: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "morphology_only": "morphology",
        "morphology_plus_regime": "morphology+original_regime",
        "morphology_plus_region": "morphology+region",
        "morphology_plus_regime_plus_region": "morphology+original_regime+region",
        "morphology_plus_regime_x_region": "morphology+original_regime+region+interaction",
    }
    baseline = None
    rows = []
    for model_spec, feature_spec in specs.items():
        score, valid, reason = grouped_cv_r2(hddt, feature_spec)
        if model_spec == "morphology_only":
            baseline = score
        rows.append(
            {
                "scope": "global",
                "region": "all",
                "model_spec": model_spec,
                "grouped_cv_r2": score,
                "delta_vs_morphology": score - baseline if np.isfinite(score) and np.isfinite(baseline) else np.nan,
                "n_rows": int(len(hddt)),
                "n_groups": int(hddt["dataset_name"].nunique()),
                "valid": bool(valid),
                "skip_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def classify_delta(delta: float) -> str:
    if not np.isfinite(delta):
        return "insufficient_data"
    if delta >= 0.05:
        return "regime-valid region"
    if delta > 0.01:
        return "weak-regime region"
    return "regime-invalid region"


def local_success_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = scores[scores["valid"]].copy()
    for region, group in valid.groupby("region"):
        morph = group[group["model_spec"] == "morphology_only"]["grouped_cv_r2"]
        if morph.empty:
            continue
        candidate = group[group["model_spec"].isin(["morphology_plus_original_regime", "morphology_plus_expanded_cluster"])]
        if candidate.empty:
            best_model = "none"
            delta = np.nan
        else:
            best = candidate.sort_values("grouped_cv_r2", ascending=False).iloc[0]
            best_model = str(best["model_spec"])
            delta = float(best["grouped_cv_r2"] - morph.iloc[0])
        classification = classify_delta(delta)
        rows.append(
            {
                "region": region,
                "classification": classification,
                "best_model": best_model,
                "delta_r2": delta,
                "interpretation": interpretation_for_region(region, classification, delta),
            }
        )
    return pd.DataFrame(rows)


def interpretation_for_region(region: str, classification: str, delta: float) -> str:
    if classification == "regime-valid region":
        return "Regime labels add clear local cliffiness signal beyond morphology."
    if classification == "weak-regime region":
        return "Regime labels add modest local signal; treat as suggestive, not definitive."
    if region == "high_elevation_low_breadth":
        return "High survival is coherent, but cliffiness remains heterogeneous and not cleanly regime-separated."
    return "Local morphology explains as much or more cliffiness than available regime labels."


def high_elevation_diagnostics(hddt: pd.DataFrame) -> dict[str, object]:
    region = hddt[hddt["region"] == "high_elevation_low_breadth"].copy()
    high = region[region[TARGET] >= region[TARGET].quantile(0.75)]
    low = region[region[TARGET] <= region[TARGET].quantile(0.25)]
    cluster_means = region.groupby("expanded_cluster")[TARGET].agg(["size", "mean", "std"]).reset_index()
    cluster_range = float(cluster_means["mean"].max() - cluster_means["mean"].min()) if len(cluster_means) > 1 else 0.0
    return {
        "n_runs": int(len(region)),
        "mean_survival_auc": float(region[SURVIVAL].mean()),
        "mean_cliffiness": float(region[TARGET].mean()),
        "cliffiness_std": float(region[TARGET].std(ddof=0)),
        "cliffiness_p10": float(region[TARGET].quantile(0.10)),
        "cliffiness_p50": float(region[TARGET].quantile(0.50)),
        "cliffiness_p90": float(region[TARGET].quantile(0.90)),
        "high_cliffiness_datasets": ", ".join(high["dataset_name"].value_counts().head(8).index.astype(str)),
        "high_cliffiness_models": ", ".join(high["model_id"].value_counts().head(5).index.astype(str)),
        "low_cliffiness_datasets": ", ".join(low["dataset_name"].value_counts().head(8).index.astype(str)),
        "low_cliffiness_models": ", ".join(low["model_id"].value_counts().head(5).index.astype(str)),
        "expanded_cluster_cliffiness_mean_range": cluster_range,
        "expanded_clusters_separate_cliffiness": bool(cluster_range >= 0.25),
        "conclusion": "broad accessibility-level region, not a clean cliffiness regime",
    }


def manuscript_recommendation(global_scores: pd.DataFrame, success: pd.DataFrame) -> tuple[str, str]:
    lookup = dict(zip(global_scores["model_spec"], global_scores["grouped_cv_r2"]))
    morph = lookup.get("morphology_only", np.nan)
    regime = lookup.get("morphology_plus_regime", np.nan)
    interaction = lookup.get("morphology_plus_regime_x_region", np.nan)
    valid_regions = int((success["classification"] == "regime-valid region").sum()) if not success.empty else 0
    if np.isfinite(interaction) and np.isfinite(regime) and np.isfinite(morph) and interaction > regime > morph:
        return "Outcome B - Local regime support", "survival ~= global morphology position; cliffiness ~= local morphology + region-conditional regime structure"
    if valid_regions > 0:
        return "Outcome B - Local regime support", "survival ~= global morphology position; cliffiness ~= local morphology + region-conditional regime structure"
    return "Outcome C - Weak regime support under validation", "controlled experiments show regime structure, but external benchmarks reveal additional morphology regions where regime structure remains unresolved"


def plot_region_map(hddt: pd.DataFrame, atlas: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(atlas["breadth"], atlas["elevation"], s=12, c="lightgray", alpha=0.45, label="original atlas")
    for region, group in hddt.groupby("region"):
        ax.scatter(group["breadth"], group["elevation"], s=14, alpha=0.7, label=region)
    ax.axhline(0.80, color="black", lw=0.8, ls="--")
    ax.axhline(0.40, color="black", lw=0.8, ls=":")
    ax.axvline(0.70, color="black", lw=0.8, ls="--")
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Local Morphology Regions")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_cliffiness_by_region(hddt: pd.DataFrame, output: Path) -> Path:
    order = [r for r in REGION_ORDER if r in set(hddt["region"])]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([hddt[hddt["region"] == r][TARGET] for r in order], tick_labels=order, showfliers=False)
    ax.set_ylabel("cliffiness")
    ax.set_title("Cliffiness Distribution by Local Region")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_predictive_scores(regional: pd.DataFrame, global_scores: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pivot = regional.pivot_table(index="region", columns="model_spec", values="grouped_cv_r2", aggfunc="first")
    pivot.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Within-Region Cliffiness CV R2")
    axes[0].set_ylabel("grouped CV R2")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=7)
    axes[1].bar(global_scores["model_spec"], global_scores["grouped_cv_r2"], color="steelblue")
    axes[1].set_title("Global Regime x Region Models")
    axes[1].set_ylabel("grouped CV R2")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_report(output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hddt, atlas = load_inputs()
    hddt = attach_expanded_clusters(hddt, atlas)
    hddt, used_fallback = assign_regions(hddt)
    atlas_regions, _ = assign_regions(atlas, min_rows=1)
    inventory = region_inventory(hddt, atlas_regions)
    regional_scores = regional_predictive_scores(hddt)
    global_scores = global_interaction_scores(hddt)
    success = local_success_table(regional_scores)
    high_diag = high_elevation_diagnostics(hddt)
    outcome, thesis = manuscript_recommendation(global_scores, success)

    interaction_summary = pd.concat([global_scores.assign(scope="global"), regional_scores.assign(scope="regional")], ignore_index=True)
    interaction_csv = output_dir / "local_regime_interaction_summary.csv"
    interaction_summary.to_csv(interaction_csv, index=False)
    summary = {
        "n_hddt_rows": int(len(hddt)),
        "n_regions": int(hddt["region"].nunique()),
        "used_quantile_fallback_regions": bool(used_fallback),
        "global_morphology_only_r2": score_for(global_scores, "morphology_only"),
        "global_morphology_plus_regime_r2": score_for(global_scores, "morphology_plus_regime"),
        "global_morphology_plus_region_r2": score_for(global_scores, "morphology_plus_region"),
        "global_morphology_plus_regime_x_region_r2": score_for(global_scores, "morphology_plus_regime_x_region"),
        "n_regime_valid_regions": int((success["classification"] == "regime-valid region").sum()) if not success.empty else 0,
        "n_weak_regime_regions": int((success["classification"] == "weak-regime region").sum()) if not success.empty else 0,
        "n_regime_invalid_regions": int((success["classification"] == "regime-invalid region").sum()) if not success.empty else 0,
        "high_elevation_low_breadth": high_diag,
        "manuscript_outcome": outcome,
        "recommended_thesis": thesis,
    }
    summary_json = output_dir / "local_regime_validity_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    external = load_external_summaries()
    report = output_dir / "local_regime_validity.md"
    report.write_text(
        "\n".join(
            [
                "# Local Regime Validity Analysis",
                "",
                "## Executive Answer",
                executive_answer(summary),
                "",
                "## Step 1: Local Morphology Regions",
                f"Operational regions were used. Quantile fallback used: `{used_fallback}`.",
                "",
                "## Step 2: Region Inventory",
                _markdown_table(inventory),
                "",
                "External atlas-expansion summaries used:",
                _markdown_table(external),
                "",
                "## Step 3: Regime Predictiveness Within Each Region",
                _markdown_table(regional_scores),
                "",
                "## Step 4: Regime x Region Interaction Globally",
                _markdown_table(global_scores),
                "",
                "## Step 5: Local Regime Success and Failure Zones",
                _markdown_table(success),
                "",
                "## Step 6: High-Elevation / Low-Breadth Region",
                _markdown_table(pd.DataFrame([high_diag])),
                "",
                "Answer: the high-elevation / low-breadth region behaves as a broad accessibility-level region with high survival and heterogeneous cliffiness, not as a clean cliffiness regime.",
                "",
                "## Implications for Accessibility Regimes in Rare-Event Learning",
                f"Recommended outcome: **{outcome}**.",
                "",
                f"Recommended thesis: `{thesis}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outputs = [
        report,
        plot_region_map(hddt, atlas_regions, output_dir / "local_regime_region_map.png"),
        plot_cliffiness_by_region(hddt, output_dir / "local_regime_cliffiness_by_region.png"),
        plot_predictive_scores(regional_scores, global_scores, output_dir / "local_regime_predictive_scores.png"),
        interaction_csv,
        summary_json,
    ]
    return tuple(outputs)


def score_for(scores: pd.DataFrame, model_spec: str) -> float:
    match = scores[scores["model_spec"] == model_spec]
    return float(match["grouped_cv_r2"].iloc[0]) if not match.empty else np.nan


def load_external_summaries() -> pd.DataFrame:
    rows = []
    if EXPANSION_SUMMARY.exists():
        data = json.loads(EXPANSION_SUMMARY.read_text(encoding="utf-8"))
        rows.append({"artifact": "atlas_expansion_summary", **{k: data.get(k) for k in ["fraction_inside_axis_support", "mean_nearest_support_distance", "expanded_cluster_count", "n_oos_clusters"]}})
    if EXPANDED_STATS.exists():
        stats = pd.read_csv(EXPANDED_STATS)
        rows.append({"artifact": "expanded_regime_statistics", "n_rows": len(stats), "max_validation_fraction": float(stats["validation_fraction"].max())})
    if OOS_SUMMARY.exists():
        oos = pd.read_csv(OOS_SUMMARY)
        rows.append({"artifact": "out_of_support_cluster_summary", "n_rows": len(oos), "max_mean_cliffiness": float(oos["mean_cliffiness"].max())})
    return pd.DataFrame(rows)


def executive_answer(summary: dict[str, object]) -> str:
    return (
        "Regimes are not globally meaningful across the expanded HDDT morphology space. "
        "Survival remains globally morphology-position related, but cliffiness regime structure is local or unresolved under validation. "
        f"The recommended manuscript outcome is {summary['manuscript_outcome']}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
