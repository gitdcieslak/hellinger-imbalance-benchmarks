"""Test whether allocator capacity governs accessibility cliffiness."""

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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_accessibility_morphology_sufficiency import CLASSICAL_CURVES, HDDT_CURVES, load_dataset  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402
from report_posterior_concentration_hypothesis import score_distribution_from_curve  # noqa: E402


OUT = ROOT / "reports" / "topology"
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"
CAPACITY_COLS = [
    "effective_support",
    "inverse_effective_support",
    "positive_score_entropy",
    "positive_score_gini",
    "joined_positive_unique_score_ratio",
    "joined_positive_effective_score_bins",
    "joined_positive_histogram_entropy",
    "top_1pct_mass",
    "top_5pct_mass",
    "top_10pct_mass",
    "top10_to_top1_ratio",
    "tail_mass_above_99pct",
    "density_around_090",
    "density_around_095",
    "density_around_099",
]


def available(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]


def load_capacity_dataset(curve_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    meta, morphology, _, thresholds = load_dataset(curve_paths)
    caps = morphology[available(CAPACITY_COLS, morphology)].copy()
    if "inverse_effective_support" not in caps.columns and "effective_support" in caps.columns:
        caps["inverse_effective_support"] = 1.0 / caps["effective_support"].replace(0, np.nan)
    return meta, morphology, caps.replace([np.inf, -np.inf], np.nan), thresholds


def _model_specs() -> dict[str, object]:
    return {
        "ridge": make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=350, min_samples_leaf=5, random_state=19, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=19, max_depth=3, learning_rate=0.04, n_estimators=250),
    }


def grouped_model_scores(meta: pd.DataFrame, features: pd.DataFrame, prefix: str, group_col: str = "dataset_id") -> pd.DataFrame:
    cols = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c]) and features[c].notna().any()]
    data = pd.concat([meta[[TARGET, group_col]].reset_index(drop=True), features[cols].reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    if len(data) < 20 or data[group_col].nunique() < 2:
        return pd.DataFrame([{"model_spec": f"{prefix}_unavailable", "grouped_cv_r2": np.nan, "grouped_cv_mae": np.nan, "n_features": len(cols), "valid": False}])
    cv = GroupKFold(n_splits=min(5, data[group_col].nunique()))
    X = data.drop(columns=[TARGET, group_col])
    y = data[TARGET]
    groups = data[group_col].astype(str)
    for name, model in _model_specs().items():
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        rows.append({"model_spec": f"{prefix}_{name}", "grouped_cv_r2": float(r2_score(y, pred)), "grouped_cv_mae": float(mean_absolute_error(y, pred)), "n_features": len(cols), "valid": True})
    return pd.DataFrame(rows)


def capacity_relationship(meta: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    return grouped_model_scores(meta, capacity, "capacity")


def family_conditioning(meta: pd.DataFrame, capacity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = pd.get_dummies(meta["model_id"].astype(str), prefix="family").reset_index(drop=True)
    family_scores = grouped_model_scores(meta, family, "family")
    combined = pd.concat([capacity.reset_index(drop=True), family], axis=1)
    combined_scores = grouped_model_scores(meta, combined, "capacity_plus_family")
    scores = pd.concat([family_scores, grouped_model_scores(meta, capacity, "capacity"), combined_scores], ignore_index=True)
    importance = block_importance(meta, capacity, family, combined)
    return scores, importance


def block_importance(meta: pd.DataFrame, capacity: pd.DataFrame, family: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([meta[[TARGET]].reset_index(drop=True), combined.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 30:
        return pd.DataFrame()
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)
    model = RandomForestRegressor(n_estimators=350, min_samples_leaf=5, random_state=21, n_jobs=-1).fit(X_train, y_train)
    base = float(r2_score(y_test, model.predict(X_test)))
    rows = []
    rng = np.random.default_rng(21)
    for block, cols in [("capacity", list(capacity.columns)), ("family", list(family.columns))]:
        drops = []
        cols = [c for c in cols if c in X_test.columns]
        for _ in range(20):
            shuffled = X_test.copy()
            for col in cols:
                shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(base - float(r2_score(y_test, model.predict(shuffled))))
        rows.append({"feature_block": block, "base_r2": base, "mean_r2_drop": float(np.mean(drops)), "fraction_of_explained_variance": float(max(0.0, np.mean(drops)) / max(1e-9, base))})
    return pd.DataFrame(rows)


def capacity_binning(meta: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([meta.reset_index(drop=True), capacity.reset_index(drop=True)], axis=1)
    data = data.dropna(subset=["effective_support", TARGET, SURVIVAL, "breadth", "elevation"])
    data["capacity_decile"] = pd.qcut(data["effective_support"].rank(method="first"), 10, labels=False) + 1
    return data.groupby("capacity_decile", as_index=False).agg(
        n_runs=(TARGET, "size"),
        mean_effective_support=("effective_support", "mean"),
        mean_cliffiness=(TARGET, "mean"),
        mean_survival_auc=(SURVIVAL, "mean"),
        mean_elevation=("elevation", "mean"),
        mean_breadth=("breadth", "mean"),
    )


def metrics_from_distribution(scores: np.ndarray, weights: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(1e-12, weights.sum())
    curve = np.asarray([weights[scores >= t].sum() for t in thresholds], dtype=float)
    drops = np.maximum(0.0, curve[:-1] - curve[1:])
    total_drop = float(drops.sum())
    max_drop = float(drops.max()) if drops.size else 0.0
    cliff = max_drop / total_drop if total_drop > 0 else 0.0
    auc = float(np.trapezoid(curve, thresholds) / max(1e-12, thresholds[-1] - thresholds[0])) if thresholds.size > 1 else float(curve.mean())
    eff_support = float(1.0 / max(1e-12, np.sum(weights**2)))
    entropy = float(-(weights[weights > 0] * np.log(weights[weights > 0])).sum())
    return {"effective_support": eff_support, "positive_score_entropy": entropy, "minority_survival_auc": auc, "minority_survival_cliffiness": cliff, "persistence": float(curve.mean())}


def collapse_distribution(scores: np.ndarray, weights: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    n = max(1, int(np.ceil(len(scores) * fraction)))
    order = np.argsort(scores)
    s = scores[order]
    w = weights[order]
    groups = np.array_split(np.arange(len(s)), n)
    new_scores = []
    new_weights = []
    for idx in groups:
        ww = w[idx]
        new_weights.append(float(ww.sum()))
        new_scores.append(float(np.average(s[idx], weights=ww)) if ww.sum() > 0 else float(s[idx].mean()))
    return np.asarray(new_scores), np.asarray(new_weights)


def expand_distribution(scores: np.ndarray, weights: np.ndarray, factor: int = 3, width: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.linspace(-width, width, factor)
    new_scores = []
    new_weights = []
    for score, weight in zip(scores, weights, strict=False):
        for offset in offsets:
            new_scores.append(float(np.clip(score + offset, 0.0, 1.0)))
            new_weights.append(float(weight / factor))
    return np.asarray(new_scores), np.asarray(new_weights)


def capacity_interventions(meta: pd.DataFrame, curve_matrix: pd.DataFrame, thresholds: np.ndarray, max_runs: int = 500) -> pd.DataFrame:
    rows = []
    sample_idx = meta.sample(min(max_runs, len(meta)), random_state=23).index if len(meta) > max_runs else meta.index
    for idx in sample_idx:
        scores, weights = score_distribution_from_curve(curve_matrix.iloc[idx].to_numpy(dtype=float), thresholds)
        base = metrics_from_distribution(scores, weights, thresholds)
        for fraction in [1.0, 0.75, 0.50, 0.25, 0.10]:
            cs, cw = collapse_distribution(scores, weights, fraction)
            metric = metrics_from_distribution(cs, cw, thresholds)
            rows.append({"run_id": meta.loc[idx, "run_id"], "model_id": meta.loc[idx, "model_id"], "intervention": "collapse", "support_fraction": fraction, "baseline_effective_support": base["effective_support"], **metric})
        for width in [0.0, 0.005, 0.01, 0.02, 0.04]:
            es, ew = expand_distribution(scores, weights, factor=5, width=width)
            metric = metrics_from_distribution(es, ew, thresholds)
            rows.append({"run_id": meta.loc[idx, "run_id"], "model_id": meta.loc[idx, "model_id"], "intervention": "expansion", "jitter_width": width, "baseline_effective_support": base["effective_support"], **metric})
    return pd.DataFrame(rows)


def permutation_importance_table(meta: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    cols = available(CAPACITY_COLS, capacity)
    data = pd.concat([meta[[TARGET]].reset_index(drop=True), capacity[cols].reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 50:
        return pd.DataFrame()
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=29)
    model = RandomForestRegressor(n_estimators=350, min_samples_leaf=5, random_state=29, n_jobs=-1).fit(X_train, y_train)
    imp = permutation_importance(model, X_test, y_test, n_repeats=12, random_state=29, n_jobs=-1)
    return pd.DataFrame({"feature": X.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std}).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def decision(cap_scores: pd.DataFrame, family_importance: pd.DataFrame) -> tuple[str, str]:
    best_r2 = float(cap_scores["grouped_cv_r2"].max()) if not cap_scores.empty else np.nan
    family_frac = np.nan
    if not family_importance.empty and (family_importance["feature_block"] == "family").any():
        family_frac = float(family_importance.loc[family_importance["feature_block"] == "family", "fraction_of_explained_variance"].iloc[0])
    if np.isfinite(best_r2) and best_r2 >= 0.80 and (not np.isfinite(family_frac) or family_frac < 0.10):
        return "supported", "Capacity metrics strongly explain cliffiness and family contribution is small after conditioning on capacity."
    if np.isfinite(best_r2) and best_r2 >= 0.80:
        return "partially_supported", "Capacity metrics strongly explain cliffiness, but residual family contribution remains non-trivial."
    return "not_supported", "Capacity metrics do not meet the predefined R2 threshold for explaining cliffiness."


def plot_scatter(meta: pd.DataFrame, capacity: pd.DataFrame, x: str, y: str, output: Path) -> Path:
    data = pd.concat([meta.reset_index(drop=True), capacity.reset_index(drop=True)], axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    codes, _ = pd.factorize(data["model_id"].astype(str))
    ax.scatter(data[x], data[y], c=codes, cmap="tab20", s=14, alpha=0.65)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_family_distributions(meta: pd.DataFrame, capacity: pd.DataFrame, output: Path) -> Path:
    data = pd.concat([meta[["model_id"]].reset_index(drop=True), capacity[["effective_support"]].reset_index(drop=True)], axis=1).dropna()
    order = data.groupby("model_id")["effective_support"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([data.loc[data["model_id"] == m, "effective_support"] for m in order], tick_labels=order, showfliers=False)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("effective_support")
    ax.set_title("Family Occupancy By Capacity")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_intervention(results: pd.DataFrame, intervention: str, output: Path) -> Path:
    data = results[results["intervention"] == intervention].copy()
    x = "support_fraction" if intervention == "collapse" else "jitter_width"
    summary = data.groupby(x, as_index=False).agg(mean_support=("effective_support", "mean"), mean_cliffiness=(TARGET, "mean"), mean_auc=(SURVIVAL, "mean"))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["mean_support"], summary["mean_cliffiness"], marker="o", label="cliffiness")
    ax.set_xlabel("mean effective support")
    ax.set_ylabel("mean cliffiness")
    ax2 = ax.twinx()
    ax2.plot(summary["mean_support"], summary["mean_auc"], marker="s", color="tab:green", label="survival AUC")
    ax2.set_ylabel("mean survival AUC")
    ax.set_title(f"Capacity {intervention.title()} Intervention")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_report(curve_paths: list[Path], output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, morphology, capacity, thresholds = load_capacity_dataset(curve_paths)
    from report_definitive_reachability_topology import load_curve_files, pivot_curves

    _, matrix, _ = pivot_curves(load_curve_files(curve_paths))
    cap_scores = capacity_relationship(meta, capacity)
    family_scores, family_importance = family_conditioning(meta, capacity)
    bins = capacity_binning(meta, capacity)
    interventions = capacity_interventions(meta, matrix, thresholds)
    importance = permutation_importance_table(meta, capacity)
    outcome, outcome_text = decision(cap_scores, family_importance)
    summary = {
        "n_runs": int(len(meta)),
        "n_capacity_features": int(len(capacity.columns)),
        "best_capacity_r2": float(cap_scores["grouped_cv_r2"].max()),
        "best_capacity_mae": float(cap_scores.loc[cap_scores["grouped_cv_r2"].idxmax(), "grouped_cv_mae"]),
        "family_fraction_after_capacity": float(family_importance.loc[family_importance["feature_block"] == "family", "fraction_of_explained_variance"].iloc[0]) if not family_importance.empty and (family_importance["feature_block"] == "family").any() else np.nan,
        "outcome": outcome,
        "outcome_text": outcome_text,
    }
    report = output_dir / "accessibility_capacity_hypothesis.md"
    summary_json = output_dir / "accessibility_capacity_summary.json"
    family_path = output_dir / "capacity_family_effects.csv"
    frontier_path = output_dir / "capacity_frontier.csv"
    intervention_path = output_dir / "capacity_intervention_results.csv"
    cap_scores_path = output_dir / "capacity_model_scores.csv"
    importance_path = output_dir / "capacity_feature_importance.csv"
    pd.concat([family_scores.assign(table="model_scores"), family_importance.assign(table="block_importance")], ignore_index=True, sort=False).to_csv(family_path, index=False)
    pd.concat([meta.reset_index(drop=True), capacity.reset_index(drop=True)], axis=1).to_csv(frontier_path, index=False)
    interventions.to_csv(intervention_path, index=False)
    cap_scores.to_csv(cap_scores_path, index=False)
    importance.to_csv(importance_path, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text("\n".join([
        "# Accessibility Capacity Hypothesis",
        "",
        "## Executive Answer",
        outcome_text,
        "",
        "## Experiment A: Capacity vs Cliffiness",
        _markdown_table(cap_scores),
        "",
        "## Capacity Feature Importance",
        _markdown_table(importance.head(20)),
        "",
        "## Experiment B: Family Conditioned Capacity",
        _markdown_table(family_scores),
        "",
        "## Family/Capacity Block Importance",
        _markdown_table(family_importance),
        "",
        "## Experiment C: Capacity Binning",
        _markdown_table(bins),
        "",
        "## Experiments D/E: Artificial Capacity Interventions",
        _markdown_table(interventions.groupby(["intervention"], as_index=False).agg(n_rows=("run_id", "size"), mean_effective_support=("effective_support", "mean"), mean_cliffiness=(TARGET, "mean"), mean_survival_auc=(SURVIVAL, "mean"))),
        "",
        "## Interpretation",
        "Success requires capacity-only `R^2 >= 0.80`; family effects should contribute `< 10%` of explained variance after capacity is included.",
    ]) + "\n", encoding="utf-8")
    return (
        report,
        summary_json,
        family_path,
        frontier_path,
        intervention_path,
        cap_scores_path,
        importance_path,
        plot_scatter(meta, capacity, "effective_support", TARGET, output_dir / "capacity_vs_cliffiness.png"),
        plot_scatter(meta, capacity, "effective_support", SURVIVAL, output_dir / "capacity_vs_survival_auc.png"),
        plot_scatter(meta, capacity, "effective_support", "elevation", output_dir / "capacity_vs_elevation.png"),
        plot_family_distributions(meta, capacity, output_dir / "capacity_family_distributions.png"),
        plot_intervention(interventions, "collapse", output_dir / "capacity_collapse_intervention.png"),
        plot_intervention(interventions, "expansion", output_dir / "capacity_expansion_intervention.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", action="append", type=Path, dest="curves", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    curves = args.curves or [HDDT_CURVES, CLASSICAL_CURVES]
    for output in write_report(curves, args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
