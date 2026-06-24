"""Test whether accessibility cliffiness is governed by collapse events."""

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
from sklearn.ensemble import RandomForestRegressor
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

from report_accessibility_morphology_decomposition import CAPACITY_FEATURES, load_decomposition_dataset, score_for  # noqa: E402
from report_accessibility_morphology_sufficiency import CLASSICAL_CURVES, HDDT_CURVES  # noqa: E402
from report_definitive_reachability_topology import load_curve_files, pivot_curves  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402


OUT = ROOT / "reports" / "topology"
TARGET = "minority_survival_cliffiness"


def entropy(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0 or v.sum() <= 0:
        return 0.0
    p = v / v.sum()
    return float(-(p * np.log(p)).sum())


def gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size == 0 or v.sum() <= 0:
        return 0.0
    n = v.size
    return float((2 * np.arange(1, n + 1) @ v) / (n * v.sum()) - (n + 1) / n)


def event_features_from_curves(matrix: pd.DataFrame, thresholds: np.ndarray, min_event_fraction: float = 0.01) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows = []
    t = np.asarray(thresholds, dtype=float)
    for _, row in matrix.iterrows():
        curve = row.to_numpy(dtype=float)
        drops = np.maximum(0.0, curve[:-1] - curve[1:])
        total = float(drops.sum())
        event_mask = drops >= max(1e-12, min_event_fraction * max(1.0, total))
        event_mass = drops[event_mask]
        event_t = t[1:][event_mask] if len(t) == len(curve) else np.linspace(0, 1, len(drops))[event_mask]
        if event_mass.size == 0:
            event_mass = np.asarray([0.0])
            event_t = np.asarray([0.0])
        probs = event_mass / max(1e-12, event_mass.sum())
        sorted_mass = np.sort(event_mass)[::-1]
        top = lambda k: float(sorted_mass[: min(k, sorted_mass.size)].sum() / max(1e-12, event_mass.sum()))
        gaps = np.diff(np.sort(event_t[np.isfinite(event_t)]))
        weighted_center = float(np.sum(event_t * probs)) if np.isfinite(event_t).all() else np.nan
        weighted_var = float(np.sum(probs * (event_t - weighted_center) ** 2)) if np.isfinite(weighted_center) else np.nan
        rows.append(
            {
                "largest_drop": float(sorted_mass[0]) if sorted_mass.size else 0.0,
                "second_largest_drop": float(sorted_mass[1]) if sorted_mass.size > 1 else 0.0,
                "third_largest_drop": float(sorted_mass[2]) if sorted_mass.size > 2 else 0.0,
                "top3_drop_mass": top(3),
                "top5_drop_mass": top(5),
                "drop_mass_entropy": entropy(event_mass),
                "cumulative_drop_mass": float(event_mass.sum()),
                "event_count": int(np.sum(event_mask)),
                "effective_event_count": float(1.0 / max(1e-12, np.sum(probs**2))),
                "event_density": float(np.mean(event_mask)) if event_mask.size else 0.0,
                "first_major_drop_threshold": float(np.nanmin(event_t)),
                "median_drop_threshold": float(np.nanmedian(event_t)),
                "weighted_drop_center": weighted_center,
                "weighted_drop_variance": weighted_var,
                "top1_fraction": top(1),
                "top3_fraction": top(3),
                "top5_fraction": top(5),
                "gini_drop_mass": gini(event_mass),
                "hhi_drop_mass": float(np.sum(probs**2)),
                "mean_gap_between_events": float(gaps.mean()) if gaps.size else 0.0,
                "variance_gap_between_events": float(gaps.var()) if gaps.size else 0.0,
                "burstiness": float(gaps.std() / max(1e-12, gaps.mean())) if gaps.size else 0.0,
                "clustering_coefficient": float(np.mean(gaps <= np.median(gaps))) if gaps.size else 0.0,
                "mass_above_95pct": float(drops[t[1:] >= 0.95].sum() / max(1e-12, total)) if len(t) == len(curve) else 0.0,
                "mass_above_99pct": float(drops[t[1:] >= 0.99].sum() / max(1e-12, total)) if len(t) == len(curve) else 0.0,
                "final_drop_fraction": float(drops[-1] / max(1e-12, total)) if drops.size else 0.0,
                "tail_event_count": int(np.sum(event_mask & (t[1:] >= 0.95))) if len(t) == len(curve) else 0,
            }
        )
    blocks = {
        "magnitude": ["largest_drop", "second_largest_drop", "third_largest_drop", "top3_drop_mass", "top5_drop_mass", "drop_mass_entropy", "cumulative_drop_mass"],
        "count": ["event_count", "effective_event_count", "event_density"],
        "timing": ["first_major_drop_threshold", "median_drop_threshold", "weighted_drop_center", "weighted_drop_variance"],
        "hierarchy": ["top1_fraction", "top3_fraction", "top5_fraction", "gini_drop_mass", "hhi_drop_mass"],
        "spacing": ["mean_gap_between_events", "variance_gap_between_events", "burstiness", "clustering_coefficient"],
        "tail": ["mass_above_95pct", "mass_above_99pct", "final_drop_fraction", "tail_event_count"],
    }
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan), blocks


def grouped_score(meta: pd.DataFrame, features: pd.DataFrame) -> tuple[float, float]:
    data = pd.concat([meta[[TARGET, "dataset_id"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    groups = data["dataset_id"].astype(str)
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    X = data.drop(columns=[TARGET, "dataset_id"])
    y = data[TARGET]
    best_r2, best_mae = -np.inf, np.nan
    for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=71, n_jobs=1)]:
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        score = float(r2_score(y, pred))
        if score > best_r2:
            best_r2 = score
            best_mae = float(mean_absolute_error(y, pred))
    return best_r2, best_mae


def model_scores(meta: pd.DataFrame, capacity: pd.DataFrame, events: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "capacity": capacity,
        "event_structure": events,
        "capacity_plus_event_structure": pd.concat([capacity.add_prefix("capacity__"), events.add_prefix("event__")], axis=1),
        "full_morphology": full,
    }
    rows = []
    for name, frame in specs.items():
        r2, mae = grouped_score(meta, frame)
        rows.append({"model_spec": name, "grouped_cv_r2": r2, "grouped_cv_mae": mae, "n_features": frame.shape[1]})
    scores = pd.DataFrame(rows)
    cap = score_for(scores.assign(target=TARGET), TARGET, "capacity")
    full_r2 = score_for(scores.assign(target=TARGET), TARGET, "full_morphology")
    scores["delta_vs_capacity"] = scores["grouped_cv_r2"] - cap
    scores["gap_to_full_morphology"] = full_r2 - scores["grouped_cv_r2"]
    return scores


def variance_partition(scores: pd.DataFrame) -> pd.DataFrame:
    lookup = dict(zip(scores["model_spec"], scores["grouped_cv_r2"]))
    cap = max(0.0, float(lookup["capacity"]))
    event = max(0.0, float(lookup["event_structure"]))
    both = max(0.0, float(lookup["capacity_plus_event_structure"]))
    return pd.DataFrame([
        {"component": "capacity_unique", "variance_fraction": max(0.0, both - event)},
        {"component": "event_unique", "variance_fraction": max(0.0, both - cap)},
        {"component": "shared", "variance_fraction": max(0.0, min(cap, event))},
        {"component": "unexplained", "variance_fraction": max(0.0, 1.0 - both)},
    ])


def feature_importance(meta: pd.DataFrame, events: pd.DataFrame, blocks: dict[str, list[str]]) -> pd.DataFrame:
    data = pd.concat([meta[[TARGET]].reset_index(drop=True), events.reset_index(drop=True)], axis=1).dropna()
    X = data.drop(columns=[TARGET]); y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=73)
    model = RandomForestRegressor(n_estimators=350, min_samples_leaf=5, random_state=73, n_jobs=1).fit(X_train, y_train)
    imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=73, n_jobs=1)
    out = pd.DataFrame({"feature": X.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std})
    block_rows = [{"feature": f"BLOCK::{name}", "importance_mean": float(out[out["feature"].isin(cols)]["importance_mean"].sum()), "importance_std": np.nan} for name, cols in blocks.items()]
    return pd.concat([pd.DataFrame(block_rows), out], ignore_index=True).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def event_archetypes(meta: pd.DataFrame, events: pd.DataFrame, n_clusters: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    X = StandardScaler().fit_transform(events.fillna(events.median(numeric_only=True)).fillna(0))
    labels = KMeans(n_clusters=min(n_clusters, len(meta)), random_state=75, n_init=20).fit_predict(X)
    data = pd.concat([meta.reset_index(drop=True), events.reset_index(drop=True)], axis=1)
    data["event_archetype"] = labels
    rows = []
    for label, group in data.groupby("event_archetype"):
        if group["top1_fraction"].mean() > 0.75:
            name = "Singular Catastrophe"
        elif group["event_count"].mean() <= 4:
            name = "Staircase Collapse"
        elif group["mass_above_95pct"].mean() > 0.30:
            name = "Tail Collapse"
        elif group["burstiness"].mean() > 1.0:
            name = "Fragmented Collapse"
        else:
            name = "Continuous Erosion"
        rows.append({"event_archetype": int(label), "archetype_name": name, "n_runs": int(len(group)), "mean_cliffiness": float(group[TARGET].mean()), "mean_survival_auc": float(group["minority_survival_auc"].mean()), "dominant_family": str(group["model_id"].value_counts().idxmax())})
    return pd.DataFrame(rows).sort_values("n_runs", ascending=False), labels


def event_frontier(meta: pd.DataFrame, capacity: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([meta.reset_index(drop=True), capacity[["effective_support"]].reset_index(drop=True), events[["top1_fraction"]].reset_index(drop=True)], axis=1).dropna()
    data["capacity_band"] = pd.qcut(data["effective_support"].rank(method="first"), 8, labels=False) + 1
    data["event_concentration_bin"] = pd.qcut(data["top1_fraction"].rank(method="first"), 4, labels=False) + 1
    return data.groupby(["capacity_band", "event_concentration_bin"], as_index=False).agg(n_runs=(TARGET, "size"), mean_effective_support=("effective_support", "mean"), mean_event_concentration=("top1_fraction", "mean"), min_cliffiness=(TARGET, "min"), median_cliffiness=(TARGET, "median"), p90_cliffiness=(TARGET, lambda s: float(s.quantile(.9))), max_cliffiness=(TARGET, "max"))


def plot_archetypes(archetypes: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(archetypes["archetype_name"], archetypes["mean_cliffiness"], color="steelblue")
    ax.set_ylabel("mean cliffiness"); ax.set_title("Event Archetypes"); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_frontier(frontier: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for bin_id, group in frontier.groupby("event_concentration_bin"):
        ax.plot(group["mean_effective_support"], group["p90_cliffiness"], marker="o", label=f"event conc {bin_id}")
    ax.set_xlabel("effective support"); ax.set_ylabel("p90 cliffiness"); ax.set_title("Event Frontier"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_importance(importance: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6)); top = importance.head(15).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], color="steelblue"); ax.set_title("Event Feature Importance")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def write_report(curve_paths: list[Path], output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, capacity, _, full = load_decomposition_dataset(curve_paths)
    _, matrix, thresholds = pivot_curves(load_curve_files(curve_paths))
    events, blocks = event_features_from_curves(matrix, thresholds)
    scores = model_scores(meta, capacity, events, full)
    partition = variance_partition(scores)
    importance = feature_importance(meta, events, blocks)
    archetypes, _ = event_archetypes(meta, events)
    frontier = event_frontier(meta, capacity, events)
    lookup = dict(zip(scores["model_spec"], scores["grouped_cv_r2"]))
    cap, both, full_r2 = lookup["capacity"], lookup["capacity_plus_event_structure"], lookup["full_morphology"]
    event_unique = float(partition[partition["component"] == "event_unique"]["variance_fraction"].iloc[0])
    if both > 0.90 and full_r2 - both < 0.05:
        outcome = "strong_support"
    elif event_unique > 0.15 and both / full_r2 > 0.80:
        outcome = "moderate_support"
    else:
        outcome = "failure"
    summary = {"n_runs": int(len(meta)), "capacity_r2": float(cap), "event_structure_r2": float(lookup["event_structure"]), "capacity_plus_event_r2": float(both), "full_morphology_r2": float(full_r2), "delta_r2_vs_capacity": float(both - cap), "gap_to_full_morphology": float(full_r2 - both), "event_unique": event_unique, "outcome": outcome}
    report = output_dir / "accessibility_event_structure_hypothesis.md"; summary_path = output_dir / "accessibility_event_structure_summary.json"; scores_path = output_dir / "event_structure_model_scores.csv"; part_path = output_dir / "event_structure_variance_partition.csv"; imp_path = output_dir / "event_structure_feature_importance.csv"; arch_path = output_dir / "event_structure_archetypes.csv"; front_path = output_dir / "event_structure_frontier.csv"
    scores.to_csv(scores_path, index=False); partition.to_csv(part_path, index=False); importance.to_csv(imp_path, index=False); archetypes.to_csv(arch_path, index=False); frontier.to_csv(front_path, index=False); summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text("\n".join(["# Accessibility Event Structure Hypothesis", "", "## Executive Answer", f"Outcome `{outcome}`: capacity+event R2 is `{both:.4f}`, delta over capacity is `{both - cap:.4f}`, gap to full morphology is `{full_r2 - both:.4f}`.", "", "## Model Scores", _markdown_table(scores), "", "## Variance Partition", _markdown_table(partition), "", "## Event Feature Importance", _markdown_table(importance.head(25)), "", "## Event Archetypes", _markdown_table(archetypes), "", "## Event Frontier", _markdown_table(frontier.head(30))]) + "\n", encoding="utf-8")
    return (report, summary_path, scores_path, part_path, imp_path, arch_path, front_path, plot_importance(importance, output_dir / "event_structure_feature_importance.png"), plot_archetypes(archetypes, output_dir / "event_structure_archetypes.png"), plot_frontier(frontier, output_dir / "event_structure_frontier.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", action="append", type=Path, dest="curves", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.curves or [HDDT_CURVES, CLASSICAL_CURVES], args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
