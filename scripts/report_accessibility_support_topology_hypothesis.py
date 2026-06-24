"""Test whether positive support topology explains accessibility cliffiness."""

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

from report_accessibility_morphology_decomposition import load_decomposition_dataset, score_for  # noqa: E402
from report_accessibility_morphology_sufficiency import HDDT_CURVES  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402
from report_minority_reachability_cohorts import DEFAULT_POSITIVE_SCORES, load_positive_scores  # noqa: E402


OUT = ROOT / "reports" / "topology"
TARGET = "minority_survival_cliffiness"
CAPACITY_COLS = ["n_score_groups", "effective_support_groups", "support_unique_score_ratio"]


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


def support_topology_features(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, group in scores.groupby("run_id", sort=False):
        s = np.clip(group["score"].to_numpy(dtype=float), 0.0, 1.0)
        rounded = np.round(s, 6)
        vals, counts = np.unique(rounded, return_counts=True)
        mass = counts.astype(float) / max(1, counts.sum())
        order = np.argsort(vals)
        vals, mass = vals[order], mass[order]
        bins = np.floor(vals * 100).astype(int).clip(0, 100)
        bin_mass = pd.Series(mass).groupby(bins).sum().sort_index()
        islands = contiguous_islands(bin_mass.index.to_numpy(), bin_mass.to_numpy())
        island_mass = np.asarray([m for _, m in islands], dtype=float) if islands else np.asarray([1.0])
        fragile = mass[vals < 0.25].sum()
        stable = mass[vals >= 0.75].sum()
        high = mass[vals >= 0.90]
        high_vals = vals[vals >= 0.90]
        first = group.iloc[0]
        rows.append(
            {
                "run_id": run_id,
                "dataset_id": first["dataset_id"],
                "task_id": first["task_id"],
                "model_id": first["model_id"],
                "minority_survival_cliffiness": float(first["minority_survival_cliffiness"]),
                "minority_survival_auc": float(first.get("minority_survival_auc", np.nan)),
                "breadth": float(first.get("breadth", np.nan)),
                "elevation": float(first.get("elevation", np.nan)),
                "n_positives": int(len(s)),
                "n_score_groups": int(len(vals)),
                "support_unique_score_ratio": float(len(vals) / max(1, len(s))),
                "group_mass_entropy": entropy(mass),
                "group_mass_gini": gini(mass),
                "group_mass_hhi": float(np.sum(mass**2)),
                "top1_group_mass": top_mass(mass, 1),
                "top3_group_mass": top_mass(mass, 3),
                "top5_group_mass": top_mass(mass, 5),
                "effective_support_groups": float(1.0 / max(1e-12, np.sum(mass**2))),
                "n_support_islands": int(len(islands)),
                "largest_island_mass": float(island_mass.max()),
                "island_entropy": entropy(island_mass),
                "island_gini": gini(island_mass),
                "small_island_fraction": float(island_mass[island_mass < 0.05].sum()),
                "support_evenness": float(entropy(mass) / max(1e-12, np.log(len(mass)))) if len(mass) > 1 else 1.0,
                "largest_to_median_group_ratio": float(mass.max() / max(1e-12, np.median(mass))),
                "tail_mass_tiny_groups": float(mass[mass < 0.01].sum()),
                "high_threshold_survivor_concentration": float(high.max() / max(1e-12, high.sum())) if high.size else 0.0,
                "persistence_weighted_mass": float(np.sum(mass * vals)),
                "fragile_support_mass": float(fragile),
                "stable_support_mass": float(stable),
                "stable_to_fragile_ratio": float(stable / max(1e-12, fragile)),
            }
        )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def contiguous_islands(bin_ids: np.ndarray, masses: np.ndarray) -> list[tuple[list[int], float]]:
    if len(bin_ids) == 0:
        return []
    islands: list[tuple[list[int], float]] = []
    current = [int(bin_ids[0])]
    total = float(masses[0])
    for b, m in zip(bin_ids[1:], masses[1:], strict=False):
        if int(b) == current[-1] + 1:
            current.append(int(b)); total += float(m)
        else:
            islands.append((current, total)); current = [int(b)]; total = float(m)
    islands.append((current, total))
    return islands


def top_mass(mass: np.ndarray, k: int) -> float:
    m = np.sort(np.asarray(mass, dtype=float))[::-1]
    return float(m[: min(k, len(m))].sum()) if m.size else 0.0


def load_full_morphology_for_runs(run_ids: pd.Series) -> pd.DataFrame:
    meta, _, _, full = load_decomposition_dataset([HDDT_CURVES])
    full = full.copy(); full["run_id"] = meta["run_id"].to_numpy()
    indexed = full.set_index("run_id")
    missing = set(run_ids) - set(indexed.index)
    if missing:
        raise KeyError("some run_ids are unavailable in full morphology")
    return indexed.loc[run_ids].reset_index(drop=True).drop(columns=["run_id"], errors="ignore")


def grouped_score(meta: pd.DataFrame, features: pd.DataFrame) -> tuple[float, float]:
    data = pd.concat([meta[[TARGET, "dataset_id"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    groups = data["dataset_id"].astype(str)
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    X = data.drop(columns=[TARGET, "dataset_id"]); y = data[TARGET]
    best_r2, best_mae = -np.inf, np.nan
    for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=81, n_jobs=1)]:
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        score = float(r2_score(y, pred))
        if score > best_r2:
            best_r2, best_mae = score, float(mean_absolute_error(y, pred))
    return best_r2, best_mae


def model_scores(meta: pd.DataFrame, support: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    capacity = support[CAPACITY_COLS]
    topology_cols = [c for c in support.columns if c not in {"run_id", "dataset_id", "task_id", "model_id", TARGET, "minority_survival_auc", "breadth", "elevation", "n_positives", *CAPACITY_COLS}]
    topology = support[topology_cols]
    specs = {"capacity": capacity, "support_topology": topology, "capacity_plus_support_topology": pd.concat([capacity.add_prefix("capacity__"), topology.add_prefix("support__")], axis=1), "full_morphology": full}
    rows = []
    for name, frame in specs.items():
        r2, mae = grouped_score(meta, frame)
        rows.append({"model_spec": name, "grouped_cv_r2": r2, "grouped_cv_mae": mae, "n_features": frame.shape[1]})
    scores = pd.DataFrame(rows)
    cap = float(scores.loc[scores["model_spec"] == "capacity", "grouped_cv_r2"].iloc[0])
    full_r2 = float(scores.loc[scores["model_spec"] == "full_morphology", "grouped_cv_r2"].iloc[0])
    scores["delta_vs_capacity"] = scores["grouped_cv_r2"] - cap
    scores["gap_to_full_morphology"] = full_r2 - scores["grouped_cv_r2"]
    return scores


def variance_partition(scores: pd.DataFrame) -> pd.DataFrame:
    lookup = dict(zip(scores["model_spec"], scores["grouped_cv_r2"]))
    cap, top, both = max(0, lookup["capacity"]), max(0, lookup["support_topology"]), max(0, lookup["capacity_plus_support_topology"])
    return pd.DataFrame([
        {"component": "capacity_unique", "variance_fraction": max(0, both - top)},
        {"component": "support_topology_unique", "variance_fraction": max(0, both - cap)},
        {"component": "shared", "variance_fraction": max(0, min(cap, top))},
        {"component": "unexplained", "variance_fraction": max(0, 1 - both)},
    ])


def controlled_effect(meta: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    risk = support[["group_mass_gini", "top1_group_mass", "largest_island_mass", "fragile_support_mass"]].mean(axis=1)
    data = pd.concat([meta.reset_index(drop=True), support[["effective_support_groups"]].reset_index(drop=True), risk.rename("support_risk")], axis=1).dropna()
    data["capacity_band"] = pd.qcut(data["effective_support_groups"].rank(method="first"), 8, labels=False) + 1
    rows = []
    for band, group in data.groupby("capacity_band"):
        low = group[group["support_risk"] <= group["support_risk"].quantile(.25)]
        high = group[group["support_risk"] >= group["support_risk"].quantile(.75)]
        rows.append({"band": int(band), "n_low": len(low), "n_high": len(high), "low_cliffiness": float(low[TARGET].mean()), "high_cliffiness": float(high[TARGET].mean()), "cliffiness_high_minus_low": float(high[TARGET].mean() - low[TARGET].mean())})
    return pd.DataFrame(rows)


def feature_importance(meta: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in support.columns if c not in {"run_id", "dataset_id", "task_id", "model_id", TARGET, "minority_survival_auc", "breadth", "elevation"} and pd.api.types.is_numeric_dtype(support[c])]
    data = pd.concat([meta[[TARGET]].reset_index(drop=True), support[cols].reset_index(drop=True)], axis=1).dropna()
    X = data.drop(columns=[TARGET]); y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=83)
    model = RandomForestRegressor(n_estimators=350, min_samples_leaf=5, random_state=83, n_jobs=1).fit(X_train, y_train)
    imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=83, n_jobs=1)
    return pd.DataFrame({"feature": X.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std}).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_umap(meta: pd.DataFrame, support: pd.DataFrame, output: Path) -> Path:
    cols = [c for c in support.columns if pd.api.types.is_numeric_dtype(support[c]) and c not in {TARGET, "minority_survival_auc", "breadth", "elevation"}]
    X = support[cols].fillna(support[cols].median(numeric_only=True)).fillna(0)
    scaled = StandardScaler().fit_transform(X)
    try:
        import umap
        emb = umap.UMAP(n_neighbors=25, min_dist=.08, random_state=85).fit_transform(scaled)
    except Exception:
        emb = PCA(n_components=2, random_state=85).fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(8, 6)); sc = ax.scatter(emb[:, 0], emb[:, 1], c=meta[TARGET], cmap="magma", s=12, alpha=.7); fig.colorbar(sc, ax=ax, label="cliffiness"); ax.set_title("Support Topology UMAP"); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_scatter(support: pd.DataFrame, x: str, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(support[x], support[TARGET], s=12, alpha=.5); ax.set_xlabel(x); ax.set_ylabel("cliffiness"); ax.set_title(f"{x} vs cliffiness"); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_controlled(effects: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5)); ax.axhline(0, color="black", linewidth=1); ax.bar(effects["band"].astype(str), effects["cliffiness_high_minus_low"], color="steelblue"); ax.set_xlabel("capacity band"); ax.set_ylabel("high risk - low risk cliffiness"); ax.set_title("Capacity-Controlled Support Topology Effect"); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def write_report(positive_scores: Path = DEFAULT_POSITIVE_SCORES, output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = load_positive_scores(positive_scores)
    support = support_topology_features(scores)
    meta = support[["run_id", "dataset_id", "task_id", "model_id", TARGET, "minority_survival_auc", "breadth", "elevation"]].copy()
    try:
        full = load_full_morphology_for_runs(meta["run_id"])
    except Exception:
        full = support[["breadth", "elevation", "minority_survival_auc", "n_score_groups", "group_mass_entropy", "top1_group_mass", "effective_support_groups"]].copy()
    model = model_scores(meta, support, full)
    partition = variance_partition(model)
    effects = controlled_effect(meta, support)
    importance = feature_importance(meta, support)
    lookup = dict(zip(model["model_spec"], model["grouped_cv_r2"]))
    cap, both, full_r2 = lookup["capacity"], lookup["capacity_plus_support_topology"], lookup["full_morphology"]
    closes = (both - cap) / max(1e-12, full_r2 - cap)
    if both - cap >= .10 and closes >= .50:
        outcome = "Outcome A"
    elif both > cap:
        outcome = "Outcome B"
    else:
        outcome = "Outcome C"
    summary = {"n_runs": int(len(meta)), "capacity_r2": cap, "support_topology_r2": lookup["support_topology"], "capacity_plus_support_topology_r2": both, "full_morphology_r2": full_r2, "delta_r2_vs_capacity": both - cap, "gap_closed_fraction": closes, "outcome": outcome}
    report = output_dir / "accessibility_support_topology_hypothesis.md"; summary_path = output_dir / "accessibility_support_topology_summary.json"; scores_path = output_dir / "support_topology_model_scores.csv"; part_path = output_dir / "support_topology_variance_partition.csv"; features_path = output_dir / "support_topology_features.csv"; effects_path = output_dir / "support_topology_capacity_controlled_effects.csv"; imp_path = output_dir / "support_topology_feature_importance.csv"
    support.to_csv(features_path, index=False); model.to_csv(scores_path, index=False); partition.to_csv(part_path, index=False); effects.to_csv(effects_path, index=False); importance.to_csv(imp_path, index=False); summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text("\n".join(["# Accessibility Support Topology Hypothesis", "", "## Executive Answer", f"{outcome}: support topology delta over capacity is `{both - cap:.4f}` and closes `{closes:.4f}` of the full morphology gap.", "", "## Model Scores", _markdown_table(model), "", "## Variance Partition", _markdown_table(partition), "", "## Feature Importance", _markdown_table(importance.head(25)), "", "## Capacity-Controlled Support Topology Effect", _markdown_table(effects)]) + "\n", encoding="utf-8")
    return (report, summary_path, scores_path, part_path, features_path, effects_path, imp_path, plot_umap(meta, support, output_dir / "support_topology_umap.png"), plot_controlled(effects, output_dir / "capacity_controlled_support_topology_effect.png"), plot_scatter(support, "n_support_islands", output_dir / "support_fragmentation_vs_cliffiness.png"), plot_scatter(support, "top1_group_mass", output_dir / "support_concentration_vs_cliffiness.png"), plot_scatter(support, "stable_to_fragile_ratio", output_dir / "stable_fragile_support_ratio_vs_cliffiness.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-scores", type=Path, default=DEFAULT_POSITIVE_SCORES)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.positive_scores, args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
