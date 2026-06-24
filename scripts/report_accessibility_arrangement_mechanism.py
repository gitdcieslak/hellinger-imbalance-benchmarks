"""Explain arrangement morphology using posterior concentration geometry."""

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

from report_accessibility_morphology_decomposition import CAPACITY_FEATURES, controlled_effects, load_decomposition_dataset, score_for, variance_partition  # noqa: E402
from report_accessibility_morphology_sufficiency import CLASSICAL_CURVES, HDDT_CURVES  # noqa: E402
from report_definitive_reachability_topology import load_curve_files, pivot_curves  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402
from report_posterior_concentration_hypothesis import score_distribution_from_curve  # noqa: E402


OUT = ROOT / "reports" / "topology"
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"
TARGETS = [TARGET, SURVIVAL, "elevation"]


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


def top_mass(weights: np.ndarray, frac: float) -> float:
    w = np.sort(np.asarray(weights, dtype=float))[::-1]
    if w.size == 0 or w.sum() <= 0:
        return 0.0
    n = max(1, int(np.ceil(frac * w.size)))
    return float(w[:n].sum() / w.sum())


def arrangement_features_from_curves(matrix: pd.DataFrame, thresholds: np.ndarray) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows = []
    t = np.asarray(thresholds, dtype=float)
    for _, row in matrix.iterrows():
        curve = row.to_numpy(dtype=float)
        scores, weights = score_distribution_from_curve(curve, t)
        weights = weights / max(1e-12, weights.sum())
        order = np.argsort(scores)
        s = scores[order]
        w = weights[order]
        gaps = np.diff(s) if s.size > 1 else np.asarray([0.0])
        drops = np.maximum(0.0, curve[:-1] - curve[1:])
        drop_w = drops / drops.sum() if drops.sum() > 0 else np.asarray([1.0])
        occupied = np.zeros(max(1, len(t) - 1), dtype=int)
        bin_idx = np.clip(np.searchsorted(t, s, side="right") - 1, 0, len(occupied) - 1)
        occupied[np.unique(bin_idx)] = 1
        runs = run_lengths(occupied)
        empty_runs = run_lengths(1 - occupied)
        rows.append(
            {
                "positive_score_gini": gini(w),
                "top_1pct_mass": top_mass(w, 0.01),
                "top_5pct_mass": top_mass(w, 0.05),
                "top_10pct_mass": top_mass(w, 0.10),
                "tail_mass_above_95pct": float(w[s >= np.quantile(s, 0.95)].sum()) if s.size else 0.0,
                "tail_mass_above_99pct": float(w[s >= np.quantile(s, 0.99)].sum()) if s.size else 0.0,
                "concentration_index": gini(w) * top_mass(w, 0.10),
                "largest_score_gap": float(gaps.max()) if gaps.size else 0.0,
                "mean_gap": float(gaps.mean()) if gaps.size else 0.0,
                "gap_variance": float(gaps.var()) if gaps.size else 0.0,
                "top10_gap_mean": top_gap_mean(s, w, 0.10),
                "top20_gap_mean": top_gap_mean(s, w, 0.20),
                "quantile_gap_entropy": entropy(np.diff(np.quantile(s, np.linspace(0, 1, min(11, max(2, s.size)))))) if s.size > 1 else 0.0,
                "occupied_bin_count": int(occupied.sum()),
                "largest_empty_gap": float(empty_runs.max() / max(1, occupied.size)) if empty_runs.size else 0.0,
                "occupancy_fragmentation": float(len(runs)),
                "occupancy_contiguity": float(runs.max() / max(1, occupied.sum())) if runs.size and occupied.sum() else 0.0,
                "run_length_entropy": entropy(runs),
                "drop_entropy": entropy(drops),
                "drop_gini": gini(drops),
                "effective_drop_count": float(1.0 / max(1e-12, np.sum(drop_w**2))),
                "max_drop": float(drops.max()) if drops.size else 0.0,
                "top3_drop_mass": float(np.sort(drops)[-3:].sum() / max(1e-12, drops.sum())) if drops.size else 0.0,
                "drop_concentration": gini(drops) * (float(drops.max()) if drops.size else 0.0),
            }
        )
    blocks = {
        "concentration": ["positive_score_gini", "top_1pct_mass", "top_5pct_mass", "top_10pct_mass", "tail_mass_above_95pct", "tail_mass_above_99pct", "concentration_index"],
        "spacing": ["largest_score_gap", "mean_gap", "gap_variance", "top10_gap_mean", "top20_gap_mean", "quantile_gap_entropy"],
        "occupancy": ["occupied_bin_count", "largest_empty_gap", "occupancy_fragmentation", "occupancy_contiguity", "run_length_entropy"],
        "drop_structure": ["drop_entropy", "drop_gini", "effective_drop_count", "max_drop", "top3_drop_mass", "drop_concentration"],
    }
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan), blocks


def run_lengths(binary: np.ndarray) -> np.ndarray:
    arr = np.asarray(binary, dtype=int)
    if arr.size == 0 or arr.sum() == 0:
        return np.asarray([], dtype=float)
    lengths = []
    count = 0
    for value in arr:
        if value:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    return np.asarray(lengths, dtype=float)


def top_gap_mean(scores: np.ndarray, weights: np.ndarray, frac: float) -> float:
    if scores.size < 2:
        return 0.0
    cutoff = np.quantile(scores, 1.0 - frac)
    subset = np.sort(scores[scores >= cutoff])
    return float(np.diff(subset).mean()) if subset.size > 1 else 0.0


def grouped_score(meta: pd.DataFrame, features: pd.DataFrame, target: str) -> tuple[float, float]:
    cols = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c]) and features[c].notna().any()]
    data = pd.concat([meta[[target, "dataset_id"]].reset_index(drop=True), features[cols].reset_index(drop=True)], axis=1).dropna()
    groups = data["dataset_id"].astype(str)
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    X = data.drop(columns=[target, "dataset_id"])
    y = data[target]
    best_r2 = -np.inf
    best_mae = np.nan
    for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=41, n_jobs=1)]:
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        score = float(r2_score(y, pred))
        if score > best_r2:
            best_r2 = score
            best_mae = float(mean_absolute_error(y, pred))
    return best_r2, best_mae


def model_scores(meta: pd.DataFrame, capacity: pd.DataFrame, arrangement: pd.DataFrame, full_morphology: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for name, frame in {
            "capacity": capacity,
            "arrangement_mechanism": arrangement,
            "capacity_plus_arrangement_mechanism": pd.concat([capacity.add_prefix("capacity__"), arrangement.add_prefix("arrangement__")], axis=1),
            "full_morphology": full_morphology,
        }.items():
            r2, mae = grouped_score(meta, frame, target)
            rows.append({"target": target, "model_spec": name, "grouped_cv_r2": r2, "grouped_cv_mae": mae, "n_features": frame.shape[1]})
    scores = pd.DataFrame(rows)
    for target in TARGETS:
        full = score_for(scores.rename(columns={"model_spec": "model_spec"}), target, "full_morphology")
        arr = score_for(scores, target, "arrangement_mechanism")
        scores.loc[scores["target"] == target, "delta_from_full_morphology"] = scores.loc[scores["target"] == target, "grouped_cv_r2"] - full
        scores.loc[scores["target"] == target, "arrangement_fraction_of_full"] = arr / full if full and np.isfinite(full) else np.nan
    return scores


def mechanism_importance(meta: pd.DataFrame, arrangement: pd.DataFrame, blocks: dict[str, list[str]]) -> pd.DataFrame:
    data = pd.concat([meta[[TARGET]].reset_index(drop=True), arrangement.reset_index(drop=True)], axis=1).dropna()
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)
    model = RandomForestRegressor(n_estimators=350, min_samples_leaf=5, random_state=43, n_jobs=1).fit(X_train, y_train)
    imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=43, n_jobs=1)
    feature_imp = pd.DataFrame({"feature": X.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std})
    rows = []
    for block, cols in blocks.items():
        rows.append({"feature": f"BLOCK::{block}", "importance_mean": float(feature_imp[feature_imp["feature"].isin(cols)]["importance_mean"].sum()), "importance_std": np.nan})
    return pd.concat([pd.DataFrame(rows), feature_imp], ignore_index=True).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def partition_from_scores(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        cap = max(0, score_for(scores, target, "capacity"))
        arr = max(0, score_for(scores, target, "arrangement_mechanism"))
        both = max(0, score_for(scores, target, "capacity_plus_arrangement_mechanism"))
        rows.extend([
            {"target": target, "component": "capacity_unique", "variance_fraction": max(0, both - arr)},
            {"target": target, "component": "arrangement_unique", "variance_fraction": max(0, both - cap)},
            {"target": target, "component": "shared", "variance_fraction": max(0, min(cap, arr))},
            {"target": target, "component": "unexplained", "variance_fraction": max(0, 1 - both)},
        ])
    return pd.DataFrame(rows)


def plot_umap(meta: pd.DataFrame, arrangement: pd.DataFrame, output: Path) -> Path:
    X = arrangement.fillna(arrangement.median(numeric_only=True)).fillna(0)
    scaled = StandardScaler().fit_transform(X)
    try:
        import umap
        emb = umap.UMAP(n_neighbors=25, min_dist=0.08, random_state=44).fit_transform(scaled)
    except Exception:
        emb = PCA(n_components=2, random_state=44).fit_transform(scaled)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, [TARGET, SURVIVAL, "model_id"], strict=True):
        values = meta[col]
        if pd.api.types.is_numeric_dtype(values):
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, cmap="viridis", s=12, alpha=0.75)
            fig.colorbar(sc, ax=ax, label=col)
        else:
            codes, _ = pd.factorize(values.astype(str)); ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap="tab20", s=12, alpha=0.75)
        ax.set_title(col)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_importance(importance: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6)); top = importance.head(15).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], color="steelblue"); ax.set_title("Arrangement Mechanism Importance")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_effects(effects: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", linewidth=1)
    if not effects.empty:
        ax.bar(effects["band"].astype(str), effects["cliffiness_high_minus_low"], color="steelblue")
    ax.set_xlabel("capacity band"); ax.set_ylabel("high arrangement - low arrangement cliffiness"); ax.set_title("Capacity-Controlled Arrangement Effect v2")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def write_report(curve_paths: list[Path], output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, capacity, _, full = load_decomposition_dataset(curve_paths)
    curves = load_curve_files(curve_paths)
    _, matrix, thresholds = pivot_curves(curves)
    arrangement, blocks = arrangement_features_from_curves(matrix, thresholds)
    scores = model_scores(meta, capacity, arrangement, full)
    part = partition_from_scores(scores)
    importance = mechanism_importance(meta, arrangement, blocks)
    cap_ctrl, _ = controlled_effects(meta, capacity, arrangement)
    arr_r2 = score_for(scores, TARGET, "arrangement_mechanism")
    full_r2 = score_for(scores, TARGET, "full_morphology")
    outcome = "Outcome A" if arr_r2 >= 0.80 and arr_r2 / full_r2 >= 0.80 else "Outcome B" if arr_r2 >= 0.80 else "Outcome C"
    summary = {
        "n_runs": int(len(meta)), "n_arrangement_features": int(arrangement.shape[1]),
        "arrangement_cliffiness_r2": arr_r2, "full_morphology_cliffiness_r2": full_r2,
        "arrangement_fraction_of_full": arr_r2 / full_r2 if full_r2 else np.nan,
        "capacity_controlled_arrangement_abs_delta": float(cap_ctrl["cliffiness_high_minus_low"].abs().mean()) if not cap_ctrl.empty else np.nan,
        "top_block": str(importance[importance["feature"].str.startswith("BLOCK::")].iloc[0]["feature"]).replace("BLOCK::", "") if not importance.empty else "",
        "outcome": outcome,
    }
    report = output_dir / "accessibility_arrangement_mechanism.md"
    summary_path = output_dir / "arrangement_mechanism_summary.json"
    imp_path = output_dir / "arrangement_feature_importance.csv"
    scores_path = output_dir / "arrangement_model_scores.csv"
    part_path = output_dir / "arrangement_variance_partition.csv"
    importance.to_csv(imp_path, index=False); scores.to_csv(scores_path, index=False); part.to_csv(part_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if outcome == "Outcome A":
        answer = f"Outcome A: arrangement is largely reconstructed by concentration/spacing/occupancy/drop geometry with cliffiness R2 `{arr_r2:.4f}`."
    elif outcome == "Outcome B":
        answer = f"Outcome B: arrangement requires multiple geometry mechanisms and reaches cliffiness R2 `{arr_r2:.4f}`."
    else:
        answer = f"Outcome C: refined arrangement geometry is informative but does not fully reconstruct arrangement; cliffiness R2 is `{arr_r2:.4f}`."
    report.write_text("\n".join([
        "# Accessibility Arrangement Mechanism",
        "", "## Executive Answer", answer,
        "", "## Arrangement Model Scores", _markdown_table(scores),
        "", "## Variance Partition", _markdown_table(part),
        "", "## Mechanism Ranking", _markdown_table(importance.head(25)),
        "", "## Capacity-Controlled Arrangement Effect", _markdown_table(cap_ctrl),
    ]) + "\n", encoding="utf-8")
    return (report, imp_path, scores_path, part_path, plot_umap(meta, arrangement, output_dir / "arrangement_umap.png"), plot_importance(importance, output_dir / "arrangement_feature_importance.png"), plot_effects(cap_ctrl, output_dir / "capacity_controlled_arrangement_effect_v2.png"), summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", action="append", type=Path, dest="curves", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.curves or [HDDT_CURVES, CLASSICAL_CURVES], args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
