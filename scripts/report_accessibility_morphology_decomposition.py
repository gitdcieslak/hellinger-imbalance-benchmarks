"""Decompose accessibility morphology into capacity and arrangement."""

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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_accessibility_morphology_sufficiency import CLASSICAL_CURVES, HDDT_CURVES, load_dataset  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402


OUT = ROOT / "reports" / "topology"
TARGET = "minority_survival_cliffiness"
SURVIVAL = "minority_survival_auc"
CAPACITY_FEATURES = [
    "effective_support",
    "inverse_effective_support",
    "positive_score_entropy",
    "joined_positive_unique_score_ratio",
    "joined_positive_effective_score_bins",
    "joined_positive_histogram_entropy",
    "positive_histogram_entropy",
]
ARRANGEMENT_FEATURES = [
    "top10_to_top1_ratio",
    "positive_score_gini",
    "top_1pct_mass",
    "top_5pct_mass",
    "top_10pct_mass",
    "tail_mass_above_99pct",
    "density_around_090",
    "density_around_095",
    "density_around_099",
    "joined_positive_score_gini_or_concentration_index",
    "joined_positive_top_bin_mass",
    "joined_concentration_entropy_product",
]


def available(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]


def load_decomposition_dataset(curve_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta, morphology, _, _ = load_dataset(curve_paths)
    capacity = morphology[available(CAPACITY_FEATURES, morphology)].copy()
    arrangement = morphology[available(ARRANGEMENT_FEATURES, morphology)].copy()
    full = morphology[[c for c in morphology.columns if pd.api.types.is_numeric_dtype(morphology[c]) and morphology[c].notna().any()]].copy()
    return meta, capacity.replace([np.inf, -np.inf], np.nan), arrangement.replace([np.inf, -np.inf], np.nan), full.replace([np.inf, -np.inf], np.nan)


def grouped_regression(meta: pd.DataFrame, features: pd.DataFrame, target: str) -> tuple[float, float, bool, str]:
    cols = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c]) and features[c].notna().any()]
    data = pd.concat([meta[[target, "dataset_id"]].reset_index(drop=True), features[cols].reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 20:
        return np.nan, np.nan, False, "too_few_rows"
    groups = data["dataset_id"].astype(str)
    if groups.nunique() < 2:
        return np.nan, np.nan, False, "too_few_groups"
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    X = data.drop(columns=[target, "dataset_id"])
    y = data[target]
    best_r2 = -np.inf
    best_mae = np.nan
    for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=31, n_jobs=-1)]:
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        score = float(r2_score(y, pred))
        if score > best_r2:
            best_r2 = score
            best_mae = float(mean_absolute_error(y, pred))
    return best_r2, best_mae, True, ""


def model_scores(meta: pd.DataFrame, capacity: pd.DataFrame, arrangement: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "capacity": capacity,
        "arrangement": arrangement,
        "capacity_plus_arrangement": pd.concat([capacity.add_prefix("capacity__"), arrangement.add_prefix("arrangement__")], axis=1),
        "full_morphology": full,
    }
    rows = []
    for target in [TARGET, SURVIVAL]:
        for name, frame in specs.items():
            r2, mae, valid, reason = grouped_regression(meta, frame, target)
            rows.append({"target": target, "model_spec": name, "grouped_cv_r2": r2, "grouped_cv_mae": mae, "n_features": int(frame.shape[1]), "valid": valid, "skip_reason": reason})
    scores = pd.DataFrame(rows)
    for target in [TARGET, SURVIVAL]:
        target_mask = scores["target"] == target
        full_r2 = score_for(scores, target, "full_morphology")
        cap_arr = score_for(scores, target, "capacity_plus_arrangement")
        scores.loc[target_mask, "delta_from_full_morphology"] = scores.loc[target_mask, "grouped_cv_r2"] - full_r2
        scores.loc[target_mask, "capacity_arrangement_gap_to_full"] = full_r2 - cap_arr
    return scores


def score_for(scores: pd.DataFrame, target: str, spec: str) -> float:
    match = scores[(scores["target"] == target) & (scores["model_spec"] == spec)]
    return float(match["grouped_cv_r2"].iloc[0]) if not match.empty else np.nan


def variance_partition(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in [TARGET, SURVIVAL]:
        cap = max(0.0, score_for(scores, target, "capacity"))
        arr = max(0.0, score_for(scores, target, "arrangement"))
        both = max(0.0, score_for(scores, target, "capacity_plus_arrangement"))
        capacity_unique = max(0.0, both - arr)
        arrangement_unique = max(0.0, both - cap)
        shared = max(0.0, min(cap, arr, both - capacity_unique - arrangement_unique))
        unexplained = max(0.0, 1.0 - both)
        rows.extend([
            {"target": target, "component": "capacity_unique", "variance_fraction": capacity_unique},
            {"target": target, "component": "arrangement_unique", "variance_fraction": arrangement_unique},
            {"target": target, "component": "shared_capacity_arrangement", "variance_fraction": shared},
            {"target": target, "component": "unexplained", "variance_fraction": unexplained},
        ])
    return pd.DataFrame(rows)


def frontier(meta: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([meta.reset_index(drop=True), capacity[["effective_support"]].reset_index(drop=True)], axis=1).dropna(subset=["effective_support", TARGET, SURVIVAL])
    data["capacity_bin"] = pd.qcut(data["effective_support"].rank(method="first"), 10, labels=False) + 1
    return data.groupby("capacity_bin", as_index=False).agg(
        n_runs=(TARGET, "size"),
        mean_effective_support=("effective_support", "mean"),
        mean_cliffiness=(TARGET, "mean"),
        min_cliffiness=(TARGET, "min"),
        p10_cliffiness=(TARGET, lambda s: float(s.quantile(0.10))),
        median_cliffiness=(TARGET, "median"),
        p90_cliffiness=(TARGET, lambda s: float(s.quantile(0.90))),
        max_cliffiness=(TARGET, "max"),
        mean_survival_auc=(SURVIVAL, "mean"),
    )


def controlled_effects(meta: pd.DataFrame, capacity: pd.DataFrame, arrangement: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    concentration = arrangement_metric(arrangement)
    data = pd.concat([meta.reset_index(drop=True), capacity[["effective_support"]].reset_index(drop=True), concentration.rename("arrangement_concentration")], axis=1).dropna(subset=["effective_support", "arrangement_concentration", TARGET, SURVIVAL])
    n_bands = max(2, min(8, len(data) // 20))
    data["capacity_band"] = pd.qcut(data["effective_support"].rank(method="first"), n_bands, labels=False) + 1
    data["arrangement_band"] = pd.qcut(data["arrangement_concentration"].rank(method="first"), n_bands, labels=False) + 1
    cap_ctrl = compare_extremes(data, group_col="capacity_band", split_col="arrangement_concentration", low_label="low_concentration", high_label="high_concentration")
    arr_ctrl = compare_extremes(data, group_col="arrangement_band", split_col="effective_support", low_label="low_capacity", high_label="high_capacity")
    return cap_ctrl, arr_ctrl


def arrangement_metric(arrangement: pd.DataFrame) -> pd.Series:
    preferred = [c for c in ["positive_score_gini", "top_1pct_mass", "top_5pct_mass", "top_10pct_mass", "tail_mass_above_99pct", "joined_positive_score_gini_or_concentration_index"] if c in arrangement.columns]
    if not preferred:
        return pd.Series(np.nan, index=arrangement.index)
    z = arrangement[preferred].replace([np.inf, -np.inf], np.nan)
    z = (z - z.mean()) / z.std(ddof=0).replace(0, np.nan)
    return z.mean(axis=1)


def compare_extremes(data: pd.DataFrame, group_col: str, split_col: str, low_label: str, high_label: str) -> pd.DataFrame:
    rows = []
    for band, group in data.groupby(group_col):
        if len(group) < 20:
            continue
        low = group[group[split_col] <= group[split_col].quantile(0.25)]
        high = group[group[split_col] >= group[split_col].quantile(0.75)]
        if low.empty or high.empty:
            continue
        rows.append({
            "band": int(band),
            "low_group": low_label,
            "high_group": high_label,
            "n_low": int(len(low)),
            "n_high": int(len(high)),
            "low_mean_cliffiness": float(low[TARGET].mean()),
            "high_mean_cliffiness": float(high[TARGET].mean()),
            "cliffiness_high_minus_low": float(high[TARGET].mean() - low[TARGET].mean()),
            "low_mean_survival_auc": float(low[SURVIVAL].mean()),
            "high_mean_survival_auc": float(high[SURVIVAL].mean()),
            "survival_high_minus_low": float(high[SURVIVAL].mean() - low[SURVIVAL].mean()),
        })
    return pd.DataFrame(rows)


def summary_decision(scores: pd.DataFrame, partition: pd.DataFrame, cap_ctrl: pd.DataFrame, front: pd.DataFrame) -> tuple[str, str, dict[str, object]]:
    full = score_for(scores, TARGET, "full_morphology")
    cap_arr = score_for(scores, TARGET, "capacity_plus_arrangement")
    cap_unique = float(partition[(partition["target"] == TARGET) & (partition["component"] == "capacity_unique")]["variance_fraction"].iloc[0])
    arr_unique = float(partition[(partition["target"] == TARGET) & (partition["component"] == "arrangement_unique")]["variance_fraction"].iloc[0])
    frontier_slope = float(np.polyfit(front["mean_effective_support"], front["p90_cliffiness"], 1)[0]) if len(front) >= 2 else np.nan
    arrangement_delta = float(cap_ctrl["cliffiness_high_minus_low"].abs().mean()) if not cap_ctrl.empty else np.nan
    supported = bool(np.isfinite(full) and np.isfinite(cap_arr) and (full - cap_arr) <= 0.03 and cap_unique > 0 and arr_unique > 0 and np.isfinite(frontier_slope) and frontier_slope < 0 and np.isfinite(arrangement_delta) and arrangement_delta >= 0.03)
    payload = {
        "full_morphology_r2": full,
        "capacity_plus_arrangement_r2": cap_arr,
        "gap_to_full_morphology": full - cap_arr if np.isfinite(full) and np.isfinite(cap_arr) else np.nan,
        "capacity_unique_variance": cap_unique,
        "arrangement_unique_variance": arr_unique,
        "frontier_p90_slope": frontier_slope,
        "mean_capacity_controlled_arrangement_abs_delta": arrangement_delta,
        "supported": supported,
    }
    if supported:
        return "supported", "Capacity defines the feasible accessibility envelope, while arrangement determines where runs land within that envelope.", payload
    return "mixed", "Capacity and arrangement both carry signal, but the decomposition does not satisfy every predefined support criterion.", payload


def plot_partition(partition: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    cliff = partition[partition["target"] == TARGET]
    ax.bar(cliff["component"], cliff["variance_fraction"], color=["#4c78a8", "#f58518", "#54a24b", "#bab0ac"])
    ax.set_ylabel("R2 fraction")
    ax.set_title("Capacity/Arrangement Variance Partition")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_frontier(front: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(front["mean_effective_support"], front["p10_cliffiness"], front["p90_cliffiness"], alpha=0.25, label="p10-p90")
    ax.plot(front["mean_effective_support"], front["median_cliffiness"], marker="o", label="median")
    ax.plot(front["mean_effective_support"], front["max_cliffiness"], linestyle="--", label="max")
    ax.set_xlabel("effective support"); ax.set_ylabel("cliffiness"); ax.set_title("Capacity Frontier Envelope"); ax.legend()
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_effects(effects: pd.DataFrame, output: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not effects.empty:
        ax.axhline(0, color="black", linewidth=1)
        ax.bar(effects["band"].astype(str), effects["cliffiness_high_minus_low"], color="steelblue")
    ax.set_xlabel("control band"); ax.set_ylabel("high - low cliffiness"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_scatter(meta: pd.DataFrame, capacity: pd.DataFrame, arrangement: pd.DataFrame, output: Path) -> Path:
    conc = arrangement_metric(arrangement)
    data = pd.concat([meta[[TARGET]].reset_index(drop=True), capacity[["effective_support"]].reset_index(drop=True), conc.rename("arrangement")], axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(data["effective_support"], data["arrangement"], c=data[TARGET], cmap="magma", s=14, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="cliffiness")
    ax.set_xlabel("effective support"); ax.set_ylabel("arrangement concentration"); ax.set_title("Capacity x Arrangement")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def write_report(curve_paths: list[Path], output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, capacity, arrangement, full = load_decomposition_dataset(curve_paths)
    scores = model_scores(meta, capacity, arrangement, full)
    partition = variance_partition(scores)
    front = frontier(meta, capacity)
    cap_ctrl, arr_ctrl = controlled_effects(meta, capacity, arrangement)
    outcome, outcome_text, payload = summary_decision(scores, partition, cap_ctrl, front)
    payload.update({"n_runs": int(len(meta)), "n_capacity_features": int(capacity.shape[1]), "n_arrangement_features": int(arrangement.shape[1]), "outcome": outcome, "outcome_text": outcome_text})

    report = output_dir / "accessibility_morphology_decomposition.md"
    summary_json = output_dir / "accessibility_morphology_decomposition_summary.json"
    scores_path = output_dir / "morphology_decomposition_model_scores.csv"
    partition_path = output_dir / "morphology_decomposition_variance_partition.csv"
    frontier_path = output_dir / "capacity_frontier_envelope.csv"
    cap_ctrl_path = output_dir / "capacity_controlled_arrangement_effects.csv"
    arr_ctrl_path = output_dir / "arrangement_controlled_capacity_effects.csv"
    scores.to_csv(scores_path, index=False); partition.to_csv(partition_path, index=False); front.to_csv(frontier_path, index=False); cap_ctrl.to_csv(cap_ctrl_path, index=False); arr_ctrl.to_csv(arr_ctrl_path, index=False)
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text("\n".join([
        "# Accessibility Morphology Decomposition",
        "",
        "## Executive Answer",
        outcome_text,
        "",
        "## Model Scores",
        _markdown_table(scores),
        "",
        "## Variance Partition",
        _markdown_table(partition),
        "",
        "## Capacity Frontier",
        _markdown_table(front),
        "",
        "## Capacity-Controlled Arrangement Effects",
        _markdown_table(cap_ctrl),
        "",
        "## Arrangement-Controlled Capacity Effects",
        _markdown_table(arr_ctrl),
        "",
        "## Interpretation",
        "Support requires capacity+arrangement to approach full morphology, positive unique variance for both blocks, a decreasing capacity frontier, and meaningful arrangement differences within capacity bands.",
    ]) + "\n", encoding="utf-8")
    return (
        report, summary_json, scores_path, partition_path, frontier_path, cap_ctrl_path, arr_ctrl_path,
        plot_partition(partition, output_dir / "morphology_decomposition_variance_partition.png"),
        plot_frontier(front, output_dir / "capacity_frontier_envelope.png"),
        plot_effects(cap_ctrl, output_dir / "capacity_controlled_arrangement_effect.png", "Arrangement Effect Within Capacity Bands"),
        plot_effects(arr_ctrl, output_dir / "arrangement_controlled_capacity_effect.png", "Capacity Effect Within Arrangement Bands"),
        plot_scatter(meta, capacity, arrangement, output_dir / "capacity_arrangement_scatter.png"),
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
