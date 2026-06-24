"""Test whether reachability transition structure explains accessibility."""

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

from report_accessibility_morphology_decomposition import CAPACITY_FEATURES, load_decomposition_dataset, score_for  # noqa: E402
from report_accessibility_morphology_sufficiency import CLASSICAL_CURVES, HDDT_CURVES  # noqa: E402
from report_definitive_reachability_topology import load_curve_files, pivot_curves  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402


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


def spectral_entropy(signal: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(np.asarray(signal, dtype=float))) ** 2
    return entropy(spectrum)


def transition_features_from_curves(matrix: pd.DataFrame, thresholds: np.ndarray) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows = []
    t = np.asarray(thresholds, dtype=float)
    for _, row in matrix.iterrows():
        curve = row.to_numpy(dtype=float)
        drops = np.maximum(0.0, curve[:-1] - curve[1:])
        total_drop = float(drops.sum())
        probs = drops / total_drop if total_drop > 0 else np.asarray([1.0])
        d1 = np.diff(curve)
        d2 = np.diff(curve, n=2)
        drop_thresholds = t[1:] if len(t) == len(curve) else np.linspace(0, 1, len(drops))
        band_means = {}
        for i in range(10):
            lo, hi = i / 10, (i + 1) / 10
            mask = (t >= lo) & (t <= hi) if len(t) == len(curve) else np.zeros_like(curve, dtype=bool)
            band_means[f"persistence_{i}0_{i+1}0"] = float(curve[mask].mean()) if mask.any() else np.nan
        active = drops > 1e-12
        rows.append(
            {
                "largest_drop": float(drops.max()) if drops.size else 0.0,
                "top3_drop_mass": float(np.sort(drops)[-3:].sum() / max(1e-12, total_drop)) if drops.size else 0.0,
                "top5_drop_mass": float(np.sort(drops)[-5:].sum() / max(1e-12, total_drop)) if drops.size else 0.0,
                "drop_mass_entropy": entropy(drops),
                "drop_gini": gini(drops),
                "effective_drop_count": float(1.0 / max(1e-12, np.sum(probs**2))),
                "drop_tail_fraction": float(drops[drop_thresholds >= 0.75].sum() / max(1e-12, total_drop)) if drops.size else 0.0,
                "early_persistence": float(curve[t <= 0.33].mean()) if len(t) == len(curve) else np.nan,
                "mid_persistence": float(curve[(t > 0.33) & (t <= 0.66)].mean()) if len(t) == len(curve) else np.nan,
                "late_persistence": float(curve[t > 0.66].mean()) if len(t) == len(curve) else np.nan,
                "total_variation": float(np.abs(d1).sum()) if d1.size else 0.0,
                "curvature_energy": float(np.sum(d2**2)) if d2.size else 0.0,
                "second_derivative_energy": float(np.sum(np.abs(d2))) if d2.size else 0.0,
                "spectral_entropy": spectral_entropy(d1) if d1.size else 0.0,
                "roughness_index": float(np.sum(np.abs(d2)) / max(1e-12, np.sum(np.abs(d1)))) if d1.size and d2.size else 0.0,
                "drop_concentration": gini(drops) * (float(drops.max()) if drops.size else 0.0),
                "drop_occupancy": float(active.mean()) if drops.size else 0.0,
                "largest_drop_fraction": float((drops.max() if drops.size else 0.0) / max(1e-12, total_drop)),
                "top10_drop_fraction": float(np.sort(drops)[-10:].sum() / max(1e-12, total_drop)) if drops.size else 0.0,
                "transition_entropy": entropy(drops),
                "transition_concentration": gini(drops),
                "transition_fragmentation": float(active.sum()),
                "transition_persistence": float(curve.mean()),
                **band_means,
            }
        )
    blocks = {
        "drop_spectrum": ["largest_drop", "top3_drop_mass", "top5_drop_mass", "drop_mass_entropy", "drop_gini", "effective_drop_count", "drop_tail_fraction"],
        "persistence": ["early_persistence", "mid_persistence", "late_persistence"] + [f"persistence_{i}0_{i+1}0" for i in range(10)],
        "roughness": ["total_variation", "curvature_energy", "second_derivative_energy", "spectral_entropy", "roughness_index"],
        "concentration": ["drop_concentration", "drop_occupancy", "largest_drop_fraction", "top10_drop_fraction"],
        "graph": ["transition_entropy", "transition_concentration", "transition_fragmentation", "transition_persistence"],
    }
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan), blocks


def grouped_score(meta: pd.DataFrame, features: pd.DataFrame, target: str) -> tuple[float, float]:
    data = pd.concat([meta[[target, "dataset_id"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    groups = data["dataset_id"].astype(str)
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    X = data.drop(columns=[target, "dataset_id"])
    y = data[target]
    best_r2, best_mae = -np.inf, np.nan
    for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=51, n_jobs=1)]:
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        score = float(r2_score(y, pred))
        if score > best_r2:
            best_r2, best_mae = score, float(mean_absolute_error(y, pred))
    return best_r2, best_mae


def model_scores(meta: pd.DataFrame, capacity: pd.DataFrame, transition: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "capacity": capacity,
        "transition": transition,
        "capacity_plus_transition": pd.concat([capacity.add_prefix("capacity__"), transition.add_prefix("transition__")], axis=1),
        "full_morphology": full,
    }
    rows = []
    for target in TARGETS:
        for name, frame in specs.items():
            r2, mae = grouped_score(meta, frame, target)
            rows.append({"target": target, "model_spec": name, "grouped_cv_r2": r2, "grouped_cv_mae": mae, "n_features": frame.shape[1]})
    scores = pd.DataFrame(rows)
    for target in TARGETS:
        cap = score_for(scores, target, "capacity")
        scores.loc[scores["target"] == target, "delta_vs_capacity"] = scores.loc[scores["target"] == target, "grouped_cv_r2"] - cap
    return scores


def variance_partition(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        cap = max(0, score_for(scores, target, "capacity"))
        trans = max(0, score_for(scores, target, "transition"))
        both = max(0, score_for(scores, target, "capacity_plus_transition"))
        rows.extend([
            {"target": target, "component": "capacity_unique", "variance_fraction": max(0, both - trans)},
            {"target": target, "component": "transition_unique", "variance_fraction": max(0, both - cap)},
            {"target": target, "component": "shared", "variance_fraction": max(0, min(cap, trans))},
            {"target": target, "component": "unexplained", "variance_fraction": max(0, 1 - both)},
        ])
    return pd.DataFrame(rows)


def feature_importance(meta: pd.DataFrame, transition: pd.DataFrame, blocks: dict[str, list[str]]) -> pd.DataFrame:
    data = pd.concat([meta[[TARGET]].reset_index(drop=True), transition.reset_index(drop=True)], axis=1).dropna()
    X = data.drop(columns=[TARGET]); y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=53)
    model = RandomForestRegressor(n_estimators=350, min_samples_leaf=5, random_state=53, n_jobs=1).fit(X_train, y_train)
    imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=53, n_jobs=1)
    out = pd.DataFrame({"feature": X.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std})
    block_rows = [{"feature": f"BLOCK::{name}", "importance_mean": float(out[out["feature"].isin(cols)]["importance_mean"].sum()), "importance_std": np.nan} for name, cols in blocks.items()]
    return pd.concat([pd.DataFrame(block_rows), out], ignore_index=True).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def capacity_controlled_effect(meta: pd.DataFrame, capacity: pd.DataFrame, transition: pd.DataFrame) -> pd.DataFrame:
    risk_cols = [c for c in ["largest_drop_fraction", "top3_drop_mass", "drop_gini", "roughness_index"] if c in transition.columns]
    z = transition[risk_cols].replace([np.inf, -np.inf], np.nan)
    risk = ((z - z.mean()) / z.std(ddof=0).replace(0, np.nan)).mean(axis=1)
    data = pd.concat([meta.reset_index(drop=True), capacity[["effective_support"]].reset_index(drop=True), risk.rename("transition_risk")], axis=1).dropna(subset=["effective_support", "transition_risk", TARGET, SURVIVAL, "elevation"])
    data["capacity_band"] = pd.qcut(data["effective_support"].rank(method="first"), 8, labels=False) + 1
    rows = []
    for band, group in data.groupby("capacity_band"):
        low = group[group["transition_risk"] <= group["transition_risk"].quantile(0.25)]
        high = group[group["transition_risk"] >= group["transition_risk"].quantile(0.75)]
        rows.append({"band": int(band), "n_low": len(low), "n_high": len(high), "low_cliffiness": float(low[TARGET].mean()), "high_cliffiness": float(high[TARGET].mean()), "cliffiness_high_minus_low": float(high[TARGET].mean() - low[TARGET].mean()), "survival_high_minus_low": float(high[SURVIVAL].mean() - low[SURVIVAL].mean()), "elevation_high_minus_low": float(high["elevation"].mean() - low["elevation"].mean())})
    return pd.DataFrame(rows)


def transition_embedding(transition: pd.DataFrame) -> np.ndarray:
    X = transition.fillna(transition.median(numeric_only=True)).fillna(0)
    scaled = StandardScaler().fit_transform(X)
    try:
        import umap
        return umap.UMAP(n_neighbors=25, min_dist=0.08, random_state=55).fit_transform(scaled)
    except Exception:
        return PCA(n_components=2, random_state=55).fit_transform(scaled)


def archetypes(meta: pd.DataFrame, transition: pd.DataFrame, matrix: pd.DataFrame, n_clusters: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    X = StandardScaler().fit_transform(transition.fillna(transition.median(numeric_only=True)).fillna(0))
    labels = KMeans(n_clusters=n_clusters, random_state=57, n_init=20).fit_predict(X)
    data = meta.copy(); data["transition_archetype"] = labels
    profiles = data.groupby("transition_archetype", as_index=False).agg(n_runs=(TARGET, "size"), mean_cliffiness=(TARGET, "mean"), mean_survival_auc=(SURVIVAL, "mean"), dominant_family=("model_id", lambda s: str(s.value_counts().idxmax())))
    names = []
    for row in profiles.itertuples(index=False):
        sub = transition.iloc[np.where(labels == row.transition_archetype)[0]]
        if sub["largest_drop_fraction"].mean() > 0.75:
            names.append("Single Collapse")
        elif sub["effective_drop_count"].mean() < 4:
            names.append("Few Collapse")
        elif sub["late_persistence"].mean() > 0.50:
            names.append("Persistent Support")
        elif sub["drop_tail_fraction"].mean() > 0.50:
            names.append("Late Collapse")
        else:
            names.append("Continuous Decay")
    profiles["archetype_name"] = names
    return profiles, labels


def frontier(meta: pd.DataFrame, capacity: pd.DataFrame, transition: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([meta.reset_index(drop=True), capacity[["effective_support"]].reset_index(drop=True), transition[["largest_drop_fraction"]].reset_index(drop=True)], axis=1).dropna()
    data["capacity_bin"] = pd.qcut(data["effective_support"].rank(method="first"), 10, labels=False) + 1
    return data.groupby("capacity_bin", as_index=False).agg(mean_effective_support=("effective_support", "mean"), p90_cliffiness=(TARGET, lambda s: float(s.quantile(.9))), max_cliffiness=(TARGET, "max"), mean_transition_risk=("largest_drop_fraction", "mean"))


def plot_umap(meta: pd.DataFrame, emb: np.ndarray, output: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, [TARGET, SURVIVAL, "model_id"], strict=True):
        vals = meta[col]
        if pd.api.types.is_numeric_dtype(vals):
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=vals, cmap="viridis", s=12, alpha=.7); fig.colorbar(sc, ax=ax, label=col)
        else:
            codes, _ = pd.factorize(vals.astype(str)); ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap="tab20", s=12, alpha=.7)
        ax.set_title(col)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_archetypes(matrix: pd.DataFrame, labels: np.ndarray, thresholds: np.ndarray, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in sorted(set(labels)):
        idx = np.where(labels == label)[0]
        ax.plot(thresholds, matrix.iloc[idx].mean(axis=0), label=f"cluster {label}")
    ax.set_xlabel("threshold"); ax.set_ylabel("R(t)"); ax.set_title("Transition Archetypes"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_effect(effects: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5)); ax.axhline(0, color="black", linewidth=1)
    ax.bar(effects["band"].astype(str), effects["cliffiness_high_minus_low"], color="steelblue")
    ax.set_xlabel("capacity band"); ax.set_ylabel("high-risk - low-risk cliffiness"); ax.set_title("Capacity-Controlled Transition Effect")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_frontier(front: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(front["mean_effective_support"], front["p90_cliffiness"], marker="o", label="p90 cliffiness"); ax.plot(front["mean_effective_support"], front["max_cliffiness"], linestyle="--", label="max cliffiness")
    ax.set_xlabel("effective support"); ax.set_ylabel("cliffiness envelope"); ax.set_title("Transition Frontier Envelope"); ax.legend(); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_importance(importance: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6)); top = importance.head(15).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], color="steelblue"); ax.set_title("Transition Feature Importance"); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def write_report(curve_paths: list[Path], output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, capacity, _, full = load_decomposition_dataset(curve_paths)
    curves = load_curve_files(curve_paths); _, matrix, thresholds = pivot_curves(curves)
    transition, blocks = transition_features_from_curves(matrix, thresholds)
    scores = model_scores(meta, capacity, transition, full)
    part = variance_partition(scores)
    imp = feature_importance(meta, transition, blocks)
    emb = transition_embedding(transition)
    profiles, labels = archetypes(meta, transition, matrix)
    effects = capacity_controlled_effect(meta, capacity, transition)
    front = frontier(meta, capacity, transition)
    cap = score_for(scores, TARGET, "capacity"); both = score_for(scores, TARGET, "capacity_plus_transition"); trans = score_for(scores, TARGET, "transition")
    summary = {"n_runs": int(len(meta)), "transition_r2": trans, "capacity_r2": cap, "capacity_plus_transition_r2": both, "delta_r2_vs_capacity": both - cap, "success": bool((both - cap) > 0.10), "top_block": str(imp[imp["feature"].str.startswith("BLOCK::")].iloc[0]["feature"]).replace("BLOCK::", "")}
    report = output_dir / "accessibility_transition_hypothesis.md"; scores_path = output_dir / "transition_model_scores.csv"; part_path = output_dir / "transition_variance_partition.csv"; imp_path = output_dir / "transition_feature_importance.csv"; summary_path = output_dir / "accessibility_transition_summary.json"
    scores.to_csv(scores_path, index=False); part.to_csv(part_path, index=False); imp.to_csv(imp_path, index=False); summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text("\n".join(["# Accessibility Transition Hypothesis", "", "## Executive Answer", f"Transition incremental R2 over capacity is `{both - cap:.4f}`; success criterion `> 0.10` is {'met' if (both - cap) > 0.10 else 'not met'}.", "", "## Model Scores", _markdown_table(scores), "", "## Variance Partition", _markdown_table(part), "", "## Transition Feature Importance", _markdown_table(imp.head(25)), "", "## Transition Archetypes", _markdown_table(profiles), "", "## Capacity-Controlled Transition Effect", _markdown_table(effects)]) + "\n", encoding="utf-8")
    return (report, scores_path, part_path, imp_path, plot_umap(meta, emb, output_dir / "transition_umap.png"), plot_archetypes(matrix, labels, thresholds, output_dir / "transition_archetypes.png"), plot_effect(effects, output_dir / "capacity_controlled_transition_effect.png"), plot_frontier(front, output_dir / "transition_frontier_envelope.png"), summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", action="append", type=Path, dest="curves", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.curves or [HDDT_CURVES, CLASSICAL_CURVES], args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
