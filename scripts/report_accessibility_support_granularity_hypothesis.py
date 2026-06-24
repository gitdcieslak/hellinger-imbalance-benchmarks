"""Test whether support granularity mediates accessibility cliffiness."""

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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_accessibility_quantization_hypothesis import QUANTIZED_FAMILIES, quantization_blocks, quantization_features  # noqa: E402
from report_accessibility_support_topology_hypothesis import CAPACITY_COLS, TARGET, entropy, gini  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402
from report_minority_reachability_cohorts import DEFAULT_POSITIVE_SCORES, load_positive_scores  # noqa: E402
from report_support_topology_family_holdout import _fit_predict  # noqa: E402


OUT = ROOT / "reports" / "topology"
QUANTIZATION_TRANSFER = OUT / "quantization_cart_rf_transfer.csv"


GRANULARITY_COLS = [
    "n_support_groups",
    "effective_support_groups",
    "group_mass_entropy",
    "top1_group_mass",
    "group_mass_hhi",
    "support_redundancy",
    "mass_outside_top3",
    "mass_outside_top5",
    "support_fragility_index",
    "mean_group_mass",
    "median_group_mass",
    "largest_to_median_ratio",
    "group_mass_gini",
]


def _top_mass(mass: np.ndarray, k: int) -> float:
    ordered = np.sort(np.asarray(mass, dtype=float))[::-1]
    return float(ordered[: min(k, len(ordered))].sum()) if ordered.size else 0.0


def granularity_features(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, group in scores.groupby("run_id", sort=False):
        vals, counts = np.unique(np.round(np.clip(group["score"].to_numpy(dtype=float), 0, 1), 6), return_counts=True)
        mass = counts.astype(float) / max(1, counts.sum())
        hhi = float(np.sum(mass**2)) if mass.size else 0.0
        eff = float(1.0 / max(1e-12, hhi))
        top1 = _top_mass(mass, 1)
        first = group.iloc[0]
        rows.append(
            {
                "run_id": run_id,
                "dataset_id": first["dataset_id"],
                "task_id": first["task_id"],
                "model_id": first["model_id"],
                TARGET: float(first[TARGET]),
                "minority_survival_auc": float(first.get("minority_survival_auc", np.nan)),
                "breadth": float(first.get("breadth", np.nan)),
                "elevation": float(first.get("elevation", np.nan)),
                "n_positives": int(len(group)),
                "n_support_groups": int(len(vals)),
                "n_score_groups": int(len(vals)),
                "effective_support_groups": eff,
                "support_unique_score_ratio": float(len(vals) / max(1, len(group))),
                "group_mass_entropy": entropy(mass),
                "top1_group_mass": top1,
                "group_mass_hhi": hhi,
                "support_redundancy": float(1.0 - top1),
                "mass_outside_top3": float(1.0 - _top_mass(mass, 3)),
                "mass_outside_top5": float(1.0 - _top_mass(mass, 5)),
                "support_fragility_index": float(top1 / max(1e-12, eff)),
                "mean_group_mass": float(mass.mean()) if mass.size else 0.0,
                "median_group_mass": float(np.median(mass)) if mass.size else 0.0,
                "largest_to_median_ratio": float(top1 / max(1e-12, np.median(mass))) if mass.size else 0.0,
                "group_mass_gini": gini(mass),
            }
        )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def feature_blocks(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    capacity = features[CAPACITY_COLS].copy()
    granularity = features[GRANULARITY_COLS].copy()
    granularity_beyond_capacity = features[[c for c in GRANULARITY_COLS if c not in CAPACITY_COLS]].copy()
    return {
        "capacity": capacity,
        "granularity": granularity,
        "capacity_plus_granularity": pd.concat([capacity.add_prefix("capacity__"), granularity_beyond_capacity.add_prefix("granularity__")], axis=1),
    }


def _clean(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)


def grouped_score(meta: pd.DataFrame, X: pd.DataFrame, target: str = TARGET) -> tuple[float, float]:
    data = pd.concat([meta[[target, "dataset_id"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    data = _clean(data)
    groups = data["dataset_id"].astype(str)
    y = data[target].astype(float)
    Xdata = data.drop(columns=[target, "dataset_id"])
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    best_r2, best_mae = -np.inf, np.nan
    for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=111, n_jobs=-1)]:
        pred = cross_val_predict(model, Xdata, y, cv=cv, groups=groups)
        score = float(r2_score(y, pred))
        if score > best_r2:
            best_r2, best_mae = score, float(mean_absolute_error(y, pred))
    return best_r2, best_mae


def explanatory_scores(features: pd.DataFrame, families: set[str], label: str) -> pd.DataFrame:
    data = features[features["model_id"].astype(str).isin(families)].reset_index(drop=True)
    rows = []
    for spec, block in feature_blocks(data).items():
        r2, mae = grouped_score(data, block)
        rows.append({"analysis_set": label, "model_spec": spec, "grouped_cv_r2": r2, "grouped_cv_mae": mae, "n_features": block.shape[1]})
    out = pd.DataFrame(rows)
    cap = float(out.loc[out["model_spec"] == "capacity", "grouped_cv_r2"].iloc[0])
    out["delta_vs_capacity"] = out["grouped_cv_r2"] - cap
    out["unique_variance_vs_capacity"] = np.where(out["model_spec"] == "capacity_plus_granularity", np.maximum(0.0, out["grouped_cv_r2"] - cap), np.nan)
    return out


def replacement_test(features: pd.DataFrame, quant_features: pd.DataFrame) -> pd.DataFrame:
    data = features[features["model_id"].astype(str).isin(QUANTIZED_FAMILIES)].reset_index(drop=True)
    quant = quant_features[quant_features["model_id"].astype(str).isin(QUANTIZED_FAMILIES)].reset_index(drop=True)
    q_blocks = quantization_blocks(quant)
    g_blocks = feature_blocks(data)
    rows = []
    specs = {
        "capacity_plus_quantization": q_blocks["capacity_plus_quantization"],
        "capacity_plus_granularity": g_blocks["capacity_plus_granularity"],
    }
    for spec, block in specs.items():
        r2, mae = grouped_score(data, block)
        rows.append({"model_spec": spec, "grouped_cv_r2": r2, "grouped_cv_mae": mae, "n_features": block.shape[1]})
    out = pd.DataFrame(rows)
    q = float(out.loc[out["model_spec"] == "capacity_plus_quantization", "grouped_cv_r2"].iloc[0])
    out["delta_vs_capacity_plus_quantization"] = out["grouped_cv_r2"] - q
    return out


def cart_rf_transfer(features: pd.DataFrame, baseline_path: Path = QUANTIZATION_TRANSFER) -> pd.DataFrame:
    data = features[features["model_id"].astype(str).isin(QUANTIZED_FAMILIES)].reset_index(drop=True)
    X = _clean(data[GRANULARITY_COLS]).reset_index(drop=True)
    y = data[TARGET].astype(float).reset_index(drop=True)
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    rows = []
    for train_family, test_family in [("random_forest", "cart"), ("cart", "random_forest")]:
        train_mask = data["model_id"].astype(str) == train_family
        test_mask = data["model_id"].astype(str) == test_family
        pred = _fit_predict(X.loc[train_mask], y.loc[train_mask], X.loc[test_mask], data.loc[train_mask, "dataset_id"])
        actual = y.loc[test_mask].to_numpy(dtype=float)
        quant_r2 = np.nan
        if not baseline.empty:
            match = baseline[(baseline["train_family"].astype(str) == train_family) & (baseline["test_family"].astype(str) == test_family)]
            if not match.empty:
                quant_r2 = float(match["quantization_r2"].iloc[0])
        r2 = float(r2_score(actual, pred)) if np.var(actual) > 0 else np.nan
        rows.append({"train_family": train_family, "test_family": test_family, "granularity_r2": r2, "granularity_mae": float(mean_absolute_error(actual, pred)), "quantization_transfer_r2": quant_r2, "improvement_vs_quantization": r2 - quant_r2 if np.isfinite(quant_r2) else np.nan})
    return pd.DataFrame(rows)


def mediation_scores(features: pd.DataFrame, families: set[str], label: str) -> pd.DataFrame:
    data = features[features["model_id"].astype(str).isin(families)].reset_index(drop=True)
    gran = _clean(data[GRANULARITY_COLS]).reset_index(drop=True)
    cap = _clean(data[CAPACITY_COLS]).reset_index(drop=True)
    groups = data["dataset_id"].astype(str)
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=113, n_jobs=-1))
    cap_pred = cross_val_predict(model, gran, cap, cv=cv, groups=groups)
    cap_r2 = float(r2_score(cap, cap_pred, multioutput="variance_weighted"))
    cap_cliff_r2, cap_cliff_mae = grouped_score(data, cap)
    gran_cliff_r2, gran_cliff_mae = grouped_score(data, gran)
    return pd.DataFrame([
        {"analysis_set": label, "path": "granularity_to_capacity", "r2": cap_r2, "mae": np.nan},
        {"analysis_set": label, "path": "capacity_to_cliffiness", "r2": cap_cliff_r2, "mae": cap_cliff_mae},
        {"analysis_set": label, "path": "granularity_to_cliffiness", "r2": gran_cliff_r2, "mae": gran_cliff_mae},
    ])


def plot_scatter(features: pd.DataFrame, col: str, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for family, group in features.groupby("model_id"):
        ax.scatter(group[col], group[TARGET], s=14, alpha=0.6, label=family)
    ax.set_xlabel(col); ax.set_ylabel("cliffiness"); ax.set_title(f"Cliffiness vs {col}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_frontier(features: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for family, group in features.groupby("model_id"):
        ax.scatter(group["effective_support_groups"], group[TARGET], s=12, alpha=0.5, label=family)
    ax.set_xlabel("effective_support_groups"); ax.set_ylabel("cliffiness"); ax.set_title("Support Granularity Frontier")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def decision(scores: pd.DataFrame, replacement: pd.DataFrame, transfer: pd.DataFrame) -> tuple[str, str]:
    quantized = scores[(scores["analysis_set"] == "quantized") & (scores["model_spec"] == "capacity_plus_granularity")]
    delta = float(quantized["delta_vs_capacity"].iloc[0]) if not quantized.empty else np.nan
    repl = dict(zip(replacement["model_spec"], replacement["grouped_cv_r2"]))
    replacement_ok = repl.get("capacity_plus_granularity", -np.inf) >= repl.get("capacity_plus_quantization", np.inf)
    transfer_ok = bool((transfer["improvement_vs_quantization"] > 0).all()) if not transfer.empty else False
    if delta >= 0.05 and replacement_ok and transfer_ok:
        return "strong_support", "Granularity adds unique signal beyond capacity, replaces quantization, and improves CART-RF transfer."
    if replacement_ok or transfer_ok or delta >= 0.05:
        return "mixed_support", "Granularity captures part of the quantized allocator signal, but not all success criteria are met."
    return "failure", "Granularity does not add enough beyond capacity or quantization baselines."


def write_report(positive_scores: Path = DEFAULT_POSITIVE_SCORES, output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = load_positive_scores(positive_scores)
    features = granularity_features(scores)
    quant_features = quantization_features(scores).merge(features[["run_id", *CAPACITY_COLS]], on="run_id", how="left")
    quantized_scores = explanatory_scores(features, QUANTIZED_FAMILIES, "quantized")
    all_scores = explanatory_scores(features, set(features["model_id"].astype(str).unique()), "all_families")
    model_scores = pd.concat([quantized_scores, all_scores], ignore_index=True)
    replacement = replacement_test(features, quant_features)
    transfer = cart_rf_transfer(features)
    mediation = pd.concat([mediation_scores(features, QUANTIZED_FAMILIES, "quantized"), mediation_scores(features, set(features["model_id"].astype(str).unique()), "all_families")], ignore_index=True)
    outcome, text = decision(model_scores, replacement, transfer)
    summary = {"outcome": outcome, "outcome_text": text, "n_runs": int(len(features)), "n_quantized_runs": int(features["model_id"].astype(str).isin(QUANTIZED_FAMILIES).sum())}
    for row in model_scores.itertuples(index=False):
        summary[f"{row.analysis_set}_{row.model_spec}_r2"] = float(row.grouped_cv_r2)
        summary[f"{row.analysis_set}_{row.model_spec}_delta_vs_capacity"] = float(row.delta_vs_capacity)
    summary["granularity_replaces_quantization"] = bool(dict(zip(replacement["model_spec"], replacement["grouped_cv_r2"])).get("capacity_plus_granularity", -np.inf) >= dict(zip(replacement["model_spec"], replacement["grouped_cv_r2"])).get("capacity_plus_quantization", np.inf))
    summary["granularity_transfer_improves_quantization"] = bool((transfer["improvement_vs_quantization"] > 0).all())
    report = output_dir / "accessibility_support_granularity_hypothesis.md"
    summary_path = output_dir / "accessibility_support_granularity_summary.json"
    features_path = output_dir / "support_granularity_features.csv"
    scores_path = output_dir / "support_granularity_model_scores.csv"
    replacement_path = output_dir / "support_granularity_quantization_replacement.csv"
    transfer_path = output_dir / "support_granularity_cart_rf_transfer.csv"
    mediation_path = output_dir / "support_granularity_mediation.csv"
    features.to_csv(features_path, index=False); model_scores.to_csv(scores_path, index=False); replacement.to_csv(replacement_path, index=False); transfer.to_csv(transfer_path, index=False); mediation.to_csv(mediation_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text("\n".join(["# Accessibility Support Granularity Hypothesis", "", "## Executive Answer", text, "", "## Explanatory Power", _markdown_table(model_scores), "", "## Quantization Replacement", _markdown_table(replacement), "", "## CART-RF Transfer", _markdown_table(transfer), "", "## Capacity Mediation", _markdown_table(mediation)]) + "\n", encoding="utf-8")
    quantized = features[features["model_id"].astype(str).isin(QUANTIZED_FAMILIES)].reset_index(drop=True)
    return (
        report,
        summary_path,
        features_path,
        scores_path,
        replacement_path,
        transfer_path,
        mediation_path,
        plot_scatter(quantized, "effective_support_groups", output_dir / "granularity_cliffiness_vs_effective_groups.png"),
        plot_scatter(quantized, "top1_group_mass", output_dir / "granularity_cliffiness_vs_top1_mass.png"),
        plot_scatter(quantized, "support_redundancy", output_dir / "granularity_cliffiness_vs_redundancy.png"),
        plot_frontier(features, output_dir / "support_granularity_frontier.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-scores", type=Path, default=DEFAULT_POSITIVE_SCORES)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.positive_scores, args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
