"""Test whether posterior quantization explains quantized allocator cliffiness."""

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

from report_accessibility_support_topology_hypothesis import CAPACITY_COLS, TARGET, entropy, gini, support_topology_features  # noqa: E402
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402
from report_minority_reachability_cohorts import DEFAULT_POSITIVE_SCORES, load_positive_scores  # noqa: E402
from report_support_topology_family_holdout import _fit_predict  # noqa: E402


OUT = ROOT / "reports" / "topology"
QUANTIZED_FAMILIES = {"cart", "random_forest"}
OPTIONAL_FAMILIES = {"hddt", "bagged_hddt"}
TOPOLOGY_BASELINE = OUT / "support_topology_regime_aware_holdout_scores.csv"


def quantization_features(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, group in scores.groupby("run_id", sort=False):
        s = np.clip(group["score"].to_numpy(dtype=float), 0.0, 1.0)
        vals, counts = np.unique(np.round(s, 6), return_counts=True)
        order = np.argsort(vals)
        vals, counts = vals[order], counts[order]
        mass = counts.astype(float) / max(1, counts.sum())
        gaps = np.diff(vals)
        if gaps.size:
            largest_idx = int(np.argmax(gaps))
            max_gap = float(gaps[largest_idx])
            mean_gap = float(gaps.mean())
            median_gap = float(np.median(gaps))
            mass_at_gap = float(mass[largest_idx] + mass[largest_idx + 1])
            mass_above_gap = float(mass[largest_idx + 1 :].sum())
            largest_gap_lower = float(vals[largest_idx])
            largest_gap_upper = float(vals[largest_idx + 1])
        else:
            max_gap = mean_gap = median_gap = mass_above_gap = 0.0
            mass_at_gap = 1.0
            largest_gap_lower = largest_gap_upper = float(vals[0]) if vals.size else 0.0
        hhi = float(np.sum(mass**2)) if mass.size else 0.0
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
                "n_positives": int(len(s)),
                "n_unique_positive_posteriors": int(len(vals)),
                "effective_posterior_alphabet": float(1.0 / max(1e-12, hhi)),
                "posterior_hhi": hhi,
                "largest_posterior_mass": float(mass.max()) if mass.size else 0.0,
                "posterior_entropy": entropy(mass),
                "gini_posterior_mass": gini(mass),
                "mean_posterior_gap": mean_gap,
                "median_posterior_gap": median_gap,
                "max_posterior_gap": max_gap,
                "mass_at_largest_gap": mass_at_gap,
                "mass_above_largest_gap": mass_above_gap,
                "largest_gap_lower": largest_gap_lower,
                "largest_gap_upper": largest_gap_upper,
                "posterior_range": float(vals.max() - vals.min()) if vals.size else 0.0,
                "single_level_positive_support": float(len(vals) == 1),
            }
        )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def quantization_blocks(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    quant_cols = [
        "n_unique_positive_posteriors",
        "effective_posterior_alphabet",
        "posterior_hhi",
        "largest_posterior_mass",
        "posterior_entropy",
        "gini_posterior_mass",
        "mean_posterior_gap",
        "median_posterior_gap",
        "max_posterior_gap",
        "mass_at_largest_gap",
        "mass_above_largest_gap",
        "largest_gap_lower",
        "largest_gap_upper",
        "posterior_range",
        "single_level_positive_support",
    ]
    noncapacity_quant_cols = [c for c in quant_cols if c not in {"n_unique_positive_posteriors", "effective_posterior_alphabet"}]
    capacity = features[CAPACITY_COLS].copy()
    quantization = features[quant_cols].copy()
    quantization_beyond_capacity = features[noncapacity_quant_cols].copy()
    return {
        "capacity": capacity,
        "quantization": quantization,
        "capacity_plus_quantization": pd.concat([capacity.add_prefix("capacity__"), quantization_beyond_capacity.add_prefix("quant__")], axis=1),
    }


def grouped_score(meta: pd.DataFrame, X: pd.DataFrame) -> tuple[float, float]:
    data = pd.concat([meta[[TARGET, "dataset_id"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.median(numeric_only=True)).fillna(0.0)
    groups = data["dataset_id"].astype(str)
    y = data[TARGET].astype(float)
    Xdata = data.drop(columns=[TARGET, "dataset_id"])
    best_r2, best_mae = -np.inf, np.nan
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=101, n_jobs=-1)]:
        pred = cross_val_predict(model, Xdata, y, cv=cv, groups=groups)
        score = float(r2_score(y, pred))
        if score > best_r2:
            best_r2, best_mae = score, float(mean_absolute_error(y, pred))
    return best_r2, best_mae


def explanatory_scores(features: pd.DataFrame, families: set[str]) -> pd.DataFrame:
    data = features[features["model_id"].astype(str).isin(families)].reset_index(drop=True)
    blocks = quantization_blocks(data)
    rows = []
    for spec, block in blocks.items():
        r2, mae = grouped_score(data, block)
        rows.append({"model_spec": spec, "grouped_cv_r2": r2, "grouped_cv_mae": mae, "n_features": block.shape[1]})
    out = pd.DataFrame(rows)
    cap = float(out.loc[out["model_spec"] == "capacity", "grouped_cv_r2"].iloc[0])
    both = float(out.loc[out["model_spec"] == "capacity_plus_quantization", "grouped_cv_r2"].iloc[0])
    out["delta_vs_capacity"] = out["grouped_cv_r2"] - cap
    out["unique_variance_vs_capacity"] = np.where(out["model_spec"] == "capacity_plus_quantization", max(0.0, both - cap), np.nan)
    return out


def cart_rf_transfer(features: pd.DataFrame, topology_baseline: Path = TOPOLOGY_BASELINE) -> pd.DataFrame:
    data = features[features["model_id"].astype(str).isin(QUANTIZED_FAMILIES)].reset_index(drop=True)
    blocks = quantization_blocks(data)
    X = blocks["quantization"].replace([np.inf, -np.inf], np.nan).fillna(blocks["quantization"].median(numeric_only=True)).fillna(0.0).reset_index(drop=True)
    y = data[TARGET].astype(float).reset_index(drop=True)
    rows = []
    baseline = pd.read_csv(topology_baseline) if topology_baseline.exists() else pd.DataFrame()
    for train_family, test_family in [("random_forest", "cart"), ("cart", "random_forest")]:
        train_mask = data["model_id"].astype(str) == train_family
        test_mask = data["model_id"].astype(str) == test_family
        pred = _fit_predict(X.loc[train_mask], y.loc[train_mask], X.loc[test_mask], data.loc[train_mask, "dataset_id"])
        actual = y.loc[test_mask].to_numpy(dtype=float)
        r2 = float(r2_score(actual, pred)) if np.var(actual) > 0 else np.nan
        topo = np.nan
        if not baseline.empty:
            match = baseline[baseline["held_out_family"].astype(str) == test_family]
            if not match.empty:
                topo = float(match["support_topology_r2"].iloc[0])
        rows.append({"train_family": train_family, "test_family": test_family, "quantization_r2": r2, "quantization_mae": float(mean_absolute_error(actual, pred)), "topology_transfer_r2": topo, "improvement_vs_topology": r2 - topo if np.isfinite(topo) else np.nan})
    return pd.DataFrame(rows)


def variance_partition(scores: pd.DataFrame) -> pd.DataFrame:
    lookup = dict(zip(scores["model_spec"], scores["grouped_cv_r2"]))
    cap = max(0.0, float(lookup.get("capacity", 0.0)))
    quant = max(0.0, float(lookup.get("quantization", 0.0)))
    both = max(0.0, float(lookup.get("capacity_plus_quantization", 0.0)))
    return pd.DataFrame([
        {"component": "capacity_unique", "variance_fraction": max(0.0, both - quant)},
        {"component": "quantization_unique", "variance_fraction": max(0.0, both - cap)},
        {"component": "shared", "variance_fraction": max(0.0, min(cap, quant))},
        {"component": "unexplained", "variance_fraction": max(0.0, 1.0 - both)},
    ])


def plot_scatter(features: pd.DataFrame, col: str, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for family, group in features.groupby("model_id"):
        ax.scatter(group[col], group[TARGET], s=14, alpha=0.6, label=family)
    ax.set_xlabel(col)
    ax.set_ylabel("cliffiness")
    ax.set_title(f"Cliffiness vs {col}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def alphabet_matrix(scores: pd.DataFrame, selected_runs: pd.DataFrame, n_bins: int = 101) -> tuple[np.ndarray, list[str]]:
    rows = []
    labels = []
    bins = np.linspace(0, 1, n_bins)
    for row in selected_runs.itertuples(index=False):
        group = scores[scores["run_id"] == row.run_id]
        vals, counts = np.unique(np.round(group["score"].to_numpy(dtype=float), 2), return_counts=True)
        mass = counts.astype(float) / max(1, counts.sum())
        series = pd.Series(mass, index=vals).groupby(level=0).sum()
        aligned = np.zeros(n_bins)
        for value, m in series.items():
            idx = int(np.argmin(np.abs(bins - value)))
            aligned[idx] += float(m)
        rows.append(aligned)
        labels.append(f"{row.model_id}:{row.cliff_group}")
    return np.vstack(rows) if rows else np.zeros((0, n_bins)), labels


def representative_runs(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in features.groupby("model_id"):
        low = group.loc[group[TARGET].idxmin()].copy(); low["cliff_group"] = "low"
        high = group.loc[group[TARGET].idxmax()].copy(); high["cliff_group"] = "high"
        rows.extend([low, high])
    return pd.DataFrame(rows)


def plot_heatmap(scores: pd.DataFrame, reps: pd.DataFrame, output: Path) -> Path:
    matrix, labels = alphabet_matrix(scores, reps)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(labels))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("posterior level x 100")
    ax.set_title("Representative Positive Posterior Alphabets")
    fig.colorbar(im, ax=ax, label="support mass")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def decision(scores: pd.DataFrame, transfer: pd.DataFrame) -> tuple[str, str]:
    both = float(scores.loc[scores["model_spec"] == "capacity_plus_quantization", "delta_vs_capacity"].iloc[0])
    transfer_ok = bool((transfer["improvement_vs_topology"] > 0).all()) if not transfer.empty else False
    if both >= 0.10 and transfer_ok:
        return "strong_support", "Quantization adds explanatory power beyond capacity and improves CART-RF transfer over topology."
    if both >= 0.10 or transfer_ok:
        return "mixed_support", "Quantization helps one criterion but does not fully satisfy both explanatory and transfer tests."
    return "failure", "Quantization does not improve enough beyond capacity or topology-transfer baselines."


def write_report(positive_scores: Path = DEFAULT_POSITIVE_SCORES, output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = load_positive_scores(positive_scores)
    support = support_topology_features(scores)
    quant = quantization_features(scores)
    features = quant.merge(support[["run_id", *CAPACITY_COLS]], on="run_id", how="left")
    quantized = features[features["model_id"].astype(str).isin(QUANTIZED_FAMILIES)].reset_index(drop=True)
    model_scores = explanatory_scores(features, QUANTIZED_FAMILIES)
    transfer = cart_rf_transfer(features)
    partition = variance_partition(model_scores)
    optional = pd.DataFrame()
    available_optional = set(features["model_id"].astype(str)) & OPTIONAL_FAMILIES
    if available_optional:
        optional = explanatory_scores(features, available_optional)
    outcome, text = decision(model_scores, transfer)
    summary = {"outcome": outcome, "outcome_text": text, "n_quantized_runs": int(len(quantized))}
    for row in model_scores.itertuples(index=False):
        summary[f"{row.model_spec}_r2"] = float(row.grouped_cv_r2)
        summary[f"{row.model_spec}_delta_vs_capacity"] = float(row.delta_vs_capacity)
    summary["transfer_improves_both_directions"] = bool((transfer["improvement_vs_topology"] > 0).all())
    report = output_dir / "accessibility_quantization_hypothesis.md"
    summary_path = output_dir / "accessibility_quantization_summary.json"
    features_path = output_dir / "quantization_features.csv"
    scores_path = output_dir / "quantization_model_scores.csv"
    transfer_path = output_dir / "quantization_cart_rf_transfer.csv"
    partition_path = output_dir / "quantization_variance_partition.csv"
    optional_path = output_dir / "quantization_optional_hddt_scores.csv"
    features.to_csv(features_path, index=False); model_scores.to_csv(scores_path, index=False); transfer.to_csv(transfer_path, index=False); partition.to_csv(partition_path, index=False); optional.to_csv(optional_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text("\n".join(["# Accessibility Quantization Hypothesis", "", "## Executive Answer", text, "", "## Quantized Allocator Model Scores", _markdown_table(model_scores), "", "## Variance Partition", _markdown_table(partition), "", "## CART-RF Transfer", _markdown_table(transfer), "", "## Optional HDDT-Like Families", _markdown_table(optional) if not optional.empty else "No optional HDDT-like families were available in this positive-score capture."]) + "\n", encoding="utf-8")
    reps = representative_runs(quantized)
    return (
        report,
        summary_path,
        features_path,
        scores_path,
        transfer_path,
        partition_path,
        optional_path,
        plot_scatter(quantized, "n_unique_positive_posteriors", output_dir / "quantization_cliffiness_vs_alphabet_size.png"),
        plot_scatter(quantized, "largest_posterior_mass", output_dir / "quantization_cliffiness_vs_largest_mass.png"),
        plot_scatter(quantized, "posterior_hhi", output_dir / "quantization_cliffiness_vs_hhi.png"),
        plot_heatmap(scores, reps, output_dir / "quantization_posterior_alphabet_heatmaps.png"),
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
