"""Validate support topology under leave-one-family-out splits."""

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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_accessibility_support_topology_hypothesis import (  # noqa: E402
    CAPACITY_COLS,
    TARGET,
    load_full_morphology_for_runs,
    support_topology_features,
)
from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402
from report_minority_reachability_cohorts import DEFAULT_POSITIVE_SCORES, load_positive_scores  # noqa: E402


OUT = ROOT / "reports" / "topology"
ALLOCATOR_REGIMES = {
    "logistic_regression": "continuous_allocator",
    "xgboost": "continuous_allocator",
    "lightgbm": "continuous_allocator",
    "cart": "quantized_allocator",
    "random_forest": "quantized_allocator",
}


def feature_blocks(support: pd.DataFrame, full: pd.DataFrame) -> dict[str, pd.DataFrame]:
    topology_cols = [
        c
        for c in support.columns
        if c
        not in {
            "run_id",
            "dataset_id",
            "task_id",
            "model_id",
            TARGET,
            "minority_survival_auc",
            "breadth",
            "elevation",
            "n_positives",
            *CAPACITY_COLS,
        }
        and pd.api.types.is_numeric_dtype(support[c])
    ]
    return {
        "capacity": support[CAPACITY_COLS].copy(),
        "support_topology": support[topology_cols].copy(),
        "capacity_plus_topology": pd.concat([support[CAPACITY_COLS].add_prefix("capacity__"), support[topology_cols].add_prefix("support__")], axis=1),
        "full_morphology": full.copy(),
    }


def _fit_predict(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, train_groups: pd.Series) -> np.ndarray:
    models = [
        make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)),
        RandomForestRegressor(n_estimators=150, min_samples_leaf=5, random_state=91, n_jobs=-1),
    ]
    best_model = models[0]
    best_score = -np.inf
    groups = train_groups.astype(str).reset_index(drop=True)
    if len(train_y) >= 20 and train_y.var(ddof=0) > 0 and groups.nunique() >= 2:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
        for model in models:
            pred = cross_val_predict(model, train_x, train_y, cv=cv, groups=groups)
            score = float(r2_score(train_y, pred))
            if score > best_score:
                best_score = score
                best_model = model
    best_model.fit(train_x, train_y)
    return np.asarray(best_model.predict(test_x), dtype=float)


def lofo_scores(meta: pd.DataFrame, blocks: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_rows = []
    y = meta[TARGET].astype(float).reset_index(drop=True)
    families = sorted(meta["model_id"].astype(str).unique())
    for family in families:
        test_mask = meta["model_id"].astype(str).reset_index(drop=True) == family
        train_mask = ~test_mask
        if test_mask.sum() < 2 or train_mask.sum() < 10:
            continue
        family_preds = {"held_out_family": family, "n_test": int(test_mask.sum())}
        for spec, X in blocks.items():
            Xclean = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).fillna(0.0).reset_index(drop=True)
            pred = _fit_predict(Xclean.loc[train_mask], y.loc[train_mask], Xclean.loc[test_mask], meta.loc[train_mask, "dataset_id"])
            actual = y.loc[test_mask].to_numpy(dtype=float)
            r2 = float(r2_score(actual, pred)) if np.var(actual) > 0 else np.nan
            mae = float(mean_absolute_error(actual, pred))
            family_preds[f"{spec}_r2"] = r2
            family_preds[f"{spec}_mae"] = mae
            if spec == "support_topology":
                for run_id, actual_value, predicted_value in zip(meta.loc[test_mask, "run_id"], actual, pred, strict=False):
                    pred_rows.append({"held_out_family": family, "run_id": run_id, "actual_cliffiness": float(actual_value), "predicted_cliffiness": float(predicted_value)})
        rows.append(family_preds)
    scores = pd.DataFrame(rows)
    return scores, pd.DataFrame(pred_rows)


def regime_aware_lofo_scores(meta: pd.DataFrame, blocks: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_rows = []
    data = meta.copy().reset_index(drop=True)
    data["allocator_regime"] = data["model_id"].astype(str).map(ALLOCATOR_REGIMES)
    data = data.dropna(subset=["allocator_regime"]).reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    aligned_blocks = {name: block.loc[data.index].reset_index(drop=True) for name, block in blocks.items()}
    y = data[TARGET].astype(float).reset_index(drop=True)
    for regime, regime_meta in data.groupby("allocator_regime", sort=True):
        regime_idx = regime_meta.index.to_numpy()
        families = sorted(regime_meta["model_id"].astype(str).unique())
        if len(families) < 2:
            continue
        for family in families:
            test_mask = pd.Series(False, index=data.index)
            test_mask.loc[regime_idx] = data.loc[regime_idx, "model_id"].astype(str) == family
            train_mask = pd.Series(False, index=data.index)
            train_mask.loc[regime_idx] = ~test_mask.loc[regime_idx]
            if test_mask.sum() < 2 or train_mask.sum() < 10:
                continue
            family_preds = {"allocator_regime": regime, "held_out_family": family, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())}
            for spec, X in aligned_blocks.items():
                Xclean = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).fillna(0.0).reset_index(drop=True)
                pred = _fit_predict(Xclean.loc[train_mask], y.loc[train_mask], Xclean.loc[test_mask], data.loc[train_mask, "dataset_id"])
                actual = y.loc[test_mask].to_numpy(dtype=float)
                r2 = float(r2_score(actual, pred)) if np.var(actual) > 0 else np.nan
                mae = float(mean_absolute_error(actual, pred))
                family_preds[f"{spec}_r2"] = r2
                family_preds[f"{spec}_mae"] = mae
                if spec == "support_topology":
                    for run_id, actual_value, predicted_value in zip(data.loc[test_mask, "run_id"], actual, pred, strict=False):
                        pred_rows.append({"allocator_regime": regime, "held_out_family": family, "run_id": run_id, "actual_cliffiness": float(actual_value), "predicted_cliffiness": float(predicted_value)})
            rows.append(family_preds)
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def score_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in scores.columns if c.endswith("_r2")]:
        values = scores[col].dropna().astype(float)
        rows.append({"metric": col, "mean_r2": float(values.mean()), "median_r2": float(values.median()), "min_r2": float(values.min()), "max_r2": float(values.max())})
    return pd.DataFrame(rows)


def regime_score_summary(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    rows = []
    for regime, group in scores.groupby("allocator_regime", sort=True):
        for col in [c for c in scores.columns if c.endswith("_r2")]:
            values = group[col].dropna().astype(float)
            rows.append({"allocator_regime": regime, "metric": col, "mean_r2": float(values.mean()), "median_r2": float(values.median()), "min_r2": float(values.min()), "max_r2": float(values.max())})
    return pd.DataFrame(rows)


def family_classification(support: pd.DataFrame, block: pd.DataFrame) -> dict[str, float | int]:
    y = support["model_id"].astype(str).reset_index(drop=True)
    X = block.replace([np.inf, -np.inf], np.nan).fillna(block.median(numeric_only=True)).fillna(0.0).reset_index(drop=True)
    min_count = int(y.value_counts().min())
    if y.nunique() < 2 or min_count < 2:
        return {"family_accuracy": np.nan, "family_baseline_accuracy": np.nan, "n_families": int(y.nunique())}
    cv = StratifiedKFold(n_splits=min(5, min_count), shuffle=True, random_state=93)
    pred = cross_val_predict(RandomForestClassifier(n_estimators=150, min_samples_leaf=4, random_state=93, n_jobs=-1), X, y, cv=cv)
    return {"family_accuracy": float(accuracy_score(y, pred)), "family_baseline_accuracy": float(y.value_counts(normalize=True).max()), "n_families": int(y.nunique())}


def decision(summary: pd.DataFrame) -> tuple[str, str]:
    row = summary[summary["metric"] == "support_topology_r2"]
    if row.empty:
        return "unavailable", "Support-topology LOFO scores were unavailable."
    mean_r2 = float(row["mean_r2"].iloc[0])
    min_r2 = float(row["min_r2"].iloc[0])
    if mean_r2 >= 0.75 and min_r2 > 0.60:
        return "strong_support", "Support topology generalizes across held-out model families."
    if mean_r2 >= 0.60:
        return "moderate_support", "Support topology partly generalizes, but some families remain difficult."
    return "failure", "Support topology LOFO performance collapses enough to suggest family-specific signal."


def support_embedding(block: pd.DataFrame) -> np.ndarray:
    X = block.replace([np.inf, -np.inf], np.nan).fillna(block.median(numeric_only=True)).fillna(0.0)
    scaled = StandardScaler().fit_transform(X)
    try:
        import umap

        return umap.UMAP(n_neighbors=25, min_dist=0.08, random_state=95).fit_transform(scaled)
    except Exception:
        return PCA(n_components=2, random_state=95).fit_transform(scaled)


def plot_holdout(scores: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(scores["held_out_family"], scores["support_topology_r2"], color="steelblue")
    ax.axhline(0.60, color="orange", linestyle="--", label="0.60")
    ax.axhline(0.75, color="green", linestyle="--", label="0.75")
    ax.set_ylabel("LOFO topology R2")
    ax.set_title("Family Holdout Performance")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_regime_holdout(scores: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{r}\n{f}" for r, f in zip(scores["allocator_regime"], scores["held_out_family"], strict=False)]
    colors = ["#4c78a8" if r == "continuous_allocator" else "#f58518" for r in scores["allocator_regime"]]
    ax.bar(labels, scores["support_topology_r2"], color=colors)
    ax.axhline(0.60, color="orange", linestyle="--", label="0.60")
    ax.axhline(0.75, color="green", linestyle="--", label="0.75")
    ax.set_ylabel("Within-regime LOFO topology R2")
    ax.set_title("Regime-Aware Family Holdout Performance")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_predictions(predictions: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    for family, group in predictions.groupby("held_out_family"):
        ax.scatter(group["actual_cliffiness"], group["predicted_cliffiness"], s=10, alpha=0.5, label=family)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("actual cliffiness"); ax.set_ylabel("predicted cliffiness")
    ax.set_title("LOFO Topology Predictions")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_embedding(meta: pd.DataFrame, emb: np.ndarray, color_col: str, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    values = meta[color_col]
    if pd.api.types.is_numeric_dtype(values):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, cmap="magma", s=12, alpha=0.7)
        fig.colorbar(sc, ax=ax, label=color_col)
    else:
        codes, _ = pd.factorize(values.astype(str))
        ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap="tab20", s=12, alpha=0.7)
    ax.set_title(f"Support Topology UMAP: {color_col}")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def write_report(positive_scores: Path = DEFAULT_POSITIVE_SCORES, output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = load_positive_scores(positive_scores)
    support = support_topology_features(scores)
    meta = support[["run_id", "dataset_id", "task_id", "model_id", TARGET, "minority_survival_auc", "breadth", "elevation"]].copy()
    try:
        full = load_full_morphology_for_runs(meta["run_id"])
    except Exception:
        full = support[["breadth", "elevation", "minority_survival_auc", "n_score_groups", "group_mass_entropy", "top1_group_mass", "effective_support_groups"]].copy()
    blocks = feature_blocks(support, full)
    lofo, predictions = lofo_scores(meta, blocks)
    summary = score_summary(lofo)
    regime_lofo, regime_predictions = regime_aware_lofo_scores(meta, blocks)
    regime_summary = regime_score_summary(regime_lofo)
    family_cls = family_classification(support, blocks["support_topology"])
    outcome, outcome_text = decision(summary)
    payload = {"outcome": outcome, "outcome_text": outcome_text, **family_cls}
    for row in summary.itertuples(index=False):
        payload[f"{row.metric}_mean"] = float(row.mean_r2)
        payload[f"{row.metric}_min"] = float(row.min_r2)
    for row in regime_summary.itertuples(index=False):
        key = f"regime_{row.allocator_regime}_{row.metric}"
        payload[f"{key}_mean"] = float(row.mean_r2)
        payload[f"{key}_min"] = float(row.min_r2)
    report = output_dir / "support_topology_family_holdout.md"
    summary_path = output_dir / "support_topology_family_holdout_summary.json"
    lofo_path = output_dir / "support_topology_family_holdout_scores.csv"
    pred_path = output_dir / "support_topology_family_holdout_predictions.csv"
    summary_csv = output_dir / "support_topology_family_holdout_summary.csv"
    regime_lofo_path = output_dir / "support_topology_regime_aware_holdout_scores.csv"
    regime_pred_path = output_dir / "support_topology_regime_aware_holdout_predictions.csv"
    regime_summary_path = output_dir / "support_topology_regime_aware_holdout_summary.csv"
    lofo.to_csv(lofo_path, index=False); predictions.to_csv(pred_path, index=False); summary.to_csv(summary_csv, index=False); summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    regime_lofo.to_csv(regime_lofo_path, index=False); regime_predictions.to_csv(regime_pred_path, index=False); regime_summary.to_csv(regime_summary_path, index=False)
    report.write_text("\n".join(["# Support Topology Family Holdout", "", "## Executive Answer", outcome_text, "", "## LOFO Scores", _markdown_table(lofo), "", "## Aggregate Summary", _markdown_table(summary), "", "## Regime-Aware LOFO Scores", _markdown_table(regime_lofo), "", "## Regime-Aware Aggregate Summary", _markdown_table(regime_summary), "", "## Family Classification", _markdown_table(pd.DataFrame([family_cls]))]) + "\n", encoding="utf-8")
    emb = support_embedding(blocks["support_topology"])
    return (
        report,
        summary_path,
        lofo_path,
        pred_path,
        summary_csv,
        regime_lofo_path,
        regime_pred_path,
        regime_summary_path,
        plot_holdout(lofo, output_dir / "support_topology_family_holdout_performance.png"),
        plot_regime_holdout(regime_lofo, output_dir / "support_topology_regime_aware_holdout_performance.png"),
        plot_predictions(predictions, output_dir / "support_topology_family_holdout_predicted_vs_actual.png"),
        plot_embedding(meta, emb, "model_id", output_dir / "support_topology_family_umap.png"),
        plot_embedding(meta, emb, TARGET, output_dir / "support_topology_cliffiness_umap.png"),
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
