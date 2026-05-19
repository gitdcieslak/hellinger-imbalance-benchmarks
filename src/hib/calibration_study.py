"""Calibration interaction study utilities for allocation geometry."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from hib.allocation import allocation_concentration_metrics
from hib.arrays import ensure_numpy_array, ensure_numpy_vector
from hib.datasets.legacy_loader import load_legacy_hddt_dataset
from hib.datasets.legacy_registry import LEGACY_HDDT_DATASET_REGISTRY
from hib.elasticity import compute_threshold_elasticity_intervals, summarize_threshold_elasticity
from hib.metrics import positive_class_scores
from hib.models import OptionalDependencyUnavailable, available_model_ids, make_model
from hib.regimes import infer_regime_labels
from hib.runner import (
    apply_model_config_params,
    apply_split_dependent_ensemble_params,
    apply_split_dependent_model_params,
    fit_or_skip_model,
    package_versions,
)
from hib.splits import generate_stratified_repeated_splits
from hib.synthetic import SyntheticSkewConfig, make_gaussian_skew_dataset
from hib.thresholds import DEFAULT_THRESHOLDS, sweep_thresholds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y = ensure_numpy_vector(y_true).astype(int)
    p = np.clip(ensure_numpy_vector(y_prob).astype(float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        mask = (p >= left) & (p < right if idx < n_bins - 1 else p <= right)
        if not np.any(mask):
            continue
        acc = float(np.mean(y[mask]))
        conf = float(np.mean(p[mask]))
        ece += (np.sum(mask) / y.size) * abs(acc - conf)
    return float(ece)


def reliability_curve_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    y = ensure_numpy_vector(y_true).astype(int)
    p = np.clip(ensure_numpy_vector(y_prob).astype(float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    points: list[dict[str, float]] = []
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        mask = (p >= left) & (p < right if idx < n_bins - 1 else p <= right)
        if not np.any(mask):
            continue
        points.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "bin_center": float((left + right) / 2.0),
                "mean_predicted": float(np.mean(p[mask])),
                "observed_frequency": float(np.mean(y[mask])),
                "count": float(np.sum(mask)),
            }
        )
    return points


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    y = ensure_numpy_vector(y_true).astype(int)
    p = np.clip(ensure_numpy_vector(y_prob).astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(logits, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _platt_transform(cal_scores: np.ndarray, cal_y: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    y_arr = ensure_numpy_vector(cal_y).astype(int)
    if np.unique(y_arr).size < 2:
        return np.clip(ensure_numpy_vector(test_scores).astype(float), 0.0, 1.0)
    p = np.clip(ensure_numpy_vector(cal_scores).astype(float), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(logits, y_arr)
    t = np.clip(ensure_numpy_vector(test_scores).astype(float), 1e-6, 1 - 1e-6)
    tlogits = np.log(t / (1.0 - t)).reshape(-1, 1)
    return lr.predict_proba(tlogits)[:, 1]


def _isotonic_transform(cal_scores: np.ndarray, cal_y: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    y_arr = ensure_numpy_vector(cal_y).astype(int)
    if np.unique(y_arr).size < 2:
        return np.clip(ensure_numpy_vector(test_scores).astype(float), 0.0, 1.0)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(ensure_numpy_vector(cal_scores).astype(float), y_arr)
    return np.clip(iso.predict(ensure_numpy_vector(test_scores).astype(float)), 0.0, 1.0)


def _threshold_records(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in sweep_thresholds(y_true, y_scores, thresholds):
        out.append({"threshold": item["threshold"], **item["metrics"]})
    return out


def _calibration_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> dict[str, Any]:
    slope, intercept = calibration_slope_intercept(y_true, y_scores)
    return {
        "ece": expected_calibration_error(y_true, y_scores),
        "brier_score": float(brier_score_loss(y_true, np.clip(y_scores, 0.0, 1.0))),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "reliability_curve": reliability_curve_points(y_true, y_scores),
    }


def run_calibration_interaction_legacy(
    dataset_ids: list[str],
    model_ids: list[str],
    extracted_dir: str | Path,
    model_params: dict[str, dict[str, Any]] | None = None,
    n_repeats: int = 5,
    test_size: float = 0.5,
    split_seed: int = 0,
    seed: int = 0,
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Run raw/Platt/isotonic calibration interaction study on legacy datasets."""

    unknown = sorted(set(model_ids) - set(available_model_ids()))
    if unknown:
        raise ValueError(f"unknown model ids: {', '.join(unknown)}")
    threshold_values = [float(t) for t in (thresholds or DEFAULT_THRESHOLDS)]
    records: list[dict[str, Any]] = []
    versions = package_versions()
    root = Path(extracted_dir)

    for dataset_id in dataset_ids:
        dataset_entry = LEGACY_HDDT_DATASET_REGISTRY[dataset_id]
        X, y, metadata = load_legacy_hddt_dataset(root, dataset_entry)
        X = ensure_numpy_array(X)
        y = ensure_numpy_vector(y).astype(int)
        splits = generate_stratified_repeated_splits(y, n_repeats=n_repeats, test_size=test_size, random_seed=split_seed)

        for split in splits:
            X_train = ensure_numpy_array(X[split.train_idx])
            y_train = ensure_numpy_vector(y[split.train_idx]).astype(int)
            X_test = ensure_numpy_array(X[split.test_idx])
            y_test = ensure_numpy_vector(y[split.test_idx]).astype(int)
            try:
                X_fit, X_cal, y_fit, y_cal = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.3,
                    random_state=split.split_seed,
                    stratify=y_train,
                )
            except ValueError:
                X_fit, X_cal, y_fit, y_cal = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.3,
                    random_state=split.split_seed,
                    stratify=None,
                )

            for model_id in model_ids:
                try:
                    model = make_model(model_id, seed=seed)
                except OptionalDependencyUnavailable:
                    continue
                model = apply_model_config_params(model, model_id, model_params)
                model = apply_split_dependent_ensemble_params(model, model_id, X_fit)
                model = apply_split_dependent_model_params(model, model_id, y_fit)
                if not fit_or_skip_model(model, X_fit, y_fit):
                    continue

                raw_cal = positive_class_scores(model, X_cal)
                raw_test = positive_class_scores(model, X_test)
                calibrated = {
                    "raw": np.asarray(raw_test, dtype=float),
                    "platt": _platt_transform(raw_cal, y_cal, raw_test),
                    "isotonic": _isotonic_transform(raw_cal, y_cal, raw_test),
                }

                for method, scores in calibrated.items():
                    rec = {
                        "source": "legacy",
                        "dataset_id": dataset_id,
                        "model_id": model_id,
                        "calibration_method": method,
                        "repeat_id": split.repeat_id,
                        "split_id": split.split_id,
                        "split_seed": split.split_seed,
                        "train_n": int(y_train.size),
                        "test_n": int(y_test.size),
                        "train_pos": int(np.sum(y_train == 1)),
                        "train_neg": int(np.sum(y_train == 0)),
                        "test_pos": int(np.sum(y_test == 1)),
                        "test_neg": int(np.sum(y_test == 0)),
                        "threshold_metrics": _threshold_records(y_test, scores, threshold_values),
                        "allocation_metrics": allocation_concentration_metrics(scores),
                        "calibration_metrics": _calibration_metrics(y_test, scores),
                        "dataset_metadata": metadata,
                        "package_versions": versions,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    records.append(rec)
    return records


def build_calibration_summary_tables(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build calibration summary, regime persistence, and threshold tables."""

    threshold_rows: list[dict[str, Any]] = []
    alloc_rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    for rec in records:
        base = {
            "dataset_id": rec["dataset_id"],
            "model_id": rec["model_id"],
            "calibration_method": rec["calibration_method"],
            "split_id": rec["split_id"],
        }
        for row in rec["threshold_metrics"]:
            threshold_rows.append({**base, **row})
        alloc_rows.append({**base, **rec["allocation_metrics"]})
        cal_rows.append({**base, **{k: rec["calibration_metrics"][k] for k in ["ece", "brier_score", "calibration_slope", "calibration_intercept"]}})

    threshold_df = pd.DataFrame(threshold_rows)
    alloc_df = pd.DataFrame(alloc_rows)
    cal_df = pd.DataFrame(cal_rows)

    # elasticity/smoothness from threshold intervals per dataset/model/method
    interval_records: list[pd.DataFrame] = []
    summary_records: list[pd.DataFrame] = []
    for (dataset_id, model_id, method), group in threshold_df.groupby(["dataset_id", "model_id", "calibration_method"], sort=True):
        agg = (
            group.groupby("threshold", as_index=False)[["precision", "recall", "f1", "balanced_accuracy"]]
            .mean()
            .rename(
                columns={
                    "precision": "precision_mean",
                    "recall": "recall_mean",
                    "f1": "f1_mean",
                    "balanced_accuracy": "balanced_accuracy_mean",
                }
            )
        )
        agg["dataset_id"] = dataset_id
        agg["model_id"] = model_id
        from hib.elasticity import compute_threshold_elasticity_intervals, summarize_threshold_elasticity

        intervals = compute_threshold_elasticity_intervals(agg)
        intervals["calibration_method"] = method
        interval_records.append(intervals)
        s = summarize_threshold_elasticity(intervals)
        s["calibration_method"] = method
        summary_records.append(s)

    elasticity_summary = pd.concat(summary_records, ignore_index=True) if summary_records else pd.DataFrame()

    merged = (
        cal_df.groupby(["dataset_id", "model_id", "calibration_method"], as_index=False)
        .mean(numeric_only=True)
        .merge(
            alloc_df.groupby(["dataset_id", "model_id", "calibration_method"], as_index=False).mean(numeric_only=True),
            on=["dataset_id", "model_id", "calibration_method"],
            how="left",
        )
        .merge(
            elasticity_summary[["dataset_id", "model_id", "calibration_method", "mean_abs_recall_elasticity", "mean_abs_precision_elasticity", "operational_smoothness_index", "max_recall_jump"]],
            on=["dataset_id", "model_id", "calibration_method"],
            how="left",
        )
    )

    # model-level calibration summary table
    model_summary = (
        merged.groupby(["model_id", "calibration_method"], as_index=False)
        .mean(numeric_only=True)
        .rename(
            columns={
                "ece": "mean_ece",
                "brier_score": "mean_brier",
                "mean_abs_recall_elasticity": "mean_recall_elasticity",
                "operational_smoothness_index": "mean_smoothness",
            }
        )
    )

    # regime persistence heuristic per method/model
    regime_input = (
        merged.groupby(["model_id", "calibration_method"], as_index=False)
        .agg(
            mean_histogram_entropy=("histogram_entropy", "mean"),
            mean_effective_support_size=("effective_support_size", "mean"),
            mean_gini_coefficient=("gini_coefficient", "mean"),
            mean_fraction_below_0_01=("fraction_scores_below_0_01", "mean"),
            mean_fraction_below_0_05=("fraction_scores_below_0_05", "mean"),
            mean_recall_elasticity=("mean_abs_recall_elasticity", "mean"),
            mean_precision_elasticity=("mean_abs_precision_elasticity", "mean"),
            mean_operational_smoothness_index=("operational_smoothness_index", "mean"),
            mean_max_recall_jump=("max_recall_jump", "mean"),
        )
    )
    regime_labeled = infer_regime_labels(regime_input)
    regime_persistence = regime_labeled[["model_id", "calibration_method", "inferred_regime"]]
    return model_summary, regime_persistence, merged


def write_calibration_study_artifacts(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "calibration_study_records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")

    summary, persistence, merged = build_calibration_summary_tables(records)
    summary_path = output_dir / "calibration_summary_table.csv"
    persistence_path = output_dir / "regime_persistence_table.csv"
    merged_path = output_dir / "calibration_interaction_metrics.csv"
    summary.to_csv(summary_path, index=False)
    persistence.to_csv(persistence_path, index=False)
    merged.to_csv(merged_path, index=False)

    md_path = output_dir / "calibration_interpretation_summary.md"
    lines = [
        "# Calibration Interaction Study",
        "",
        "Calibration effects are summarized across raw, Platt, and isotonic probability outputs.",
        "",
        "## Key Readout",
        "- Compare `mean_ece` and `mean_brier` across calibration methods.",
        "- Compare `mean_recall_elasticity` and `mean_smoothness` to evaluate operational geometry persistence.",
        "- Compare `inferred_regime` transitions in `regime_persistence_table.csv`.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "records_jsonl": jsonl_path,
        "summary_csv": summary_path,
        "regime_persistence_csv": persistence_path,
        "interaction_metrics_csv": merged_path,
        "interpretation_md": md_path,
    }


def plot_reliability_diagrams(records: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in records:
        for point in rec["calibration_metrics"]["reliability_curve"]:
            rows.append(
                {
                    "model_id": rec["model_id"],
                    "method": rec["calibration_method"],
                    "bin_center": point["bin_center"],
                    "mean_predicted": point["mean_predicted"],
                    "observed_frequency": point["observed_frequency"],
                }
            )
    frame = pd.DataFrame(rows)
    created: list[Path] = []
    for model_id, group in frame.groupby("model_id", sort=True):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        for method, mg in group.groupby("method", sort=True):
            g = mg.groupby("bin_center", as_index=False)[["mean_predicted", "observed_frequency"]].mean()
            ax.plot(g["mean_predicted"], g["observed_frequency"], marker="o", label=method)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"Reliability Diagram: {model_id}")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        png = output_dir / f"reliability_{model_id}.png"
        svg = output_dir / f"reliability_{model_id}.svg"
        fig.savefig(png, dpi=120)
        fig.savefig(svg)
        plt.close(fig)
        created.extend([png, svg])
    return created


def plot_pre_post_threshold_curves(records: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in records:
        for metric_row in rec["threshold_metrics"]:
            rows.append(
                {
                    "dataset_id": rec["dataset_id"],
                    "model_id": rec["model_id"],
                    "method": rec["calibration_method"],
                    "threshold": metric_row["threshold"],
                    "recall": metric_row["recall"],
                    "precision": metric_row["precision"],
                }
            )
    frame = pd.DataFrame(rows)
    created: list[Path] = []
    for (dataset_id, model_id), group in frame.groupby(["dataset_id", "model_id"], sort=True):
        for metric in ["recall", "precision"]:
            fig, ax = plt.subplots(figsize=(7, 5))
            for method, mg in group.groupby("method", sort=True):
                agg = mg.groupby("threshold", as_index=False)[metric].mean().sort_values("threshold", ascending=False)
                ax.plot(agg["threshold"], agg[metric], marker="o", label=method)
            ax.set_xlim(0.5, 0.01)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Threshold")
            ax.set_ylabel(metric.title())
            ax.set_title(f"{dataset_id} {model_id}: {metric.title()} vs Threshold")
            ax.grid(True, alpha=0.35)
            ax.legend(loc="best")
            fig.tight_layout()
            png = output_dir / f"{dataset_id}_{model_id}_{metric}_pre_post.png"
            svg = output_dir / f"{dataset_id}_{model_id}_{metric}_pre_post.svg"
            fig.savefig(png, dpi=120)
            fig.savefig(svg)
            plt.close(fig)
            created.extend([png, svg])
    return created


def plot_elasticity_shift(records: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, _, merged = build_calibration_summary_tables(records)
    raw = summary[summary["calibration_method"] == "raw"][ ["model_id", "mean_recall_elasticity"] ]
    out_paths: list[Path] = []
    for method in ["platt", "isotonic"]:
        cmp = summary[summary["calibration_method"] == method][["model_id", "mean_recall_elasticity"]].merge(raw, on="model_id", suffixes=("_post", "_pre"))
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(cmp["mean_recall_elasticity_pre"], cmp["mean_recall_elasticity_post"])
        for _, row in cmp.iterrows():
            ax.annotate(str(row["model_id"]), (row["mean_recall_elasticity_pre"], row["mean_recall_elasticity_post"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        maxv = float(max(cmp["mean_recall_elasticity_pre"].max(), cmp["mean_recall_elasticity_post"].max(), 1.0))
        ax.plot([0, maxv], [0, maxv], linestyle="--", color="gray")
        ax.set_xlabel("Pre-calibration recall elasticity")
        ax.set_ylabel(f"{method.title()} recall elasticity")
        ax.set_title(f"Elasticity Shift: raw vs {method}")
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        png = output_dir / f"elasticity_shift_{method}.png"
        svg = output_dir / f"elasticity_shift_{method}.svg"
        fig.savefig(png, dpi=120)
        fig.savefig(svg)
        plt.close(fig)
        out_paths.extend([png, svg])

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    pivot = merged.groupby(["model_id", "calibration_method"], as_index=False).mean(numeric_only=True)
    for method, mg in pivot.groupby("calibration_method", sort=True):
        ax2.scatter(mg["histogram_entropy"], mg["mean_abs_recall_elasticity"], label=method)
    ax2.set_xlabel("Entropy")
    ax2.set_ylabel("Recall elasticity")
    ax2.set_title("Entropy vs Recall Elasticity after Calibration")
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="best")
    fig2.tight_layout()
    png2 = output_dir / "entropy_vs_elasticity_post_calibration.png"
    svg2 = output_dir / "entropy_vs_elasticity_post_calibration.svg"
    fig2.savefig(png2, dpi=120)
    fig2.savefig(svg2)
    plt.close(fig2)
    out_paths.extend([png2, svg2])
    return out_paths
