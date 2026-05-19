"""Posterior occupancy figure family for paper-facing analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from hib.arrays import ensure_numpy_array, ensure_numpy_vector
from hib.datasets.legacy_loader import load_legacy_hddt_dataset
from hib.datasets.legacy_registry import LEGACY_HDDT_DATASET_REGISTRY
from hib.models import OptionalDependencyUnavailable, available_model_ids, make_model
from hib.metrics import positive_class_scores
from hib.runner import (
    apply_model_config_params,
    apply_split_dependent_ensemble_params,
    apply_split_dependent_model_params,
    fit_or_skip_model,
)
from hib.splits import generate_stratified_repeated_splits
from hib.thresholds import DEFAULT_THRESHOLDS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def compute_ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    y = np.arange(1, arr.size + 1, dtype=float) / float(arr.size)
    return arr, y


def compute_reachability_curve(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    scores = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    grid = np.asarray(thresholds, dtype=float)
    if scores.size == 0:
        return np.zeros_like(grid)
    return np.asarray([float(np.mean(scores >= t)) for t in grid], dtype=float)


def summarize_unique_posterior_support(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fixed_thresholds: list[float],
    round_decimals: int = 6,
    top_k: int = 5,
) -> dict[str, Any]:
    y = ensure_numpy_vector(y_true).astype(int)
    s = np.clip(ensure_numpy_vector(y_score).astype(float), 0.0, 1.0)
    rounded = np.round(s, int(round_decimals))
    unique_vals, counts = np.unique(rounded, return_counts=True)
    fractions = counts.astype(float) / float(max(1, rounded.size))
    order = np.argsort(fractions)[::-1]
    sorted_frac = fractions[order]
    sorted_vals = unique_vals[order]
    spacing = np.diff(np.sort(unique_vals))
    nonzero_spacing = spacing[spacing > 0]

    pos = s[y == 1]
    neg = s[y == 0]
    out: dict[str, Any] = {
        "n_samples": int(s.size),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "unique_score_count": int(unique_vals.size),
        "normalized_unique_score_count": float(unique_vals.size / float(max(1, s.size))),
        "largest_mass_point": float(sorted_vals[0]) if sorted_vals.size else 0.0,
        "largest_mass_fraction": float(sorted_frac[0]) if sorted_frac.size else 0.0,
        "top_5_mass_fraction": float(np.sum(sorted_frac[: int(top_k)])) if sorted_frac.size else 0.0,
        "min_nonzero_score_spacing": float(np.min(nonzero_spacing)) if nonzero_spacing.size else 0.0,
        "threshold_crossing_count": int(sum(1 for t in fixed_thresholds if np.any(s >= float(t)))),
    }
    for threshold in fixed_thresholds:
        label = f"{float(threshold):.2f}".replace(".", "_")
        out[f"minority_reachability_at_{label}"] = float(np.mean(pos >= float(threshold))) if pos.size else 0.0
        out[f"majority_reachability_at_{label}"] = float(np.mean(neg >= float(threshold))) if neg.size else 0.0
    return out


def _slug(record: dict[str, Any]) -> str:
    dataset_id = str(record["dataset_id"])
    model_id = str(record["model_id"])
    split_id = str(record["split_id"]).replace(" ", "_")
    return f"{dataset_id}_{model_id}_{split_id}"


def generate_class_conditional_ecdf_plots(
    posterior_records: list[dict[str, Any]],
    output_dir: str | Path,
    thresholds: list[float],
    max_plots_per_model: int | None = None,
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    per_model: dict[str, int] = {}

    for rec in posterior_records:
        model_id = str(rec["model_id"])
        per_model.setdefault(model_id, 0)
        if max_plots_per_model is not None and per_model[model_id] >= int(max_plots_per_model):
            continue

        y = ensure_numpy_vector(rec["y_true"]).astype(int)
        s = np.clip(ensure_numpy_vector(rec["y_score"]).astype(float), 0.0, 1.0)
        pos_x, pos_y = compute_ecdf(s[y == 1])
        neg_x, neg_y = compute_ecdf(s[y == 0])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(neg_x, neg_y, where="post", label="majority (y=0)")
        ax.step(pos_x, pos_y, where="post", label="minority (y=1)")
        for t in thresholds:
            ax.axvline(float(t), linestyle="--", color="gray", alpha=0.4)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted positive-class probability")
        ax.set_ylabel("ECDF")
        ax.set_title(f"Class-Conditional ECDF: {rec['dataset_id']} {model_id}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()

        prefix = _slug(rec)
        png = out / f"{prefix}_ecdf.png"
        svg = out / f"{prefix}_ecdf.svg"
        fig.savefig(png, dpi=120)
        fig.savefig(svg)
        plt.close(fig)
        created.extend([png, svg])
        per_model[model_id] += 1
    return created


def generate_threshold_reachability_curves(
    posterior_records: list[dict[str, Any]],
    output_dir: str | Path,
    grid_size: int = 1001,
    max_plots_per_model: int | None = None,
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    grid = np.linspace(0.0, 1.0, int(grid_size))
    per_model: dict[str, int] = {}

    for rec in posterior_records:
        model_id = str(rec["model_id"])
        per_model.setdefault(model_id, 0)
        if max_plots_per_model is not None and per_model[model_id] >= int(max_plots_per_model):
            continue

        y = ensure_numpy_vector(rec["y_true"]).astype(int)
        s = np.clip(ensure_numpy_vector(rec["y_score"]).astype(float), 0.0, 1.0)
        pos_curve = compute_reachability_curve(s[y == 1], grid)
        neg_curve = compute_reachability_curve(s[y == 0], grid)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(grid, pos_curve, label="minority reachability")
        ax.plot(grid, neg_curve, label="majority reachability", alpha=0.8)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Threshold t")
        ax.set_ylabel("Reachability P(score >= t)")
        ax.set_title(f"Threshold Reachability: {rec['dataset_id']} {model_id}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()

        prefix = _slug(rec)
        png = out / f"{prefix}_reachability.png"
        svg = out / f"{prefix}_reachability.svg"
        fig.savefig(png, dpi=120)
        fig.savefig(svg)
        plt.close(fig)
        created.extend([png, svg])
        per_model[model_id] += 1
    return created


def generate_unique_posterior_support_histograms(
    posterior_records: list[dict[str, Any]],
    output_dir: str | Path,
    thresholds: list[float],
    round_decimals: int = 6,
    max_plots_per_model: int | None = None,
) -> tuple[list[Path], pd.DataFrame]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    rows: list[dict[str, Any]] = []
    per_model: dict[str, int] = {}

    for rec in posterior_records:
        y = ensure_numpy_vector(rec["y_true"]).astype(int)
        s = np.clip(ensure_numpy_vector(rec["y_score"]).astype(float), 0.0, 1.0)
        summary = summarize_unique_posterior_support(y, s, thresholds, round_decimals=int(round_decimals))
        row = {
            "dataset_id": rec["dataset_id"],
            "model_id": rec["model_id"],
            "split_id": rec["split_id"],
            **summary,
        }
        rows.append(row)

        model_id = str(rec["model_id"])
        per_model.setdefault(model_id, 0)
        if max_plots_per_model is not None and per_model[model_id] >= int(max_plots_per_model):
            continue

        rounded = np.round(s, int(round_decimals))
        unique_vals, counts = np.unique(rounded, return_counts=True)
        if unique_vals.size > 80:
            idx = np.argsort(counts)[::-1][:80]
            unique_vals = unique_vals[idx]
            counts = counts[idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(unique_vals, counts, width=max(0.003, 1.0 / max(200.0, unique_vals.size * 4.0)))
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Unique posterior value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Unique Posterior Support: {rec['dataset_id']} {model_id}")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        prefix = _slug(rec)
        png = out / f"{prefix}_support.png"
        svg = out / f"{prefix}_support.svg"
        fig.savefig(png, dpi=120)
        fig.savefig(svg)
        plt.close(fig)
        created.extend([png, svg])
        per_model[model_id] += 1

    frame = pd.DataFrame(rows).sort_values(["dataset_id", "model_id", "split_id"]).reset_index(drop=True)
    return created, frame


def posterior_occupancy_summary_to_markdown(
    summary: pd.DataFrame,
    output_dir: str | Path,
    ecdf_paths: list[Path],
    reachability_paths: list[Path],
    support_paths: list[Path],
) -> str:
    model_table = (
        summary.groupby("model_id", as_index=False)
        .agg(
            unique_score_count_mean=("unique_score_count", "mean"),
            largest_mass_fraction_mean=("largest_mass_fraction", "mean"),
            top_5_mass_fraction_mean=("top_5_mass_fraction", "mean"),
            minority_reachability_at_0_05_mean=("minority_reachability_at_0_05", "mean"),
            majority_reachability_at_0_05_mean=("majority_reachability_at_0_05", "mean"),
        )
        .sort_values("model_id")
    )

    table_cols = [
        "model_id",
        "unique_score_count_mean",
        "largest_mass_fraction_mean",
        "top_5_mass_fraction_mean",
        "minority_reachability_at_0_05_mean",
        "majority_reachability_at_0_05_mean",
    ]
    lines = [
        "# Posterior Occupancy Figure Summary",
        "",
        "## Generated Figure Families",
        f"- ECDF figures: {len(ecdf_paths)} files in `{Path(output_dir) / 'ecdf'}`",
        f"- Reachability curves: {len(reachability_paths)} files in `{Path(output_dir) / 'reachability'}`",
        f"- Unique support histograms: {len(support_paths)} files in `{Path(output_dir) / 'support'}`",
        "",
        "## Model-Level Summary",
        "",
        "| " + " | ".join(table_cols) + " |",
        "| " + " | ".join(["---"] * len(table_cols)) + " |",
    ]
    for _, row in model_table.iterrows():
        vals = [row["model_id"]] + [f"{float(row[col]):.4f}" for col in table_cols[1:]]
        lines.append("| " + " | ".join(vals) + " |")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "### CART",
            "- Expected quantized posterior support with high mass concentration and staircase reachability.",
            "### XGBoost",
            "- Expected compressed minority occupancy with cliff-like reachability transitions.",
            "### HDDT",
            "- Expected broader posterior occupancy and smoother minority reachability than CART.",
            "### Bagged HDDT",
            "- Expected broadened support relative to single-tree HDDT with reduced quantization artifacts.",
            "### Random Forest",
            "- Expected discrete but broader posterior occupancy with moderate threshold reachability smoothness.",
            "### LightGBM",
            "- Expected broader occupancy than CART with possible compressed minority bands under severe imbalance.",
            "",
            "Operational interpretation focuses on quantized support, compressed minority occupancy, threshold reachability, operational fragility, and broad posterior occupancy.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_posterior_occupancy_figure_family(
    posterior_records: list[dict[str, Any]],
    output_dir: str | Path,
    thresholds: list[float] | None = None,
    grid_size: int = 1001,
    round_decimals: int = 6,
    max_plots_per_model: int | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    threshold_values = [float(t) for t in (thresholds or list(DEFAULT_THRESHOLDS))]

    ecdf_paths = generate_class_conditional_ecdf_plots(
        posterior_records,
        out / "ecdf",
        thresholds=threshold_values,
        max_plots_per_model=max_plots_per_model,
    )
    reachability_paths = generate_threshold_reachability_curves(
        posterior_records,
        out / "reachability",
        grid_size=int(grid_size),
        max_plots_per_model=max_plots_per_model,
    )
    support_paths, summary = generate_unique_posterior_support_histograms(
        posterior_records,
        out / "support",
        thresholds=threshold_values,
        round_decimals=int(round_decimals),
        max_plots_per_model=max_plots_per_model,
    )

    csv_path = out / "posterior_support_summary.csv"
    summary.to_csv(csv_path, index=False)
    md_path = out / "posterior_occupancy_figure_summary.md"
    md_path.write_text(
        posterior_occupancy_summary_to_markdown(summary, out, ecdf_paths, reachability_paths, support_paths),
        encoding="utf-8",
    )

    return {
        "ecdf_paths": ecdf_paths,
        "reachability_paths": reachability_paths,
        "support_paths": support_paths,
        "summary_csv": csv_path,
        "summary_md": md_path,
    }


def collect_legacy_posterior_records(
    dataset_ids: list[str],
    model_ids: list[str],
    extracted_dir: str | Path,
    model_params: dict[str, dict[str, Any]] | None = None,
    n_repeats: int = 5,
    test_size: float = 0.5,
    split_seed: int = 0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Collect per-split posterior arrays for figure generation."""

    unknown_models = sorted(set(model_ids) - set(available_model_ids()))
    if unknown_models:
        raise ValueError(f"unknown model ids: {', '.join(unknown_models)}")

    records: list[dict[str, Any]] = []
    extracted = Path(extracted_dir)
    for dataset_id in dataset_ids:
        if dataset_id not in LEGACY_HDDT_DATASET_REGISTRY:
            raise ValueError(f"unknown legacy dataset id: {dataset_id}")
        dataset_entry = LEGACY_HDDT_DATASET_REGISTRY[dataset_id]
        X, y, _metadata = load_legacy_hddt_dataset(extracted, dataset_entry)
        X = ensure_numpy_array(X)
        y = ensure_numpy_vector(y)
        splits = generate_stratified_repeated_splits(
            y,
            n_repeats=int(n_repeats),
            test_size=float(test_size),
            random_seed=int(split_seed),
        )
        for split in splits:
            X_train = ensure_numpy_array(X[split.train_idx])
            y_train = ensure_numpy_vector(y[split.train_idx])
            X_test = ensure_numpy_array(X[split.test_idx])
            y_test = ensure_numpy_vector(y[split.test_idx])
            for model_id in model_ids:
                try:
                    model = make_model(model_id, seed=int(seed))
                except OptionalDependencyUnavailable:
                    continue
                model = apply_model_config_params(model, model_id, model_params)
                model = apply_split_dependent_ensemble_params(model, model_id, X_train)
                model = apply_split_dependent_model_params(model, model_id, y_train)
                if not fit_or_skip_model(model, X_train, y_train):
                    continue
                y_score = np.asarray(positive_class_scores(model, X_test), dtype=float)
                records.append(
                    {
                        "source": "legacy",
                        "dataset_id": dataset_id,
                        "model_id": model_id,
                        "repeat_id": split.repeat_id,
                        "split_id": split.split_id,
                        "split_seed": split.split_seed,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "y_true": np.asarray(y_test, dtype=int),
                        "y_score": np.clip(y_score, 0.0, 1.0),
                    }
                )
    return records
