"""Diagnostics utilities for baseline behavior checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hib.metrics import positive_class_scores
from hib.models import OptionalDependencyUnavailable, make_model
from hib.runner import apply_split_dependent_model_params, compute_binary_class_weight_ratio


def probability_differences(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    """Compute pairwise probability difference diagnostics."""

    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    abs_diff = np.abs(first - second)
    identical_count = int(np.sum(abs_diff == 0.0))
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        corr = 1.0 if np.allclose(first, second) else 0.0
    else:
        corr = float(np.corrcoef(first, second)[0, 1])
    return {
        "max_abs_probability_diff": float(np.max(abs_diff)),
        "correlation": corr,
        "identical_prediction_count": identical_count,
    }


def resolve_lightgbm_variant_params(y_train: np.ndarray, seed: int = 0) -> dict[str, dict[str, Any]]:
    """Resolve fitted parameter views for lightgbm variants."""

    variants = ["lightgbm", "lightgbm_unbalanced", "lightgbm_weighted"]
    resolved: dict[str, dict[str, Any]] = {}
    for model_id in variants:
        model = make_model(model_id, seed=seed)
        model = apply_split_dependent_model_params(model, model_id, y_train)
        params = model.get_params()
        resolved[model_id] = {
            "is_unbalance": params.get("is_unbalance"),
            "scale_pos_weight": params.get("scale_pos_weight"),
            "objective": params.get("objective"),
            "min_child_samples": params.get("min_child_samples"),
            "min_data_in_leaf": params.get("min_data_in_leaf"),
            "verbose": params.get("verbose"),
        }
    return resolved


def lightgbm_weighting_diagnostic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 0,
) -> dict[str, Any]:
    """Run LightGBM weighting equivalence diagnostics on one split."""

    try:
        ratio = compute_binary_class_weight_ratio(y_train)
        params = resolve_lightgbm_variant_params(y_train, seed=seed)
    except OptionalDependencyUnavailable:
        raise

    variants = ["lightgbm", "lightgbm_unbalanced", "lightgbm_weighted"]
    probabilities: dict[str, np.ndarray] = {}
    for model_id in variants:
        model = make_model(model_id, seed=seed)
        model = apply_split_dependent_model_params(model, model_id, y_train)
        model.fit(X_train, y_train)
        probabilities[model_id] = positive_class_scores(model, X_test)

    comparisons = {
        "lightgbm_vs_lightgbm_unbalanced": probability_differences(
            probabilities["lightgbm"],
            probabilities["lightgbm_unbalanced"],
        ),
        "lightgbm_vs_lightgbm_weighted": probability_differences(
            probabilities["lightgbm"],
            probabilities["lightgbm_weighted"],
        ),
        "lightgbm_unbalanced_vs_lightgbm_weighted": probability_differences(
            probabilities["lightgbm_unbalanced"],
            probabilities["lightgbm_weighted"],
        ),
    }

    return {
        "n_positive_train": int(np.sum(np.asarray(y_train) == 1)),
        "n_negative_train": int(np.sum(np.asarray(y_train) == 0)),
        "computed_ratio": float(ratio),
        "resolved_params": params,
        "comparisons": comparisons,
        "n_test": int(np.asarray(y_test).size),
    }


def write_lightgbm_diagnostic_json(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write LightGBM diagnostic JSON report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_lightgbm_diagnostic_markdown(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write LightGBM diagnostic markdown report."""

    lines = [
        "# LightGBM Weighting Diagnostic",
        "",
        f"- Train positives: {report['n_positive_train']}",
        f"- Train negatives: {report['n_negative_train']}",
        f"- Computed ratio (neg/pos): {report['computed_ratio']:.6f}",
        "",
        "## Resolved Params",
        "",
    ]
    for model_id, params in report["resolved_params"].items():
        lines.append(f"### {model_id}")
        lines.append("")
        for key, value in params.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.extend(["## Probability Comparisons", ""])
    for comparison_name, values in report["comparisons"].items():
        lines.append(f"### {comparison_name}")
        lines.append("")
        for key, value in values.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path
