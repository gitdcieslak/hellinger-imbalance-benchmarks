"""Run classical allocators in the accessibility morphology benchmark."""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
for module_name in list(sys.modules):
    if module_name == "hib" or module_name.startswith("hib."):
        del sys.modules[module_name]

from hib.allocation_shape import positive_score_allocation_shape_metrics  # noqa: E402
from hib.models import MODEL_REGISTRY, OptionalDependencyUnavailable, make_model  # noqa: E402
from hib.occupancy import compute_occupancy_metrics, empirical_reachability_curve  # noqa: E402
from hib.synthetic import SyntheticSkewConfig, make_train_test_split  # noqa: E402


DEFAULT_MODELS = ["cart", "hddt", "bagged_hddt", "random_forest", "xgboost", "lightgbm"]
DEFAULT_SKEWS = [25, 100, 500, 1000]
DEFAULT_SEEDS = list(range(20))
THRESHOLDS = [0.50, 0.25, 0.10, 0.05, 0.01]
REACHABILITY_THRESHOLDS = np.linspace(0.0, 1.0, 101).tolist()
METRIC_COLUMNS = [
    "auroc",
    "average_precision",
    "minority_survival_auc",
    "minority_survival_cliffiness",
    "minority_survival_max_drop",
    "minority_survival_effective_drop_count",
    "breadth",
    "effective_breadth",
    "elevation",
    "peak_concentration",
    "positive_unique_score_ratio",
    "positive_quantization_score",
    "positive_histogram_entropy",
    "positive_effective_score_bins",
    "positive_max_bin_mass",
    "positive_top_bin_mass",
    "positive_score_iqr",
    "positive_score_q10_q90_width",
]
OUTPUT_COLUMNS = ["model_id", "seed", "skew_ratio", "minority_count", "fit_failed", "failure_reason", *METRIC_COLUMNS]
REACHABILITY_COLUMNS = ["run_id", "dataset_id", "task_id", "model_id", "seed", "threshold", "minority_reachability", "minority_survival_auc", "minority_survival_cliffiness", "breadth", "elevation"]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_model_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_model_availability(model_ids: list[str], seed: int = 0) -> tuple[list[str], dict[str, str]]:
    available = []
    missing = {}
    for model_id in model_ids:
        if model_id not in MODEL_REGISTRY:
            missing[model_id] = "not in model registry"
            continue
        try:
            make_model(model_id, seed)
        except OptionalDependencyUnavailable as exc:
            missing[model_id] = str(exc)
        except Exception as exc:  # pragma: no cover - defensive availability guard
            missing[model_id] = f"instantiation failed: {exc}"
        else:
            available.append(model_id)
    return available, missing


def prepare_split(seed: int, skew_ratio: int, minority_count: int, test_size: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = SyntheticSkewConfig(
        skew_ratio=int(skew_ratio),
        seed=int(seed),
        separation=2.0,
        minority_count=int(minority_count),
        n_features=6,
        noise=1.0,
        test_size=float(test_size),
    )
    X_train, X_test, y_train, y_test = make_train_test_split(config)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def _positive_class_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.clip(proba[:, 1], 0.0, 1.0)
        if proba.ndim == 2 and proba.shape[1] == 1:
            return np.clip(proba[:, 0], 0.0, 1.0)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    pred = np.asarray(model.predict(X), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def build_result_row(model_id: str, seed: int, skew_ratio: int, minority_count: int, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    s = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
    positive_scores = s[y == 1]
    allocation = positive_score_allocation_shape_metrics(positive_scores)
    occupancy = compute_occupancy_metrics(y, s, thresholds=THRESHOLDS)
    return {
        "model_id": model_id,
        "seed": int(seed),
        "skew_ratio": int(skew_ratio),
        "minority_count": int(minority_count),
        "fit_failed": False,
        "failure_reason": "",
        "auroc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "minority_survival_auc": float(occupancy["minority_survival_auc"]),
        "minority_survival_cliffiness": float(occupancy["minority_survival_cliffiness"]),
        "minority_survival_max_drop": float(occupancy["minority_survival_max_drop"]),
        "minority_survival_effective_drop_count": float(occupancy["minority_survival_effective_drop_count"]),
        "breadth": float(allocation["positive_histogram_entropy"]),
        "effective_breadth": float(allocation["positive_effective_score_bins"]),
        "elevation": float(allocation["positive_top_bin_mass"]),
        "peak_concentration": float(allocation["positive_max_bin_mass"]),
        "positive_unique_score_ratio": float(allocation["positive_unique_score_ratio"]),
        "positive_quantization_score": float(allocation["positive_quantization_score"]),
        "positive_histogram_entropy": float(allocation["positive_histogram_entropy"]),
        "positive_effective_score_bins": float(allocation["positive_effective_score_bins"]),
        "positive_max_bin_mass": float(allocation["positive_max_bin_mass"]),
        "positive_top_bin_mass": float(allocation["positive_top_bin_mass"]),
        "positive_score_iqr": float(allocation["positive_score_iqr"]),
        "positive_score_q10_q90_width": float(allocation["positive_score_q10_q90_width"]),
    }


def build_reachability_rows(row: dict[str, Any], y_true: np.ndarray, y_score: np.ndarray, thresholds: list[float] = REACHABILITY_THRESHOLDS) -> list[dict[str, Any]]:
    run_id = f"classical|skew={row['skew_ratio']}|model={row['model_id']}|seed={row['seed']}"
    task_id = f"synthetic_skew_{row['skew_ratio']}"
    return [
        {
            "run_id": run_id,
            "dataset_id": "synthetic_severe_skew",
            "task_id": task_id,
            "model_id": row["model_id"],
            "seed": int(row["seed"]),
            "threshold": point["threshold"],
            "minority_reachability": point["minority_reachability"],
            "minority_survival_auc": row.get("minority_survival_auc", np.nan),
            "minority_survival_cliffiness": row.get("minority_survival_cliffiness", np.nan),
            "breadth": row.get("breadth", np.nan),
            "elevation": row.get("elevation", np.nan),
        }
        for point in empirical_reachability_curve(y_true, y_score, thresholds)
    ]


def failed_result_row(model_id: str, seed: int, skew_ratio: int, minority_count: int, reason: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_id": model_id,
        "seed": int(seed),
        "skew_ratio": int(skew_ratio),
        "minority_count": int(minority_count),
        "fit_failed": True,
        "failure_reason": reason[:500],
    }
    row.update({column: np.nan for column in METRIC_COLUMNS})
    return row


def run_one(model_id: str, seed: int, skew_ratio: int, minority_count: int, test_size: float = 0.5) -> dict[str, Any]:
    try:
        X_train, X_test, y_train, y_test = prepare_split(seed, skew_ratio, minority_count, test_size=test_size)
        model = make_model(model_id, seed)
        model.fit(X_train, y_train)
        y_score = _positive_class_scores(model, X_test)
        return build_result_row(model_id, seed, skew_ratio, minority_count, y_test, y_score)
    except Exception as exc:
        return failed_result_row(model_id, seed, skew_ratio, minority_count, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")


def run_one_with_reachability(model_id: str, seed: int, skew_ratio: int, minority_count: int, test_size: float = 0.5) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        X_train, X_test, y_train, y_test = prepare_split(seed, skew_ratio, minority_count, test_size=test_size)
        model = make_model(model_id, seed)
        model.fit(X_train, y_train)
        y_score = _positive_class_scores(model, X_test)
        row = build_result_row(model_id, seed, skew_ratio, minority_count, y_test, y_score)
        return row, build_reachability_rows(row, y_test, y_score)
    except Exception as exc:
        return failed_result_row(model_id, seed, skew_ratio, minority_count, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"), []


def write_results(rows: list[dict[str, Any]], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, np.nan) for column in OUTPUT_COLUMNS})
    return output


def completed_keys(output: Path) -> set[tuple[str, int, int]]:
    if not output.exists():
        return set()
    with output.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            (str(row["model_id"]), int(float(row["skew_ratio"])), int(float(row["seed"])))
            for row in reader
            if row.get("model_id") and row.get("skew_ratio") and row.get("seed")
        }


def append_result_row(row: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, np.nan) for column in OUTPUT_COLUMNS})
        handle.flush()


def append_reachability_rows(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REACHABILITY_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, np.nan) for column in REACHABILITY_COLUMNS})
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=parse_model_list, default=DEFAULT_MODELS)
    parser.add_argument("--skew-ratios", type=parse_int_list, default=DEFAULT_SKEWS)
    parser.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--minority-count", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "topology" / "classical_allocator_morphology.csv")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in the output CSV and append new rows.")
    parser.add_argument("--save-reachability-curves", type=Path, default=None)
    args = parser.parse_args()

    available, missing = resolve_model_availability(args.models)
    for model_id, reason in missing.items():
        print(f"skipping {model_id}: {reason}")

    done = completed_keys(args.output) if args.resume else set()
    if not args.resume:
        if args.output.exists():
            args.output.unlink()
        if args.save_reachability_curves is not None and args.save_reachability_curves.exists():
            args.save_reachability_curves.unlink()
    rows_written = 0
    for model_id in available:
        for skew_ratio in args.skew_ratios:
            for seed in args.seeds:
                key = (model_id, int(skew_ratio), int(seed))
                if key in done:
                    print(f"skip: model={model_id} skew={skew_ratio} seed={seed}")
                    continue
                if args.save_reachability_curves is None:
                    row = run_one(model_id, seed, skew_ratio, args.minority_count, test_size=args.test_size)
                    curve_rows = []
                else:
                    row, curve_rows = run_one_with_reachability(model_id, seed, skew_ratio, args.minority_count, test_size=args.test_size)
                append_result_row(row, args.output)
                if args.save_reachability_curves is not None:
                    append_reachability_rows(curve_rows, args.save_reachability_curves)
                rows_written += 1
                status = "failed" if row["fit_failed"] else "ok"
                print(f"{status}: model={model_id} skew={skew_ratio} seed={seed}")

    print(f"wrote {args.output} new_rows={rows_written}")
    if args.save_reachability_curves is not None:
        print(f"wrote {args.save_reachability_curves}")


if __name__ == "__main__":
    main()
