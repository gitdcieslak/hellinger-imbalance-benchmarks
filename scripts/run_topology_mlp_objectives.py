"""Run multi-seed topology comparison for MLP objectives."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hib.allocation import allocation_concentration_metrics  # noqa: E402
from hib.allocation_shape import positive_score_allocation_shape_metrics  # noqa: E402
from hib.metrics import positive_class_scores  # noqa: E402
from hib.models import available_model_ids, make_model  # noqa: E402
from hib.occupancy import compute_occupancy_metrics  # noqa: E402
from hib.synthetic import SyntheticSkewConfig, make_train_test_split  # noqa: E402
from hib.topology import (  # noqa: E402
    compute_minority_support_radius_topology,
    compute_minority_support_topology,
    compute_positive_distance_quantile_radii,
)
from run_topology_smoke import build_topology_rows, extract_penultimate_embeddings_mlp  # noqa: E402


OUTPUT_COLUMNS = [
    "dataset_id",
    "model_name",
    "objective",
    "seed",
    "split_id",
    "topology_mode",
    "k",
    "effective_k",
    "k_capped",
    "radius_quantile",
    "radius",
    "auroc",
    "average_precision",
    "persistence",
    "minority_survival_auc",
    "minority_survival_max_drop",
    "minority_survival_total_variation",
    "minority_survival_drop_entropy",
    "minority_survival_effective_drop_count",
    "minority_survival_cliffiness",
    "positive_score_entropy",
    "positive_unique_score_ratio",
    "positive_quantization_score",
    "positive_histogram_entropy",
    "positive_effective_score_bins",
    "positive_max_bin_mass",
    "positive_top_bin_mass",
    "positive_score_iqr",
    "positive_score_q10_q90_width",
    "positive_score_gini_or_concentration_index",
    "n_positive",
    "n_components",
    "giant_component_fraction",
    "mean_component_size",
    "median_component_size",
    "component_entropy",
    "isolated_positive_fraction",
    "meets_min_auroc",
    "fit_failed",
    "failure_reason",
]


OBJECTIVE_CANDIDATES = [
    ("mlp_bce", "bce"),
    ("mlp_bce_dropout_0_1", "bce_dropout_0_1"),
    ("mlp_bce_dropout_0_3", "bce_dropout_0_3"),
    ("mlp_weighted_bce", "weighted_bce"),
    ("mlp_weighted_bce_dropout_0_1", "weighted_bce_dropout_0_1"),
    ("mlp_weighted_bce_dropout_0_3", "weighted_bce_dropout_0_3"),
    ("mlp_oversampled_bce", "oversampled_bce"),
]

OBJECTIVE_ALIASES = {
    "mlp_weighted_bce": "mlp_weighted",
    "mlp_oversampled_bce": "mlp_oversampled",
}


def resolve_objective_model_ids() -> list[tuple[str, str]]:
    available = set(available_model_ids())
    selected: list[tuple[str, str]] = []
    for preferred_id, objective_name in OBJECTIVE_CANDIDATES:
        if preferred_id in available:
            selected.append((preferred_id, objective_name))
            continue
        alias = OBJECTIVE_ALIASES.get(preferred_id)
        if alias and alias in available:
            selected.append((alias, objective_name))
    return selected


def _base_failure_row(
    *,
    dataset_id: str,
    model_name: str,
    objective: str,
    seed: int,
    topology_mode: str,
    k: object,
    effective_k: object,
    k_capped: object,
    radius_quantile: object,
    radius: object,
    n_positive: int,
    reason: str,
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "model_name": model_name,
        "objective": objective,
        "seed": int(seed),
        "split_id": 0,
        "topology_mode": topology_mode,
        "k": k,
        "effective_k": effective_k,
        "k_capped": k_capped,
        "radius_quantile": radius_quantile,
        "radius": radius,
        "auroc": float("nan"),
        "average_precision": float("nan"),
        "persistence": float("nan"),
        "minority_survival_auc": float("nan"),
        "minority_survival_max_drop": float("nan"),
        "minority_survival_total_variation": float("nan"),
        "minority_survival_drop_entropy": float("nan"),
        "minority_survival_effective_drop_count": float("nan"),
        "minority_survival_cliffiness": float("nan"),
        "positive_score_entropy": float("nan"),
        "positive_unique_score_ratio": float("nan"),
        "positive_quantization_score": float("nan"),
        "positive_histogram_entropy": float("nan"),
        "positive_effective_score_bins": float("nan"),
        "positive_max_bin_mass": float("nan"),
        "positive_top_bin_mass": float("nan"),
        "positive_score_iqr": float("nan"),
        "positive_score_q10_q90_width": float("nan"),
        "positive_score_gini_or_concentration_index": float("nan"),
        "n_positive": int(n_positive),
        "n_components": -1,
        "giant_component_fraction": float("nan"),
        "mean_component_size": float("nan"),
        "median_component_size": float("nan"),
        "component_entropy": float("nan"),
        "isolated_positive_fraction": float("nan"),
        "meets_min_auroc": False,
        "fit_failed": True,
        "failure_reason": reason,
    }


def _failure_rows(
    *,
    dataset_id: str,
    model_name: str,
    objective: str,
    seed: int,
    requested_k_values: list[int],
    radius_quantiles: list[float],
    topology_mode: str,
    n_positive: int,
    reason: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if topology_mode in {"knn", "both"}:
        for requested_k in requested_k_values:
            effective_k = min(max(int(requested_k), 0), max(n_positive - 1, 0))
            rows.append(
                _base_failure_row(
                    dataset_id=dataset_id,
                    model_name=model_name,
                    objective=objective,
                    seed=seed,
                    topology_mode="knn",
                    k=int(requested_k),
                    effective_k=int(effective_k),
                    k_capped=bool(int(requested_k) >= n_positive),
                    radius_quantile="",
                    radius="",
                    n_positive=n_positive,
                    reason=reason,
                )
            )
    if topology_mode in {"radius", "both"}:
        for radius_quantile in radius_quantiles:
            rows.append(
                _base_failure_row(
                    dataset_id=dataset_id,
                    model_name=model_name,
                    objective=objective,
                    seed=seed,
                    topology_mode="radius",
                    k="",
                    effective_k="",
                    k_capped="",
                    radius_quantile=float(radius_quantile),
                    radius=float("nan"),
                    n_positive=n_positive,
                    reason=reason,
                )
            )
    return rows


def _add_run_metadata(
    rows: list[dict[str, object]],
    *,
    meets_min_auroc: bool,
    minority_survival_auc: float,
    minority_survival_max_drop: float,
    minority_survival_total_variation: float,
    minority_survival_drop_entropy: float,
    minority_survival_effective_drop_count: float,
    minority_survival_cliffiness: float,
    allocation_shape: dict[str, float],
) -> list[dict[str, object]]:
    for row in rows:
        row["meets_min_auroc"] = bool(meets_min_auroc)
        row["fit_failed"] = False
        row["failure_reason"] = ""
        row["minority_survival_auc"] = minority_survival_auc
        row["minority_survival_max_drop"] = minority_survival_max_drop
        row["minority_survival_total_variation"] = minority_survival_total_variation
        row["minority_survival_drop_entropy"] = minority_survival_drop_entropy
        row["minority_survival_effective_drop_count"] = minority_survival_effective_drop_count
        row["minority_survival_cliffiness"] = minority_survival_cliffiness
        row.update(allocation_shape)
    return rows


def build_radius_topology_rows(
    *,
    dataset_id: str,
    model_name: str,
    objective: str,
    seed: int,
    split_id: int,
    auroc: float,
    average_precision: float,
    persistence: float,
    positive_score_entropy: float,
    radius_by_quantile: dict[float, float],
    topology_by_radius: dict[float, dict[str, float | int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for quantile in sorted(radius_by_quantile):
        radius = float(radius_by_quantile[quantile])
        topology = topology_by_radius[radius]
        rows.append(
            {
                "dataset_id": dataset_id,
                "model_name": model_name,
                "objective": objective,
                "seed": int(seed),
                "split_id": int(split_id),
                "topology_mode": "radius",
                "k": "",
                "effective_k": "",
                "k_capped": "",
                "radius_quantile": float(quantile),
                "radius": radius,
                "auroc": float(auroc),
                "average_precision": float(average_precision),
                "persistence": float(persistence),
                "positive_score_entropy": float(positive_score_entropy),
                "n_positive": int(topology["n_positive"]),
                "n_components": int(topology["n_components"]),
                "giant_component_fraction": float(topology["giant_component_fraction"]),
                "mean_component_size": float(topology["mean_component_size"]),
                "median_component_size": float(topology["median_component_size"]),
                "component_entropy": float(topology["component_entropy"]),
                "isolated_positive_fraction": float(topology["isolated_positive_fraction"]),
            }
        )
    return rows


def build_objective_topology_rows(
    *,
    dataset_id: str,
    model_name: str,
    objective: str,
    seed: int,
    split_id: int,
    auroc: float,
    average_precision: float,
    persistence: float,
    positive_score_entropy: float,
    k_values: list[int],
    topology_by_k: dict[int, dict[str, float | int]],
    radius_by_quantile: dict[float, float],
    topology_by_radius: dict[float, dict[str, float | int]],
    topology_mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if topology_mode in {"knn", "both"}:
        knn_rows = build_topology_rows(
            dataset_id=dataset_id,
            model_name=model_name,
            objective=objective,
            seed=int(seed),
            split_id=int(split_id),
            auroc=auroc,
            average_precision=average_precision,
            persistence=persistence,
            positive_score_entropy=positive_score_entropy,
            requested_k_values=k_values,
            topology_by_k=topology_by_k,
        )
        for row in knn_rows:
            row["topology_mode"] = "knn"
            row["radius_quantile"] = ""
            row["radius"] = ""
        rows.extend(knn_rows)
    if topology_mode in {"radius", "both"}:
        rows.extend(
            build_radius_topology_rows(
                dataset_id=dataset_id,
                model_name=model_name,
                objective=objective,
                seed=int(seed),
                split_id=int(split_id),
                auroc=auroc,
                average_precision=average_precision,
                persistence=persistence,
                positive_score_entropy=positive_score_entropy,
                radius_by_quantile=radius_by_quantile,
                topology_by_radius=topology_by_radius,
            )
        )
    return rows


def run_experiment(
    *,
    seeds: list[int],
    skew_ratio: int,
    minority_count: int,
    test_size: float,
    k_values: list[int],
    radius_quantiles: list[float],
    topology_mode: str,
    min_auroc: float,
    include_failed_runs: bool,
    output_path: Path,
) -> Path:
    objective_models = resolve_objective_model_ids()
    rows: list[dict[str, object]] = []

    for seed in seeds:
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
        n_positive = int(np.sum(y_test == 1))

        for model_id, objective_name in objective_models:
            try:
                model = make_model(model_id, seed=int(seed))
                model.fit(X_train, y_train)
            except Exception as exc:
                if include_failed_runs:
                    rows.extend(
                        _failure_rows(
                            dataset_id=config.dataset_id,
                            model_name=model_id,
                            objective=objective_name,
                            seed=int(seed),
                            requested_k_values=k_values,
                            radius_quantiles=radius_quantiles,
                            topology_mode=topology_mode,
                            n_positive=n_positive,
                            reason=str(exc),
                        )
                    )
                continue

            y_score = positive_class_scores(model, X_test)
            auroc = float(roc_auc_score(y_test, y_score))
            average_precision = float(average_precision_score(y_test, y_score))
            occupancy = compute_occupancy_metrics(
                y_test,
                y_score,
                thresholds=[0.50, 0.25, 0.10, 0.05, 0.01],
            )
            persistence = float(occupancy["threshold_occupancy_persistence"])
            minority_survival_auc = float(occupancy["minority_survival_auc"])
            minority_survival_max_drop = float(occupancy["minority_survival_max_drop"])
            minority_survival_total_variation = float(
                occupancy["minority_survival_total_variation"]
            )
            minority_survival_drop_entropy = float(occupancy["minority_survival_drop_entropy"])
            minority_survival_effective_drop_count = float(
                occupancy["minority_survival_effective_drop_count"]
            )
            minority_survival_cliffiness = float(occupancy["minority_survival_cliffiness"])
            positive_scores = y_score[np.asarray(y_test) == 1]
            positive_score_entropy = float(
                allocation_concentration_metrics(positive_scores)["histogram_entropy"]
            )
            allocation_shape = positive_score_allocation_shape_metrics(positive_scores)

            embedding_model = getattr(model, "_model", model)
            embeddings = extract_penultimate_embeddings_mlp(embedding_model, X_test)
            topology_by_k = (
                compute_minority_support_topology(embeddings, y_test, k_values)
                if topology_mode in {"knn", "both"}
                else {}
            )
            radius_by_quantile = (
                compute_positive_distance_quantile_radii(
                    embeddings,
                    y_test,
                    quantiles=radius_quantiles,
                )
                if topology_mode in {"radius", "both"}
                else {}
            )
            topology_by_radius = (
                compute_minority_support_radius_topology(
                    embeddings,
                    y_test,
                    radii=radius_by_quantile.values(),
                )
                if topology_mode in {"radius", "both"}
                else {}
            )
            objective_rows = build_objective_topology_rows(
                dataset_id=config.dataset_id,
                model_name=model_id,
                objective=objective_name,
                seed=int(seed),
                split_id=0,
                auroc=auroc,
                average_precision=average_precision,
                persistence=persistence,
                positive_score_entropy=positive_score_entropy,
                k_values=k_values,
                topology_by_k=topology_by_k,
                radius_by_quantile=radius_by_quantile,
                topology_by_radius=topology_by_radius,
                topology_mode=topology_mode,
            )

            meets = auroc >= float(min_auroc)
            rows.extend(
                _add_run_metadata(
                    objective_rows,
                    meets_min_auroc=meets,
                    minority_survival_auc=minority_survival_auc,
                    minority_survival_max_drop=minority_survival_max_drop,
                    minority_survival_total_variation=minority_survival_total_variation,
                    minority_survival_drop_entropy=minority_survival_drop_entropy,
                    minority_survival_effective_drop_count=minority_survival_effective_drop_count,
                    minority_survival_cliffiness=minority_survival_cliffiness,
                    allocation_shape=allocation_shape,
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--skew-ratio", type=int, default=100)
    parser.add_argument("--minority-count", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--k-values", type=str, default="1,2,3,5,10")
    parser.add_argument("--topology-mode", choices=["knn", "radius", "both"], default="both")
    parser.add_argument("--radius-quantiles", type=str, default="0.05,0.10,0.20,0.30,0.50")
    parser.add_argument("--min-auroc", type=float, default=0.5)
    parser.add_argument("--include-failed-runs", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "topology" / "mlp_objectives_topology.csv",
    )
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    k_values = [int(item.strip()) for item in args.k_values.split(",") if item.strip()]
    radius_quantiles = [
        float(item.strip()) for item in args.radius_quantiles.split(",") if item.strip()
    ]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if not k_values:
        raise ValueError("--k-values must contain at least one integer")
    if not radius_quantiles:
        raise ValueError("--radius-quantiles must contain at least one value")

    output = run_experiment(
        seeds=seeds,
        skew_ratio=args.skew_ratio,
        minority_count=args.minority_count,
        test_size=args.test_size,
        k_values=k_values,
        radius_quantiles=radius_quantiles,
        topology_mode=args.topology_mode,
        min_auroc=args.min_auroc,
        include_failed_runs=args.include_failed_runs,
        output_path=args.output,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
