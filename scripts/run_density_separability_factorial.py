"""Run density x separability factorial allocation morphology experiment."""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.allocation_shape import positive_score_allocation_shape_metrics  # noqa: E402
from hib.models import make_model  # noqa: E402
from hib.occupancy import compute_occupancy_metrics  # noqa: E402


DENSITY_LEVELS = {
    "high_density": 0.10,
    "medium_density": 0.50,
    "low_density": 1.00,
}
SEPARABILITY_LEVELS = {
    "high_separability": 4.00,
    "medium_separability": 2.00,
    "low_separability": 1.00,
}
DEFAULT_MODELS = ["mlp_weighted_bce", "mlp_weighted_bce_dropout_0_1"]
MODEL_ALIASES = {
    "mlp_weighted_bce": "mlp_weighted",
    "mlp_oversampled_bce": "mlp_oversampled",
}
OUTPUT_COLUMNS = [
    "density_level",
    "separability_level",
    "minority_cov",
    "centroid_distance",
    "model_id",
    "registry_model_id",
    "seed",
    "skew_ratio",
    "minority_count",
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
    "mean_positive_knn_distance",
    "positive_density_proxy",
    "local_positive_ratio_mean",
    "local_label_entropy_mean",
]


class FactorialDataset(NamedTuple):
    X: np.ndarray
    y: np.ndarray


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def construct_grid(
    density_levels: list[str],
    separability_levels: list[str],
    model_ids: list[str],
    seeds: list[int],
) -> list[dict[str, str | int | float]]:
    rows = []
    for density_level in density_levels:
        if density_level not in DENSITY_LEVELS:
            raise ValueError(f"unknown density level {density_level!r}")
        for separability_level in separability_levels:
            if separability_level not in SEPARABILITY_LEVELS:
                raise ValueError(f"unknown separability level {separability_level!r}")
            for model_id in model_ids:
                for seed in seeds:
                    rows.append(
                        {
                            "density_level": density_level,
                            "separability_level": separability_level,
                            "minority_cov": DENSITY_LEVELS[density_level],
                            "centroid_distance": SEPARABILITY_LEVELS[separability_level],
                            "model_id": model_id,
                            "seed": int(seed),
                        }
                    )
    return rows


def make_factorial_dataset(
    *,
    minority_cov: float,
    centroid_distance: float,
    seed: int,
    skew_ratio: int = 100,
    minority_count: int = 100,
    n_features: int = 6,
) -> FactorialDataset:
    rng = np.random.default_rng(seed)
    majority_count = int(skew_ratio) * int(minority_count)
    majority_center = np.zeros(n_features)
    minority_center = np.zeros(n_features)
    minority_center[0] = float(centroid_distance)
    majority = rng.normal(majority_center, 1.0, size=(majority_count, n_features))
    minority = rng.normal(minority_center, float(minority_cov), size=(minority_count, n_features))
    X = np.vstack([majority, minority])
    y = np.concatenate([np.zeros(majority_count, dtype=int), np.ones(minority_count, dtype=int)])
    permutation = rng.permutation(y.size)
    return FactorialDataset(X=X[permutation], y=y[permutation])


def train_test_scaled_split(
    dataset: FactorialDataset,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.5,
        random_state=int(seed),
        stratify=dataset.y,
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def _label_entropy(labels: np.ndarray) -> float:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2).astype(float)
    probs = counts / float(max(1.0, counts.sum()))
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log(probs)))


def information_density_metrics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    positive_k: int = 5,
    local_k: int = 20,
) -> dict[str, float]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    pos = X_arr[y_arr == 1]
    if pos.shape[0] < 2:
        mean_distance = 0.0
    else:
        diff = pos[:, None, :] - pos[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(distances, np.inf)
        k_pos = min(int(positive_k), pos.shape[0] - 1)
        knn_distances = np.partition(distances, kth=k_pos - 1, axis=1)[:, :k_pos]
        mean_distance = float(np.mean(knn_distances))

    local_ratios = []
    local_entropies = []
    k_local = min(int(local_k), max(1, X_arr.shape[0] - 1))
    for point in pos:
        distances = np.sqrt(np.sum((X_arr - point) ** 2, axis=1))
        nearest = np.argpartition(distances, kth=k_local)[: k_local + 1]
        nearest = nearest[np.argsort(distances[nearest])]
        if distances[nearest[0]] <= 1e-12:
            nearest = nearest[1:]
        labels = y_arr[nearest[:k_local]]
        local_ratios.append(float(np.mean(labels == 1)))
        local_entropies.append(_label_entropy(labels))

    return {
        "mean_positive_knn_distance": mean_distance,
        "positive_density_proxy": float(1.0 / max(1e-9, mean_distance)),
        "local_positive_ratio_mean": float(np.mean(local_ratios)) if local_ratios else 0.0,
        "local_label_entropy_mean": float(np.mean(local_entropies)) if local_entropies else 0.0,
    }


def build_output_row(
    *,
    density_level: str,
    separability_level: str,
    minority_cov: float,
    centroid_distance: float,
    model_id: str,
    registry_model_id: str,
    seed: int,
    skew_ratio: int,
    minority_count: int,
    y_true: np.ndarray,
    y_score: np.ndarray,
    density_metrics: dict[str, float],
) -> dict[str, float | int | str]:
    positive_scores = y_score[np.asarray(y_true) == 1]
    allocation = positive_score_allocation_shape_metrics(positive_scores)
    occupancy = compute_occupancy_metrics(y_true, y_score, thresholds=[0.50, 0.25, 0.10, 0.05, 0.01])
    row = {
        "density_level": density_level,
        "separability_level": separability_level,
        "minority_cov": float(minority_cov),
        "centroid_distance": float(centroid_distance),
        "model_id": model_id,
        "registry_model_id": registry_model_id,
        "seed": int(seed),
        "skew_ratio": int(skew_ratio),
        "minority_count": int(minority_count),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "minority_survival_auc": float(occupancy["minority_survival_auc"]),
        "minority_survival_cliffiness": float(occupancy["minority_survival_cliffiness"]),
        "minority_survival_max_drop": float(occupancy["minority_survival_max_drop"]),
        "minority_survival_effective_drop_count": float(occupancy["minority_survival_effective_drop_count"]),
        "breadth": float(allocation["positive_histogram_entropy"]),
        "effective_breadth": float(allocation["positive_effective_score_bins"]),
        "elevation": float(allocation["positive_top_bin_mass"]),
        "peak_concentration": float(allocation["positive_max_bin_mass"]),
    }
    row.update(density_metrics)
    return row


def run_cell(
    *,
    density_level: str,
    separability_level: str,
    model_id: str,
    seed: int,
    skew_ratio: int,
    minority_count: int,
) -> dict[str, float | int | str]:
    minority_cov = DENSITY_LEVELS[density_level]
    centroid_distance = SEPARABILITY_LEVELS[separability_level]
    dataset = make_factorial_dataset(
        minority_cov=minority_cov,
        centroid_distance=centroid_distance,
        seed=seed,
        skew_ratio=skew_ratio,
        minority_count=minority_count,
    )
    X_train, X_test, y_train, y_test = train_test_scaled_split(dataset, seed)
    density_metrics = information_density_metrics(X_train, y_train)
    registry_model_id = MODEL_ALIASES.get(model_id, model_id)
    model = make_model(registry_model_id, seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    return build_output_row(
        density_level=density_level,
        separability_level=separability_level,
        minority_cov=minority_cov,
        centroid_distance=centroid_distance,
        model_id=model_id,
        registry_model_id=registry_model_id,
        seed=seed,
        skew_ratio=skew_ratio,
        minority_count=minority_count,
        y_true=y_test,
        y_score=y_score,
        density_metrics=density_metrics,
    )


def run_experiment(
    *,
    density_levels: list[str],
    separability_levels: list[str],
    model_ids: list[str],
    seeds: list[int],
    skew_ratio: int,
    minority_count: int,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = construct_grid(density_levels, separability_levels, model_ids, seeds)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for cell in grid:
            writer.writerow(
                run_cell(
                    density_level=str(cell["density_level"]),
                    separability_level=str(cell["separability_level"]),
                    model_id=str(cell["model_id"]),
                    seed=int(cell["seed"]),
                    skew_ratio=skew_ratio,
                    minority_count=minority_count,
                )
            )
            handle.flush()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density-levels", default=",".join(DENSITY_LEVELS))
    parser.add_argument("--separability-levels", default=",".join(SEPARABILITY_LEVELS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(20)))
    parser.add_argument("--skew-ratio", type=int, default=100)
    parser.add_argument("--minority-count", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "topology" / "density_separability_factorial.csv",
    )
    args = parser.parse_args()
    output = run_experiment(
        density_levels=parse_list(args.density_levels),
        separability_levels=parse_list(args.separability_levels),
        model_ids=parse_list(args.models),
        seeds=parse_int_list(args.seeds),
        skew_ratio=args.skew_ratio,
        minority_count=args.minority_count,
        output_path=args.output,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
