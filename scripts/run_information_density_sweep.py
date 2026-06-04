"""Run minority information-density allocation morphology sweep."""

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
from hib.models import OptionalDependencyUnavailable, make_model  # noqa: E402
from hib.occupancy import compute_occupancy_metrics  # noqa: E402


MINIMAL_REGIMES = ["compact_dense", "fragmented_islands_20", "boundary_mixed"]
FULL_REGIMES = [
    "compact_dense",
    "diffuse_dense",
    "fragmented_islands_5",
    "fragmented_islands_20",
    "singleton_islands",
    "boundary_mixed",
]
MINIMAL_MODELS = ["mlp_weighted_bce", "mlp_weighted_bce_dropout_0_1"]
FULL_MODELS = [
    "mlp_bce",
    "mlp_weighted_bce",
    "mlp_oversampled_bce",
    "mlp_weighted_bce_dropout_0_1",
    "random_forest",
    "lightgbm",
]
MODEL_ALIASES = {
    "mlp_weighted_bce": "mlp_weighted",
    "mlp_oversampled_bce": "mlp_oversampled",
}
OUTPUT_COLUMNS = [
    "support_regime",
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
    "median_positive_knn_distance",
    "positive_density_proxy",
    "local_positive_ratio_mean",
    "local_label_entropy_mean",
    "minority_cluster_count",
    "positives_per_cluster",
]


class SupportDataset(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    minority_cluster: np.ndarray
    minority_cluster_count: int
    positives_per_cluster: float


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _cluster_centers(n_clusters: int, n_features: int, radius: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, n_clusters, endpoint=False)
    centers = np.zeros((n_clusters, n_features), dtype=float)
    centers[:, 0] = radius * np.cos(angles) + 2.6
    if n_features > 1:
        centers[:, 1] = radius * np.sin(angles)
    if n_features > 2:
        centers[:, 2:] = rng.normal(0.0, 0.25, size=(n_clusters, n_features - 2))
    return centers


def make_support_dataset(
    regime: str,
    *,
    seed: int,
    skew_ratio: int = 100,
    minority_count: int = 100,
    n_features: int = 6,
) -> SupportDataset:
    rng = np.random.default_rng(seed)
    majority_count = int(skew_ratio) * int(minority_count)
    majority = rng.normal(0.0, 1.0, size=(majority_count, n_features))

    if regime == "compact_dense":
        cluster_count = 1
        positives_per_cluster = float(minority_count)
        centers = np.zeros((1, n_features))
        centers[0, 0] = 2.8
        minority = rng.normal(centers[0], 0.25, size=(minority_count, n_features))
        cluster_ids = np.zeros(minority_count, dtype=int)
    elif regime == "diffuse_dense":
        cluster_count = 1
        positives_per_cluster = float(minority_count)
        centers = np.zeros((1, n_features))
        centers[0, 0] = 2.4
        minority = rng.normal(centers[0], 1.25, size=(minority_count, n_features))
        cluster_ids = np.zeros(minority_count, dtype=int)
    elif regime == "fragmented_islands_5":
        cluster_count = 5
        positives_per_cluster = float(minority_count / cluster_count)
        centers = _cluster_centers(cluster_count, n_features, radius=1.8, seed=seed)
        cluster_ids = np.repeat(np.arange(cluster_count), int(minority_count / cluster_count))
        minority = centers[cluster_ids] + rng.normal(0.0, 0.20, size=(minority_count, n_features))
    elif regime == "fragmented_islands_20":
        cluster_count = 20
        positives_per_cluster = float(minority_count / cluster_count)
        centers = _cluster_centers(cluster_count, n_features, radius=2.2, seed=seed)
        cluster_ids = np.repeat(np.arange(cluster_count), int(minority_count / cluster_count))
        minority = centers[cluster_ids] + rng.normal(0.0, 0.16, size=(minority_count, n_features))
    elif regime == "singleton_islands":
        cluster_count = minority_count
        positives_per_cluster = 1.0
        centers = _cluster_centers(cluster_count, n_features, radius=2.5, seed=seed)
        cluster_ids = np.arange(minority_count)
        minority = centers + rng.normal(0.0, 0.04, size=(minority_count, n_features))
    elif regime == "boundary_mixed":
        cluster_count = 1
        positives_per_cluster = float(minority_count)
        centers = np.zeros((1, n_features))
        centers[0, 0] = 0.65
        minority = rng.normal(centers[0], 0.85, size=(minority_count, n_features))
        cluster_ids = np.zeros(minority_count, dtype=int)
    else:
        raise ValueError(f"unknown support regime {regime!r}")

    X = np.vstack([majority, minority])
    y = np.concatenate([np.zeros(majority_count, dtype=int), np.ones(minority_count, dtype=int)])
    minority_cluster = np.concatenate([np.full(majority_count, -1, dtype=int), cluster_ids])
    permutation = rng.permutation(y.size)
    return SupportDataset(
        X=X[permutation],
        y=y[permutation],
        minority_cluster=minority_cluster[permutation],
        minority_cluster_count=int(cluster_count),
        positives_per_cluster=float(positives_per_cluster),
    )


def train_test_support_split(
    dataset: SupportDataset,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        dataset.X,
        dataset.y,
        dataset.minority_cluster,
        test_size=0.5,
        random_state=int(seed),
        stratify=dataset.y,
    )


def _label_entropy(labels: np.ndarray) -> float:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2).astype(float)
    probs = counts / float(max(1.0, counts.sum()))
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log(probs)))


def information_density_metrics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    minority_cluster_count: int,
    positives_per_cluster: float,
    positive_k: int = 5,
    local_k: int = 20,
) -> dict[str, float | int]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    pos = X_arr[y_arr == 1]
    if pos.shape[0] < 2:
        mean_distance = 0.0
        median_distance = 0.0
    else:
        diff = pos[:, None, :] - pos[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(distances, np.inf)
        k_pos = min(int(positive_k), pos.shape[0] - 1)
        knn_distances = np.partition(distances, kth=k_pos - 1, axis=1)[:, :k_pos]
        per_point_distance = np.mean(knn_distances, axis=1)
        mean_distance = float(np.mean(per_point_distance))
        median_distance = float(np.median(per_point_distance))

    local_ratios = []
    local_entropies = []
    k_local = min(int(local_k), max(1, X_arr.shape[0] - 1))
    for point in pos:
        distances = np.sqrt(np.sum((X_arr - point) ** 2, axis=1))
        nearest = np.argpartition(distances, kth=k_local)[: k_local + 1]
        nearest = nearest[np.argsort(distances[nearest])]
        if distances[nearest[0]] <= 1e-12:
            nearest = nearest[1:]
        nearest = nearest[:k_local]
        labels = y_arr[nearest]
        local_ratios.append(float(np.mean(labels == 1)))
        local_entropies.append(_label_entropy(labels))

    density_proxy = 1.0 / max(1e-9, mean_distance)
    return {
        "mean_positive_knn_distance": mean_distance,
        "median_positive_knn_distance": median_distance,
        "positive_density_proxy": float(density_proxy),
        "local_positive_ratio_mean": float(np.mean(local_ratios)) if local_ratios else 0.0,
        "local_label_entropy_mean": float(np.mean(local_entropies)) if local_entropies else 0.0,
        "minority_cluster_count": int(minority_cluster_count),
        "positives_per_cluster": float(positives_per_cluster),
    }


def build_output_row(
    *,
    support_regime: str,
    model_id: str,
    registry_model_id: str,
    seed: int,
    skew_ratio: int,
    minority_count: int,
    y_true: np.ndarray,
    y_score: np.ndarray,
    density: dict[str, float | int],
) -> dict[str, float | int | str]:
    positive_scores = y_score[np.asarray(y_true) == 1]
    allocation = positive_score_allocation_shape_metrics(positive_scores)
    occupancy = compute_occupancy_metrics(y_true, y_score, thresholds=[0.50, 0.25, 0.10, 0.05, 0.01])
    row = {
        "support_regime": support_regime,
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
    row.update(density)
    return row


def run_cell(
    *,
    support_regime: str,
    model_id: str,
    seed: int,
    skew_ratio: int,
    minority_count: int,
) -> dict[str, float | int | str] | None:
    dataset = make_support_dataset(
        support_regime,
        seed=seed,
        skew_ratio=skew_ratio,
        minority_count=minority_count,
    )
    X_train, X_test, y_train, y_test, _, _ = train_test_support_split(dataset, seed)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    density = information_density_metrics(
        X_train_scaled,
        y_train,
        minority_cluster_count=dataset.minority_cluster_count,
        positives_per_cluster=dataset.positives_per_cluster,
    )
    registry_model_id = MODEL_ALIASES.get(model_id, model_id)
    try:
        model = make_model(registry_model_id, seed)
    except OptionalDependencyUnavailable as exc:
        print(f"skipping {model_id}: {exc}")
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train_scaled, y_train)
    y_score = model.predict_proba(X_test_scaled)[:, 1]
    return build_output_row(
        support_regime=support_regime,
        model_id=model_id,
        registry_model_id=registry_model_id,
        seed=seed,
        skew_ratio=skew_ratio,
        minority_count=minority_count,
        y_true=y_test,
        y_score=y_score,
        density=density,
    )


def run_experiment(
    *,
    support_regimes: list[str],
    model_ids: list[str],
    seeds: list[int],
    skew_ratio: int,
    minority_count: int,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for support_regime in support_regimes:
            for model_id in model_ids:
                for seed in seeds:
                    row = run_cell(
                        support_regime=support_regime,
                        model_id=model_id,
                        seed=seed,
                        skew_ratio=skew_ratio,
                        minority_count=minority_count,
                    )
                    if row is not None:
                        writer.writerow(row)
                        handle.flush()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-regimes", default=",".join(MINIMAL_REGIMES))
    parser.add_argument("--models", default=",".join(MINIMAL_MODELS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(10)))
    parser.add_argument("--skew-ratio", type=int, default=100)
    parser.add_argument("--minority-count", type=int, default=100)
    parser.add_argument("--full", action="store_true", help="Use full regime/model defaults.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "topology" / "information_density_sweep.csv",
    )
    args = parser.parse_args()
    support_regimes = FULL_REGIMES if args.full else parse_list(args.support_regimes)
    model_ids = FULL_MODELS if args.full else parse_list(args.models)
    output = run_experiment(
        support_regimes=support_regimes,
        model_ids=model_ids,
        seeds=parse_int_list(args.seeds),
        skew_ratio=args.skew_ratio,
        minority_count=args.minority_count,
        output_path=args.output,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
