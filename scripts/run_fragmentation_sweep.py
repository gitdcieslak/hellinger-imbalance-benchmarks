"""Run minority fragmentation sweep under fixed island density settings."""

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


DEFAULT_ISLANDS = [1, 2, 4, 5, 10, 20, 50, 100]
DEFAULT_MODELS = ["mlp_weighted_bce", "mlp_weighted_bce_dropout_0_1"]
MODEL_ALIASES = {"mlp_weighted_bce": "mlp_weighted", "mlp_oversampled_bce": "mlp_oversampled"}
OUTPUT_COLUMNS = [
    "n_islands",
    "model_id",
    "registry_model_id",
    "seed",
    "skew_ratio",
    "minority_count",
    "n_features",
    "island_cov",
    "centroid_radius",
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
    "minority_cluster_count",
    "positives_per_cluster",
]


class FragmentationDataset(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    cluster_ids: np.ndarray
    positives_per_cluster: float


def parse_int_list(value: str) -> list[int]:
    values: list[int] = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        if "-" in item:
            start, end = item.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(item))
    return values


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def positives_per_island_counts(minority_count: int, n_islands: int) -> np.ndarray:
    if n_islands < 1:
        raise ValueError("n_islands must be positive")
    base = int(minority_count) // int(n_islands)
    remainder = int(minority_count) % int(n_islands)
    counts = np.full(int(n_islands), base, dtype=int)
    counts[:remainder] += 1
    return counts


def island_centers(n_islands: int, n_features: int, radius: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.zeros((int(n_islands), int(n_features)), dtype=float)
    if n_islands == 1:
        centers[0, 0] = float(radius)
        return centers
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_islands), endpoint=False)
    centers[:, 0] = float(radius) * np.cos(angles)
    centers[:, 1] = float(radius) * np.sin(angles)
    if n_features > 2:
        centers[:, 2:] = rng.normal(0.0, 0.15, size=(int(n_islands), int(n_features) - 2))
    return centers


def make_fragmentation_dataset(
    *,
    n_islands: int,
    seed: int,
    skew_ratio: int = 100,
    minority_count: int = 100,
    n_features: int = 10,
    island_cov: float = 0.16,
    centroid_radius: float = 3.0,
) -> FragmentationDataset:
    rng = np.random.default_rng(seed)
    majority_count = int(skew_ratio) * int(minority_count)
    majority = rng.normal(0.0, 1.0, size=(majority_count, n_features))
    counts = positives_per_island_counts(minority_count, n_islands)
    centers = island_centers(n_islands, n_features, centroid_radius, seed)
    minority_parts = []
    minority_cluster_ids = []
    for cluster_id, count in enumerate(counts):
        if count <= 0:
            continue
        minority_parts.append(rng.normal(centers[cluster_id], float(island_cov), size=(int(count), n_features)))
        minority_cluster_ids.extend([cluster_id] * int(count))
    minority = np.vstack(minority_parts)
    cluster_ids = np.asarray(minority_cluster_ids, dtype=int)
    X = np.vstack([majority, minority])
    y = np.concatenate([np.zeros(majority_count, dtype=int), np.ones(minority.shape[0], dtype=int)])
    all_cluster_ids = np.concatenate([np.full(majority_count, -1, dtype=int), cluster_ids])
    permutation = rng.permutation(y.size)
    return FragmentationDataset(
        X=X[permutation],
        y=y[permutation],
        cluster_ids=all_cluster_ids[permutation],
        positives_per_cluster=float(minority_count) / float(n_islands),
    )


def train_test_scaled_split(dataset: FragmentationDataset, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def information_density_metrics(X: np.ndarray, y: np.ndarray, *, positive_k: int = 5, local_k: int = 20) -> dict[str, float]:
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
    n_islands: int,
    model_id: str,
    registry_model_id: str,
    seed: int,
    skew_ratio: int,
    minority_count: int,
    n_features: int,
    island_cov: float,
    centroid_radius: float,
    y_true: np.ndarray,
    y_score: np.ndarray,
    density_metrics: dict[str, float],
) -> dict[str, float | int | str]:
    positive_scores = y_score[np.asarray(y_true) == 1]
    allocation = positive_score_allocation_shape_metrics(positive_scores)
    occupancy = compute_occupancy_metrics(y_true, y_score, thresholds=[0.50, 0.25, 0.10, 0.05, 0.01])
    row = {
        "n_islands": int(n_islands),
        "model_id": model_id,
        "registry_model_id": registry_model_id,
        "seed": int(seed),
        "skew_ratio": int(skew_ratio),
        "minority_count": int(minority_count),
        "n_features": int(n_features),
        "island_cov": float(island_cov),
        "centroid_radius": float(centroid_radius),
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
        "minority_cluster_count": int(n_islands),
        "positives_per_cluster": float(minority_count) / float(n_islands),
    }
    row.update(density_metrics)
    return row


def run_cell(*, n_islands: int, model_id: str, seed: int, skew_ratio: int, minority_count: int, n_features: int, island_cov: float, centroid_radius: float) -> dict[str, float | int | str]:
    dataset = make_fragmentation_dataset(
        n_islands=n_islands,
        seed=seed,
        skew_ratio=skew_ratio,
        minority_count=minority_count,
        n_features=n_features,
        island_cov=island_cov,
        centroid_radius=centroid_radius,
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
        n_islands=n_islands,
        model_id=model_id,
        registry_model_id=registry_model_id,
        seed=seed,
        skew_ratio=skew_ratio,
        minority_count=minority_count,
        n_features=n_features,
        island_cov=island_cov,
        centroid_radius=centroid_radius,
        y_true=y_test,
        y_score=y_score,
        density_metrics=density_metrics,
    )


def run_experiment(*, n_islands_values: list[int], model_ids: list[str], seeds: list[int], skew_ratio: int, minority_count: int, n_features: int, island_cov: float, centroid_radius: float, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for n_islands in n_islands_values:
            for model_id in model_ids:
                for seed in seeds:
                    writer.writerow(
                        run_cell(
                            n_islands=n_islands,
                            model_id=model_id,
                            seed=seed,
                            skew_ratio=skew_ratio,
                            minority_count=minority_count,
                            n_features=n_features,
                            island_cov=island_cov,
                            centroid_radius=centroid_radius,
                        )
                    )
                    handle.flush()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-islands", default=",".join(str(v) for v in DEFAULT_ISLANDS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(20)))
    parser.add_argument("--skew-ratio", type=int, default=100)
    parser.add_argument("--minority-count", type=int, default=100)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--island-cov", type=float, default=0.16)
    parser.add_argument("--centroid-radius", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "topology" / "fragmentation_sweep.csv")
    args = parser.parse_args()
    output = run_experiment(
        n_islands_values=parse_int_list(args.n_islands),
        model_ids=parse_list(args.models),
        seeds=parse_int_list(args.seeds),
        skew_ratio=args.skew_ratio,
        minority_count=args.minority_count,
        n_features=args.n_features,
        island_cov=args.island_cov,
        centroid_radius=args.centroid_radius,
        output_path=args.output,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
