"""Run a smoke topology experiment on one severe-skew synthetic dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.allocation import allocation_concentration_metrics  # noqa: E402
from hib.metrics import positive_class_scores  # noqa: E402
from hib.models import make_model  # noqa: E402
from hib.occupancy import compute_occupancy_metrics  # noqa: E402
from hib.synthetic import SyntheticSkewConfig, make_train_test_split  # noqa: E402
from hib.topology import compute_minority_support_topology  # noqa: E402


OUTPUT_COLUMNS = [
    "dataset_id",
    "model_name",
    "objective",
    "seed",
    "split_id",
    "k",
    "effective_k",
    "k_capped",
    "auroc",
    "average_precision",
    "persistence",
    "positive_score_entropy",
    "n_positive",
    "n_components",
    "giant_component_fraction",
    "mean_component_size",
    "median_component_size",
    "component_entropy",
    "isolated_positive_fraction",
]


def _hidden_activation(values: np.ndarray, activation_name: str) -> np.ndarray:
    if activation_name == "relu":
        return np.maximum(values, 0.0)
    if activation_name == "tanh":
        return np.tanh(values)
    if activation_name == "logistic":
        return 1.0 / (1.0 + np.exp(-values))
    if activation_name == "identity":
        return values
    raise ValueError(f"unsupported MLP activation {activation_name!r}")


def extract_penultimate_embeddings_mlp(model: object, X: np.ndarray) -> np.ndarray:
    """Extract penultimate embeddings from a fitted sklearn MLPClassifier."""

    if not hasattr(model, "coefs_") or not hasattr(model, "intercepts_"):
        raise TypeError("model does not expose sklearn MLP parameters")

    coefs = getattr(model, "coefs_")
    intercepts = getattr(model, "intercepts_")
    if len(coefs) < 2:
        return np.asarray(X, dtype=float)

    activations = np.asarray(X, dtype=float)
    activation_name = str(getattr(model, "activation", "relu"))
    for layer_idx in range(len(coefs) - 1):
        activations = activations @ coefs[layer_idx] + intercepts[layer_idx]
        activations = _hidden_activation(activations, activation_name)
    return np.asarray(activations, dtype=float)


def build_topology_rows(
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
    requested_k_values: Iterable[int],
    topology_by_k: dict[int, dict[str, float | int]],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for requested_k in requested_k_values:
        k = int(requested_k)
        topology = topology_by_k[k]
        n_positive = int(topology["n_positive"])
        effective_k = min(max(k, 0), max(n_positive - 1, 0))
        rows.append(
            {
                "dataset_id": dataset_id,
                "model_name": model_name,
                "objective": objective,
                "seed": int(seed),
                "split_id": int(split_id),
                "k": k,
                "effective_k": int(effective_k),
                "k_capped": bool(k >= n_positive),
                "auroc": float(auroc),
                "average_precision": float(average_precision),
                "persistence": float(persistence),
                "positive_score_entropy": float(positive_score_entropy),
                "n_positive": n_positive,
                "n_components": int(topology["n_components"]),
                "giant_component_fraction": float(topology["giant_component_fraction"]),
                "mean_component_size": float(topology["mean_component_size"]),
                "median_component_size": float(topology["median_component_size"]),
                "component_entropy": float(topology["component_entropy"]),
                "isolated_positive_fraction": float(topology["isolated_positive_fraction"]),
            }
        )
    return rows


def run_smoke_experiment(
    seed: int,
    skew_ratio: int,
    minority_count: int,
    test_size: float,
    k_values: list[int],
    output_path: Path,
) -> Path:
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

    model = make_model("mlp_bce", seed=int(seed))
    model.fit(X_train, y_train)
    y_score = positive_class_scores(model, X_test)

    auroc = float(roc_auc_score(y_test, y_score))
    average_precision = float(average_precision_score(y_test, y_score))

    occupancy = compute_occupancy_metrics(y_test, y_score, thresholds=[0.50, 0.25, 0.10, 0.05, 0.01])
    persistence = float(occupancy["threshold_occupancy_persistence"])

    positive_scores = y_score[np.asarray(y_test) == 1]
    positive_score_entropy = float(
        allocation_concentration_metrics(positive_scores)["histogram_entropy"]
    )

    embeddings = extract_penultimate_embeddings_mlp(model, X_test)
    topology_by_k = compute_minority_support_topology(embeddings, y_test, k_values=k_values)

    rows = build_topology_rows(
        dataset_id=config.dataset_id,
        model_name="mlp_bce",
        objective="bce",
        seed=int(seed),
        split_id=0,
        auroc=auroc,
        average_precision=average_precision,
        persistence=persistence,
        positive_score_entropy=positive_score_entropy,
        requested_k_values=k_values,
        topology_by_k=topology_by_k,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skew-ratio", type=int, default=100)
    parser.add_argument("--minority-count", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--k-values", type=str, default="1,2,3,5,10")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "topology" / "topology_smoke.csv",
    )
    args = parser.parse_args()
    k_values = [int(item.strip()) for item in args.k_values.split(",") if item.strip()]
    if not k_values:
        raise ValueError("--k-values must contain at least one integer")
    output = run_smoke_experiment(
        seed=args.seed,
        skew_ratio=args.skew_ratio,
        minority_count=args.minority_count,
        test_size=args.test_size,
        k_values=k_values,
        output_path=args.output,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
