"""Run allocation morphology trajectory checkpoints for MLP objectives.

Implementation note: sklearn's MLPClassifier has no native dropout or epoch
callback. This script approximates checkpoint training with explicit
``partial_fit`` epochs. Dropout variants use Bernoulli feature-mask augmentation
during each epoch, which approximates feature-subspace sampling rather than true
hidden-unit dropout.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.allocation_shape import positive_score_allocation_shape_metrics  # noqa: E402
from hib.occupancy import compute_occupancy_metrics  # noqa: E402
from hib.synthetic import SyntheticSkewConfig, make_train_test_split  # noqa: E402


OBJECTIVES = [
    "mlp_bce",
    "mlp_weighted_bce",
    "mlp_oversampled_bce",
    "mlp_bce_dropout_0_1",
    "mlp_weighted_bce_dropout_0_1",
]

CHECKPOINTS = [1, 2, 5, 10, 20, 50, 100]
OUTPUT_COLUMNS = [
    "objective",
    "seed",
    "epoch",
    "breadth",
    "elevation",
    "minority_survival_auc",
    "minority_survival_cliffiness",
    "auroc",
    "average_precision",
]


def _make_mlp(seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        random_state=int(seed),
        max_iter=1,
        warm_start=True,
        shuffle=True,
    )


def _class_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    count_map = {int(cls): int(count) for cls, count in zip(classes, counts, strict=False)}
    n_total = int(y.size)
    weights = {int(cls): n_total / (2.0 * float(count_map[int(cls)])) for cls in classes}
    return np.asarray([weights[int(label)] for label in y], dtype=float)


def _oversampled_epoch_batch(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    if classes.size != 2:
        return X, y
    majority = int(np.max(counts))
    sampled_idx = []
    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        sampled_idx.append(rng.choice(cls_idx, size=majority, replace=cls_idx.size < majority))
    idx = np.concatenate(sampled_idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _feature_dropout_batch(
    X: np.ndarray,
    y: np.ndarray,
    dropout_rate: float,
    rng: np.random.Generator,
    n_augments: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    keep_probability = 1.0 - float(dropout_rate)
    X_parts = [X]
    y_parts = [y]
    for _ in range(int(n_augments)):
        mask = rng.binomial(1, keep_probability, size=X.shape).astype(float)
        X_parts.append((X * mask) / keep_probability)
        y_parts.append(y)
    return np.vstack(X_parts), np.concatenate(y_parts)


def _objective_epoch_batch(
    objective: str,
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if objective == "mlp_oversampled_bce":
        X_epoch, y_epoch = _oversampled_epoch_batch(X, y, rng)
        return X_epoch, y_epoch, None

    dropout_rate = 0.0
    if objective in {"mlp_bce_dropout_0_1", "mlp_weighted_bce_dropout_0_1"}:
        dropout_rate = 0.1
    X_epoch, y_epoch = X, y
    if dropout_rate > 0.0:
        X_epoch, y_epoch = _feature_dropout_batch(X, y, dropout_rate, rng)

    if objective in {"mlp_weighted_bce", "mlp_weighted_bce_dropout_0_1"}:
        return X_epoch, y_epoch, _class_weights(y_epoch)
    return X_epoch, y_epoch, None


def _partial_fit_epoch(
    model: MLPClassifier,
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    sample_weight: np.ndarray | None,
) -> None:
    if sample_weight is not None and "sample_weight" in inspect.signature(model.partial_fit).parameters:
        model.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
        return
    model.partial_fit(X, y, classes=classes)


def build_checkpoint_row(
    objective: str,
    seed: int,
    epoch: int,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | int | str]:
    positive_scores = y_score[np.asarray(y_true) == 1]
    allocation = positive_score_allocation_shape_metrics(positive_scores)
    occupancy = compute_occupancy_metrics(
        y_true,
        y_score,
        thresholds=[0.50, 0.25, 0.10, 0.05, 0.01],
    )
    return {
        "objective": objective,
        "seed": int(seed),
        "epoch": int(epoch),
        "breadth": float(allocation["positive_histogram_entropy"]),
        "elevation": float(allocation["positive_top_bin_mass"]),
        "minority_survival_auc": float(occupancy["minority_survival_auc"]),
        "minority_survival_cliffiness": float(occupancy["minority_survival_cliffiness"]),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def run_objective_seed(
    objective: str,
    seed: int,
    checkpoints: list[int],
    skew_ratio: int,
    minority_count: int,
) -> list[dict[str, float | int | str]]:
    config = SyntheticSkewConfig(
        skew_ratio=int(skew_ratio),
        seed=int(seed),
        separation=2.0,
        minority_count=int(minority_count),
        n_features=6,
        noise=1.0,
        test_size=0.5,
    )
    X_train, X_test, y_train, y_test = make_train_test_split(config)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = _make_mlp(seed)
    classes = np.array([0, 1], dtype=int)
    rng = np.random.default_rng(int(seed) + 7919)
    checkpoint_set = set(int(epoch) for epoch in checkpoints)
    rows = []
    for epoch in range(1, max(checkpoint_set) + 1):
        X_epoch, y_epoch, sample_weight = _objective_epoch_batch(objective, X_train, y_train, rng)
        _partial_fit_epoch(model, X_epoch, y_epoch, classes, sample_weight)
        if epoch in checkpoint_set:
            y_score = model.predict_proba(X_test)[:, 1]
            rows.append(build_checkpoint_row(objective, seed, epoch, y_test, y_score))
    return rows


def run_experiment(
    seeds: list[int],
    objectives: list[str],
    checkpoints: list[int],
    skew_ratio: int,
    minority_count: int,
    output_path: Path,
) -> Path:
    rows = []
    for seed in seeds:
        for objective in objectives:
            rows.extend(
                run_objective_seed(
                    objective=objective,
                    seed=int(seed),
                    checkpoints=checkpoints,
                    skew_ratio=int(skew_ratio),
                    minority_count=int(minority_count),
                )
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--objectives", type=str, default=",".join(OBJECTIVES))
    parser.add_argument("--checkpoints", type=str, default=",".join(str(epoch) for epoch in CHECKPOINTS))
    parser.add_argument("--skew-ratio", type=int, default=100)
    parser.add_argument("--minority-count", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "topology" / "allocation_trajectory.csv",
    )
    args = parser.parse_args()
    seeds = _parse_int_list(args.seeds)
    checkpoints = _parse_int_list(args.checkpoints)
    objectives = [item.strip() for item in args.objectives.split(",") if item.strip()]
    output = run_experiment(
        seeds=seeds,
        objectives=objectives,
        checkpoints=checkpoints,
        skew_ratio=args.skew_ratio,
        minority_count=args.minority_count,
        output_path=args.output,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
