"""Run weighted BCE MLP feature-dropout sweep under severe imbalance.

For dense skew regimes, the weighted update uses class-balanced epoch sampling
so each dropout rate follows an identical protocol without runtime scaling with
the majority count. Dropout 0.00 is the no-dropout baseline under this same
training loop.
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.allocation_shape import positive_score_allocation_shape_metrics  # noqa: E402
from hib.occupancy import compute_occupancy_metrics  # noqa: E402
from hib.synthetic import SyntheticSkewConfig, make_train_test_split  # noqa: E402


DROPOUT_RATES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
SKEW_RATIOS = [25, 100, 500, 1000]
OUTPUT_COLUMNS = [
    "skew_ratio",
    "dropout_rate",
    "seed",
    "epoch",
    "breadth",
    "effective_breadth",
    "elevation",
    "peak_concentration",
    "minority_survival_auc",
    "minority_survival_cliffiness",
    "minority_survival_max_drop",
    "minority_survival_effective_drop_count",
    "auroc",
    "average_precision",
]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _make_mlp(seed: int, max_iter: int = 100) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        random_state=int(seed),
        max_iter=int(max_iter),
        shuffle=True,
    )


def _class_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    n_total = int(y.size)
    weights = {int(cls): n_total / (2.0 * float(count)) for cls, count in zip(classes, counts, strict=False)}
    return np.asarray([weights[int(label)] for label in y], dtype=float)


def _balanced_resample(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    if classes.size != 2:
        return X, y
    majority = int(np.max(counts))
    sampled = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        sampled.append(rng.choice(idx, size=majority, replace=idx.size < majority))
    train_idx = np.concatenate(sampled)
    rng.shuffle(train_idx)
    return X[train_idx], y[train_idx]


def _feature_dropout_batch(
    X: np.ndarray,
    y: np.ndarray,
    dropout_rate: float,
    rng: np.random.Generator,
    n_augments: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    if dropout_rate <= 0.0:
        return X, y
    keep_probability = 1.0 - float(dropout_rate)
    X_parts = [X]
    y_parts = [y]
    for _ in range(int(n_augments)):
        mask = rng.binomial(1, keep_probability, size=X.shape).astype(float)
        X_parts.append((X * mask) / keep_probability)
        y_parts.append(y)
    return np.vstack(X_parts), np.concatenate(y_parts)


def _partial_fit_weighted(
    model: MLPClassifier,
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    rng: np.random.Generator,
) -> None:
    model.partial_fit(X, y, classes=classes)


def build_output_row(
    skew_ratio: int,
    dropout_rate: float,
    seed: int,
    epoch: int,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | int]:
    positive_scores = y_score[np.asarray(y_true) == 1]
    allocation = positive_score_allocation_shape_metrics(positive_scores)
    occupancy = compute_occupancy_metrics(
        y_true,
        y_score,
        thresholds=[0.50, 0.25, 0.10, 0.05, 0.01],
    )
    return {
        "skew_ratio": int(skew_ratio),
        "dropout_rate": float(dropout_rate),
        "seed": int(seed),
        "epoch": int(epoch),
        "breadth": float(allocation["positive_histogram_entropy"]),
        "effective_breadth": float(allocation["positive_effective_score_bins"]),
        "elevation": float(allocation["positive_top_bin_mass"]),
        "peak_concentration": float(allocation["positive_max_bin_mass"]),
        "minority_survival_auc": float(occupancy["minority_survival_auc"]),
        "minority_survival_cliffiness": float(occupancy["minority_survival_cliffiness"]),
        "minority_survival_max_drop": float(occupancy["minority_survival_max_drop"]),
        "minority_survival_effective_drop_count": float(occupancy["minority_survival_effective_drop_count"]),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def run_seed_dropout(
    seed: int,
    dropout_rate: float,
    epochs: int,
    skew_ratio: int,
    minority_count: int,
) -> dict[str, float | int]:
    X_train, X_test, y_train, y_test = prepare_split(seed, skew_ratio, minority_count)
    return train_dropout_on_split(
        seed=seed,
        dropout_rate=dropout_rate,
        epochs=epochs,
        skew_ratio=skew_ratio,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def prepare_split(
    seed: int,
    skew_ratio: int,
    minority_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return X_train, X_test, y_train, y_test


def train_dropout_on_split(
    *,
    seed: int,
    dropout_rate: float,
    epochs: int,
    skew_ratio: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float | int]:
    rng = np.random.default_rng(int(seed) + int(round(float(dropout_rate) * 10_000)) + 104729)
    X_balanced, y_balanced = _balanced_resample(X_train, y_train, rng)
    X_epoch, y_epoch = _feature_dropout_batch(X_balanced, y_balanced, dropout_rate, rng)
    model = _make_mlp(seed, max_iter=epochs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_epoch, y_epoch)
    y_score = model.predict_proba(X_test)[:, 1]
    return build_output_row(skew_ratio, dropout_rate, seed, epochs, y_test, y_score)


def run_experiment(
    seeds: list[int],
    dropout_rates: list[float],
    skew_ratios: list[int],
    epochs: int,
    minority_count: int,
    output_path: Path,
    resume: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[int, int, float]] = set()
    if resume and output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as existing:
            reader = csv.DictReader(existing)
            for row in reader:
                completed.add((int(row["skew_ratio"]), int(row["seed"]), float(row["dropout_rate"])))

    mode = "a" if resume and output_path.exists() else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if mode == "w":
            writer.writeheader()
        for skew_ratio in skew_ratios:
            for seed in seeds:
                X_train, X_test, y_train, y_test = prepare_split(seed, skew_ratio, minority_count)
                for dropout_rate in dropout_rates:
                    key = (int(skew_ratio), int(seed), float(dropout_rate))
                    if key in completed:
                        continue
                    writer.writerow(
                        train_dropout_on_split(
                            seed=seed,
                            dropout_rate=dropout_rate,
                            epochs=epochs,
                            skew_ratio=skew_ratio,
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                        )
                    )
                    handle.flush()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in range(20)))
    parser.add_argument("--dropout-rates", type=str, default=",".join(f"{rate:.2f}" for rate in DROPOUT_RATES))
    parser.add_argument("--skew-ratios", type=str, default=",".join(str(skew) for skew in SKEW_RATIOS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--minority-count", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "topology" / "weighted_dropout_sweep_dense.csv",
    )
    args = parser.parse_args()
    output = run_experiment(
        seeds=parse_int_list(args.seeds),
        dropout_rates=parse_float_list(args.dropout_rates),
        skew_ratios=parse_int_list(args.skew_ratios),
        epochs=args.epochs,
        minority_count=args.minority_count,
        output_path=args.output,
        resume=args.resume,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
