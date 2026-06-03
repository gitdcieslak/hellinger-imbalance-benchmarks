"""Run weighted BCE x positive oversampling allocation morphology grid."""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
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


WEIGHT_MULTIPLIERS = [1, 2, 5, 10]
OVERSAMPLING_MULTIPLIERS = [1, 2, 5, 10]
OUTPUT_COLUMNS = [
    "class_weight_multiplier",
    "oversampling_multiplier",
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


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _make_mlp(seed: int, max_iter: int) -> MLPClassifier:
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


def _oversample_positives(
    X: np.ndarray,
    y: np.ndarray,
    multiplier: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if multiplier <= 1:
        return X, y
    pos_idx = np.flatnonzero(y == 1)
    if pos_idx.size == 0:
        return X, y
    extra_idx = rng.choice(pos_idx, size=pos_idx.size * (int(multiplier) - 1), replace=True)
    idx = np.concatenate([np.arange(y.size), extra_idx])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _sample_weights(y: np.ndarray, multiplier: int) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    n_total = int(y.size)
    weights = {int(cls): n_total / (2.0 * float(count)) for cls, count in zip(classes, counts, strict=False)}
    arr = np.asarray([weights[int(label)] for label in y], dtype=float)
    arr[y == 1] *= float(multiplier)
    return arr


def build_output_row(
    class_weight_multiplier: int,
    oversampling_multiplier: int,
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
        "class_weight_multiplier": int(class_weight_multiplier),
        "oversampling_multiplier": int(oversampling_multiplier),
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


def prepare_split(seed: int, skew_ratio: int, minority_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def train_grid_cell(
    *,
    class_weight_multiplier: int,
    oversampling_multiplier: int,
    seed: int,
    epoch: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float | int]:
    rng = np.random.default_rng(
        int(seed) + 1009 * int(class_weight_multiplier) + 7919 * int(oversampling_multiplier)
    )
    X_fit, y_fit = _oversample_positives(X_train, y_train, oversampling_multiplier, rng)
    model = _make_mlp(seed, max_iter=epoch)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if "sample_weight" in inspect.signature(model.fit).parameters:
            model.fit(X_fit, y_fit, sample_weight=_sample_weights(y_fit, class_weight_multiplier))
        else:
            model.fit(X_fit, y_fit)
    y_score = model.predict_proba(X_test)[:, 1]
    return build_output_row(
        class_weight_multiplier,
        oversampling_multiplier,
        seed,
        epoch,
        y_test,
        y_score,
    )


def run_experiment(
    *,
    seeds: list[int],
    weight_multipliers: list[int],
    oversampling_multipliers: list[int],
    epochs: int,
    skew_ratio: int,
    minority_count: int,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for seed in seeds:
            X_train, X_test, y_train, y_test = prepare_split(seed, skew_ratio, minority_count)
            for class_weight_multiplier in weight_multipliers:
                for oversampling_multiplier in oversampling_multipliers:
                    writer.writerow(
                        train_grid_cell(
                            class_weight_multiplier=class_weight_multiplier,
                            oversampling_multiplier=oversampling_multiplier,
                            seed=seed,
                            epoch=epochs,
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
    parser.add_argument("--class-weight-multipliers", type=str, default=",".join(str(v) for v in WEIGHT_MULTIPLIERS))
    parser.add_argument("--oversampling-multipliers", type=str, default=",".join(str(v) for v in OVERSAMPLING_MULTIPLIERS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--skew-ratio", type=int, default=100)
    parser.add_argument("--minority-count", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "topology" / "weighted_oversampling_grid.csv",
    )
    args = parser.parse_args()
    output = run_experiment(
        seeds=parse_int_list(args.seeds),
        weight_multipliers=parse_int_list(args.class_weight_multipliers),
        oversampling_multipliers=parse_int_list(args.oversampling_multipliers),
        epochs=args.epochs,
        skew_ratio=args.skew_ratio,
        minority_count=args.minority_count,
        output_path=args.output,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
