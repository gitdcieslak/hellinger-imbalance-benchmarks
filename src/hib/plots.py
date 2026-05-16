"""Score distribution visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from hib.scores import HISTOGRAM_BINS
from hib.thresholds import DEFAULT_THRESHOLDS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOW_SCORE_BINS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.075, 0.10]
MEDIUM_SCORE_BINS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
VALID_NORMALIZATIONS = {"class_fraction", "count", "density"}


def skew_ratio_slug(skew_ratio: str) -> str:
    """Convert a skew ratio label like '100:1' to '100'."""

    return str(skew_ratio).split(":", maxsplit=1)[0]


def deterministic_plot_paths(model_id: str, skew_ratio: str, output_dir: str | Path) -> dict[str, Path]:
    """Build deterministic file paths for score distribution plots."""

    skew_slug = skew_ratio_slug(skew_ratio)
    base_dir = Path(output_dir)
    return {
        "histogram_full": base_dir / f"score_histogram_full_{model_id}_skew_{skew_slug}.png",
        "histogram_zoom_010": base_dir / f"score_histogram_zoom_010_{model_id}_skew_{skew_slug}.png",
        "histogram_zoom_025": base_dir / f"score_histogram_zoom_025_{model_id}_skew_{skew_slug}.png",
        "ecdf_full": base_dir / f"score_ecdf_full_{model_id}_skew_{skew_slug}.png",
        "ecdf_zoom_010": base_dir / f"score_ecdf_zoom_010_{model_id}_skew_{skew_slug}.png",
        # Backward compatibility aliases
        "histogram": base_dir / f"score_histogram_full_{model_id}_skew_{skew_slug}.png",
        "ecdf": base_dir / f"score_ecdf_full_{model_id}_skew_{skew_slug}.png",
    }


def _validate_normalization(normalization: str) -> None:
    if normalization not in VALID_NORMALIZATIONS:
        known = ", ".join(sorted(VALID_NORMALIZATIONS))
        raise ValueError(f"invalid normalization {normalization!r}; expected one of: {known}")


def normalized_histogram(
    scores: np.ndarray,
    bins: list[float] | None = None,
    normalization: str = "class_fraction",
) -> np.ndarray:
    """Compute histogram values under count/class-fraction/density modes."""

    _validate_normalization(normalization)
    bin_edges = HISTOGRAM_BINS if bins is None else bins
    arr = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    counts, _ = np.histogram(arr, bins=bin_edges)

    if normalization == "count":
        return counts.astype(float)

    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)

    if normalization == "density":
        widths = np.diff(np.asarray(bin_edges, dtype=float))
        return counts.astype(float) / (float(total) * widths)

    return counts.astype(float) / float(total)


def _require_non_empty_scores(positive_scores: np.ndarray, negative_scores: np.ndarray) -> None:
    if np.asarray(positive_scores).size == 0 or np.asarray(negative_scores).size == 0:
        raise ValueError("positive_scores and negative_scores must both be non-empty")


def _draw_threshold_lines(thresholds: list[float]) -> None:
    for threshold in thresholds:
        plt.axvline(float(threshold), color="gray", linestyle="--", linewidth=0.8, alpha=0.6)


def plot_score_histogram(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    output_path: str | Path,
    thresholds: list[float] | None = None,
    bins: list[float] | None = None,
    normalization: str = "class_fraction",
    x_range: tuple[float, float] = (0.0, 1.0),
    model_id: str = "unknown",
    skew_ratio: str = "unknown",
) -> Path:
    """Plot overlaid positive/negative score histograms."""

    _require_non_empty_scores(positive_scores, negative_scores)
    _validate_normalization(normalization)
    threshold_values = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    bin_edges = HISTOGRAM_BINS if bins is None else bins

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)

    density = normalization == "density"
    pos_weights = None
    neg_weights = None
    if normalization == "class_fraction":
        pos_weights = np.full(pos.shape[0], 1.0 / pos.shape[0], dtype=float)
        neg_weights = np.full(neg.shape[0], 1.0 / neg.shape[0], dtype=float)

    plt.figure(figsize=(8, 5))
    plt.hist(
        neg,
        bins=bin_edges,
        alpha=0.55,
        density=density,
        weights=neg_weights,
        label="negative",
    )
    plt.hist(
        pos,
        bins=bin_edges,
        alpha=0.55,
        density=density,
        weights=pos_weights,
        label="positive",
    )
    _draw_threshold_lines(threshold_values)
    plt.xlim(*x_range)
    plt.xlabel("Predicted positive-class probability")
    y_label = {
        "count": "Count",
        "class_fraction": "Class fraction",
        "density": "Density",
    }[normalization]
    plt.ylabel(y_label)
    plt.title(
        f"Score Histogram: {model_id}, skew {skew_ratio}, "
        f"range [{x_range[0]:.2f}, {x_range[1]:.2f}], {normalization.replace('_', '-')}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, arr.size + 1, dtype=float) / float(arr.size)
    return arr, y


def plot_score_ecdf(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    output_path: str | Path,
    thresholds: list[float] | None = None,
    x_range: tuple[float, float] = (0.0, 1.0),
    model_id: str = "unknown",
    skew_ratio: str = "unknown",
) -> Path:
    """Plot class-conditional empirical CDF curves."""

    _require_non_empty_scores(positive_scores, negative_scores)
    threshold_values = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pos_x, pos_y = _ecdf(np.asarray(positive_scores))
    neg_x, neg_y = _ecdf(np.asarray(negative_scores))

    plt.figure(figsize=(8, 5))
    plt.plot(neg_x, neg_y, label="negative")
    plt.plot(pos_x, pos_y, label="positive")
    _draw_threshold_lines(threshold_values)
    plt.xlim(*x_range)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Predicted positive-class probability")
    plt.ylabel("ECDF")
    plt.title(f"Score ECDF: {model_id}, skew {skew_ratio}, range [{x_range[0]:.2f}, {x_range[1]:.2f}]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def generate_score_distribution_plots(
    score_data: list[dict[str, Any]],
    output_dir: str | Path = "reports/plots",
    thresholds: list[float] | None = None,
    normalization: str = "class_fraction",
    include_medium_zoom: bool = True,
) -> list[Path]:
    """Generate histogram and ECDF plots for each model/skew pair."""

    _validate_normalization(normalization)
    created_paths: list[Path] = []
    for item in sorted(score_data, key=lambda row: (row["skew_ratio"], row["model_id"])):
        paths = deterministic_plot_paths(item["model_id"], item["skew_ratio"], output_dir)
        created_paths.append(
            plot_score_histogram(
                item["positive_scores"],
                item["negative_scores"],
                paths["histogram_full"],
                thresholds=thresholds,
                bins=HISTOGRAM_BINS,
                normalization=normalization,
                x_range=(0.0, 1.0),
                model_id=item["model_id"],
                skew_ratio=item["skew_ratio"],
            )
        )
        created_paths.append(
            plot_score_histogram(
                item["positive_scores"],
                item["negative_scores"],
                paths["histogram_zoom_010"],
                thresholds=thresholds,
                bins=LOW_SCORE_BINS,
                normalization=normalization,
                x_range=(0.0, 0.10),
                model_id=item["model_id"],
                skew_ratio=item["skew_ratio"],
            )
        )
        if include_medium_zoom:
            created_paths.append(
                plot_score_histogram(
                    item["positive_scores"],
                    item["negative_scores"],
                    paths["histogram_zoom_025"],
                    thresholds=thresholds,
                    bins=MEDIUM_SCORE_BINS,
                    normalization=normalization,
                    x_range=(0.0, 0.25),
                    model_id=item["model_id"],
                    skew_ratio=item["skew_ratio"],
                )
            )
        created_paths.append(
            plot_score_ecdf(
                item["positive_scores"],
                item["negative_scores"],
                paths["ecdf_full"],
                thresholds=thresholds,
                x_range=(0.0, 1.0),
                model_id=item["model_id"],
                skew_ratio=item["skew_ratio"],
            )
        )
        created_paths.append(
            plot_score_ecdf(
                item["positive_scores"],
                item["negative_scores"],
                paths["ecdf_zoom_010"],
                thresholds=thresholds,
                x_range=(0.0, 0.10),
                model_id=item["model_id"],
                skew_ratio=item["skew_ratio"],
            )
        )
    return created_paths
