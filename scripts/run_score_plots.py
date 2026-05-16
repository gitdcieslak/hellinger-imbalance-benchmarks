"""Generate score distribution visualization artifacts for synthetic benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.plots import generate_score_distribution_plots  # noqa: E402
from hib.runner import collect_score_distribution_data, load_runner_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=ROOT / "configs" / "experiments" / "synthetic_smoke.yaml",
        type=Path,
        help="Path to the synthetic smoke YAML config.",
    )
    parser.add_argument(
        "--models",
        action="append",
        default=None,
        type=Path,
        help="Path to a model YAML config. May be passed multiple times.",
    )
    parser.add_argument(
        "--metrics",
        default=ROOT / "configs" / "metrics" / "core_imbalance.yaml",
        type=Path,
        help="Path to a metric YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=ROOT / "reports" / "plots",
        type=Path,
        help="Output directory for generated PNG plots.",
    )
    parser.add_argument(
        "--normalization",
        default="class_fraction",
        choices=["class_fraction", "count", "density"],
        help="Histogram normalization mode.",
    )
    parser.add_argument(
        "--include-medium-zoom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include [0, 0.25] zoom histogram variants.",
    )
    parser.add_argument(
        "--no-threshold-overlays",
        action="store_true",
        help="Disable threshold overlay lines in plots.",
    )
    args = parser.parse_args()

    model_paths = args.models or [ROOT / "configs" / "models" / "synthetic_smoke.yaml"]
    config = load_runner_config(args.config, model_paths, args.metrics)
    score_data = collect_score_distribution_data(config)
    thresholds = [] if args.no_threshold_overlays else None
    output_paths = generate_score_distribution_plots(
        score_data,
        output_dir=args.output_dir,
        thresholds=thresholds,
        normalization=args.normalization,
        include_medium_zoom=args.include_medium_zoom,
    )
    for output_path in output_paths:
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
