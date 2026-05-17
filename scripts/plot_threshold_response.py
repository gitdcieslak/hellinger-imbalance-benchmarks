"""Plot narrative threshold-response visualizations from summary CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.threshold_plots import (  # noqa: E402
    DEFAULT_METRICS,
    load_threshold_summary_csv,
    plot_threshold_response,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        default=ROOT / "reports" / "legacy_hddt_threshold_sweep_summary.csv",
        type=Path,
        help="Threshold summary CSV path.",
    )
    parser.add_argument("--source", default="legacy", choices=["legacy", "synthetic"], help="Data source label for filenames.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to plot. Defaults to all in summary.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to plot (default: recall f1 precision balanced_accuracy).",
    )
    parser.add_argument(
        "--output-dir",
        default=ROOT / "reports" / "plots" / "threshold_response",
        type=Path,
        help="Output directory for PNG/SVG plots.",
    )
    parser.add_argument("--no-error-bands", action="store_true", help="Disable shaded +/-1 std bands.")
    args = parser.parse_args()

    summary = load_threshold_summary_csv(args.summary_csv)
    created = plot_threshold_response(
        summary=summary,
        source=args.source,
        datasets=args.datasets,
        metrics=args.metrics,
        output_dir=args.output_dir,
        error_bands=not args.no_error_bands,
    )
    print(f"created {len(created)} files")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
