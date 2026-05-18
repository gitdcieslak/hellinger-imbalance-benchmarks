"""Analyze threshold elasticity from threshold sweep summary CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.elasticity import (  # noqa: E402
    compute_threshold_elasticity_intervals,
    load_threshold_summary,
    summarize_threshold_elasticity,
)
from hib.reporting import write_summary_csv, write_threshold_elasticity_summary_markdown  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        default=ROOT / "reports" / "legacy_hddt_threshold_sweep_summary.csv",
        type=Path,
    )
    parser.add_argument("--source", choices=["legacy", "synthetic"], default="legacy")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument(
        "--output-intervals",
        default=ROOT / "reports" / "threshold_elasticity_intervals.csv",
        type=Path,
    )
    parser.add_argument(
        "--output-summary-csv",
        default=ROOT / "reports" / "threshold_elasticity_summary.csv",
        type=Path,
    )
    parser.add_argument(
        "--output-summary-md",
        default=ROOT / "reports" / "threshold_elasticity_summary.md",
        type=Path,
    )
    args = parser.parse_args()

    summary = load_threshold_summary(args.summary_csv)
    intervals = compute_threshold_elasticity_intervals(summary, datasets=args.datasets)
    summary_frame = summarize_threshold_elasticity(intervals)

    intervals_path = write_summary_csv(intervals, args.output_intervals)
    summary_csv_path = write_summary_csv(summary_frame, args.output_summary_csv)
    summary_md_path = write_threshold_elasticity_summary_markdown(summary_frame, args.output_summary_md)

    print(f"wrote {intervals_path}")
    print(f"wrote {summary_csv_path}")
    print(f"wrote {summary_md_path}")


if __name__ == "__main__":
    main()
