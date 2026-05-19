"""Create consolidated paper-facing operational synthesis markdown report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.synthesis import write_paper_synthesis_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--occupancy-summary",
        default=ROOT / "reports" / "prediction_space_occupancy_summary.csv",
        type=Path,
    )
    parser.add_argument(
        "--elasticity-summary",
        default=ROOT / "reports" / "threshold_elasticity_summary.csv",
        type=Path,
    )
    parser.add_argument(
        "--regime-summary",
        default=ROOT / "reports" / "allocation_regime_summary.csv",
        type=Path,
    )
    parser.add_argument(
        "--calibration-summary",
        default=ROOT / "reports" / "calibration_interaction" / "calibration_summary_table.csv",
        type=Path,
    )
    parser.add_argument(
        "--regime-persistence",
        default=ROOT / "reports" / "calibration_interaction" / "regime_persistence_table.csv",
        type=Path,
    )
    parser.add_argument(
        "--output-md",
        default=ROOT / "reports" / "paper_operational_synthesis.md",
        type=Path,
    )
    args = parser.parse_args()

    output_path = write_paper_synthesis_report(
        occupancy_summary_path=args.occupancy_summary,
        elasticity_summary_path=args.elasticity_summary,
        regime_summary_path=args.regime_summary,
        calibration_summary_path=args.calibration_summary,
        regime_persistence_path=args.regime_persistence,
        output_markdown_path=args.output_md,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
