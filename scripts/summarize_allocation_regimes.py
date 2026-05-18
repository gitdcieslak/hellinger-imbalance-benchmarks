"""Summarize model-level operational allocation regimes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.regimes import (  # noqa: E402
    allocation_regime_summary_to_markdown,
    combine_regime_metrics,
    load_allocation_summary,
    load_elasticity_summary,
    plot_allocation_regime_scatter,
    summarize_operational_regimes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocation-summary",
        default=ROOT / "reports" / "allocation_concentration_summary.csv",
        type=Path,
    )
    parser.add_argument(
        "--elasticity-summary",
        default=ROOT / "reports" / "threshold_elasticity_summary.csv",
        type=Path,
    )
    parser.add_argument(
        "--output-csv",
        default=ROOT / "reports" / "allocation_regime_summary.csv",
        type=Path,
    )
    parser.add_argument(
        "--output-md",
        default=ROOT / "reports" / "allocation_regime_summary.md",
        type=Path,
    )
    parser.add_argument(
        "--output-json",
        default=ROOT / "reports" / "allocation_regime_summary.json",
        type=Path,
    )
    parser.add_argument(
        "--plots-dir",
        default=ROOT / "reports" / "plots" / "regimes",
        type=Path,
    )
    args = parser.parse_args()

    allocation = load_allocation_summary(args.allocation_summary)
    elasticity = load_elasticity_summary(args.elasticity_summary)
    combined = combine_regime_metrics(allocation, elasticity)
    summary = summarize_operational_regimes(combined)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(allocation_regime_summary_to_markdown(summary), encoding="utf-8")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")

    plot_paths = plot_allocation_regime_scatter(summary, args.plots_dir)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_json}")
    for path in plot_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
