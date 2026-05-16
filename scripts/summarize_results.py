"""Summarize synthetic benchmark JSONL results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.reporting import summarize_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=ROOT / "results" / "synthetic_smoke.jsonl",
        type=Path,
        help="Path to synthetic benchmark JSONL results.",
    )
    parser.add_argument(
        "--csv-output",
        default=ROOT / "reports" / "synthetic_smoke_summary.csv",
        type=Path,
        help="Path for the summary CSV report.",
    )
    parser.add_argument(
        "--md-output",
        default=ROOT / "reports" / "synthetic_smoke_summary.md",
        type=Path,
        help="Path for the summary markdown report.",
    )
    args = parser.parse_args()
    csv_path, markdown_path = summarize_jsonl(
        args.input,
        args.csv_output,
        args.md_output,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
