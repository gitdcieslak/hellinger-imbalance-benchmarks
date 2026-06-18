"""Report local accessibility edge diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from report_accessibility_ridges import DEFAULT_INPUTS, ROOT, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "topology" / "accessibility_edge_summary.md")
    args = parser.parse_args()
    for path in write_report("edge", args.inputs, args.output_md):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
