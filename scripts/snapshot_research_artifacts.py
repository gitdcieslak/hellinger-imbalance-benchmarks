"""Snapshot benchmark artifacts for research reproducibility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.artifacts import snapshot_research_artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Unique run identifier.")
    parser.add_argument("--reports", nargs="*", default=[], type=Path, help="Report files to snapshot.")
    parser.add_argument("--results", nargs="*", default=[], type=Path, help="Result files to snapshot.")
    parser.add_argument("--configs", nargs="*", default=[], type=Path, help="Config files to snapshot.")
    parser.add_argument("--notes", default="", help="Optional notes for this run.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing run directory.")
    args = parser.parse_args()

    run_dir = snapshot_research_artifacts(
        run_id=args.run_id,
        reports=[path.resolve() for path in args.reports],
        results=[path.resolve() for path in args.results],
        configs=[path.resolve() for path in args.configs],
        notes=args.notes,
        command=" ".join(sys.argv),
        workspace_root=ROOT,
        force=args.force,
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
