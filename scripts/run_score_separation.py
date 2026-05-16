"""Run score-separation analysis for synthetic skew benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.reporting import summarize_separation_jsonl  # noqa: E402
from hib.runner import run_score_separation_from_config  # noqa: E402


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
        "--output",
        default=ROOT / "results" / "synthetic_score_separation.jsonl",
        type=Path,
        help="Path for score-separation JSONL records.",
    )
    parser.add_argument(
        "--csv-output",
        default=ROOT / "reports" / "synthetic_score_separation_summary.csv",
        type=Path,
        help="Path for score-separation summary CSV.",
    )
    parser.add_argument(
        "--md-output",
        default=ROOT / "reports" / "synthetic_score_separation_summary.md",
        type=Path,
        help="Path for score-separation summary markdown.",
    )
    args = parser.parse_args()

    model_paths = args.models or [ROOT / "configs" / "models" / "synthetic_smoke.yaml"]
    output_path = run_score_separation_from_config(
        args.config,
        model_paths,
        args.metrics,
        args.output,
    )
    csv_path, markdown_path = summarize_separation_jsonl(
        output_path,
        args.csv_output,
        args.md_output,
    )
    print(f"wrote {output_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
