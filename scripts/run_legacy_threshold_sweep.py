"""Run threshold sweeps for curated legacy HDDT datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.datasets.legacy_registry import LEGACY_HDDT_DATASET_REGISTRY  # noqa: E402
from hib.reporting import (  # noqa: E402
    summarize_legacy_threshold_records,
    write_legacy_threshold_summary_markdown,
    write_summary_csv,
)
from hib.runner import load_model_configs, run_legacy_threshold_sweep, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Legacy dataset ids to run (space separated).",
    )
    parser.add_argument(
        "--models",
        action="append",
        default=None,
        type=Path,
        help="Path to model YAML config. May be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        default=ROOT / "results" / "legacy_hddt_threshold_sweep.jsonl",
        type=Path,
        help="Path for threshold-sweep JSONL records.",
    )
    parser.add_argument(
        "--summary-csv",
        default=ROOT / "reports" / "legacy_hddt_threshold_sweep_summary.csv",
        type=Path,
        help="Path for threshold-sweep summary CSV.",
    )
    parser.add_argument(
        "--summary-md",
        default=ROOT / "reports" / "legacy_hddt_threshold_sweep_summary.md",
        type=Path,
        help="Path for threshold-sweep summary markdown.",
    )
    args = parser.parse_args()

    model_paths = args.models or [ROOT / "configs" / "models" / "synthetic_smoke.yaml"]
    model_configs = load_model_configs(model_paths)
    dataset_ids = args.datasets or list(LEGACY_HDDT_DATASET_REGISTRY)

    records = run_legacy_threshold_sweep(
        dataset_ids=dataset_ids,
        model_ids=list(model_configs),
        model_params=model_configs,
    )
    output_path = write_jsonl(records, args.output)
    summary = summarize_legacy_threshold_records(records)
    csv_path = write_summary_csv(summary, args.summary_csv)
    md_path = write_legacy_threshold_summary_markdown(summary, args.summary_md)

    print(f"wrote {output_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
