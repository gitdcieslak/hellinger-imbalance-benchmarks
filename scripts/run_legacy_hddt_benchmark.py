"""Run curated legacy HDDT benchmark datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.datasets.legacy_registry import LEGACY_HDDT_DATASET_REGISTRY  # noqa: E402
from hib.runner import (  # noqa: E402
    load_model_configs,
    run_legacy_hddt_benchmark,
    summarize_legacy_records,
    write_jsonl,
    write_legacy_summary_markdown,
)


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
        type=Path,
        default=ROOT / "results" / "legacy_hddt_benchmark.jsonl",
        help="JSONL output path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "reports" / "legacy_hddt_benchmark_summary.csv",
        help="Summary CSV output path.",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "reports" / "legacy_hddt_benchmark_summary.md",
        help="Summary markdown output path.",
    )
    args = parser.parse_args()

    model_paths = args.models or [ROOT / "configs" / "models" / "synthetic_smoke.yaml"]
    model_configs = load_model_configs(model_paths)
    dataset_ids = args.datasets or list(LEGACY_HDDT_DATASET_REGISTRY)

    records = run_legacy_hddt_benchmark(
        dataset_ids=dataset_ids,
        model_ids=list(model_configs),
        model_params=model_configs,
    )
    jsonl_path = write_jsonl(records, args.output)

    summary = summarize_legacy_records(records)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    md_path = write_legacy_summary_markdown(summary, args.summary_md)

    print(f"wrote {jsonl_path}")
    print(f"wrote {args.summary_csv.resolve()}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
