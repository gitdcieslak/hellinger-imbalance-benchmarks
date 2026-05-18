"""Run allocation concentration analysis for legacy or synthetic sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.datasets.legacy_registry import LEGACY_HDDT_DATASET_REGISTRY  # noqa: E402
from hib.reporting import summarize_allocation_records, write_allocation_summary_markdown, write_summary_csv  # noqa: E402
from hib.runner import (  # noqa: E402
    load_runner_config,
    load_model_configs,
    run_allocation_concentration_legacy,
    run_allocation_concentration_synthetic,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["legacy", "synthetic"], default="legacy")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset ids for legacy source.")
    parser.add_argument(
        "--models",
        action="append",
        default=None,
        type=Path,
        help="Path to model YAML config. May be passed multiple times.",
    )
    parser.add_argument("--output", default=ROOT / "results" / "allocation_concentration.jsonl", type=Path)
    parser.add_argument("--summary-csv", default=ROOT / "reports" / "allocation_concentration_summary.csv", type=Path)
    parser.add_argument("--summary-md", default=ROOT / "reports" / "allocation_concentration_summary.md", type=Path)
    parser.add_argument(
        "--config",
        default=ROOT / "configs" / "experiments" / "synthetic_smoke.yaml",
        type=Path,
        help="Synthetic experiment config path when source=synthetic.",
    )
    parser.add_argument(
        "--metrics",
        default=ROOT / "configs" / "metrics" / "core_imbalance.yaml",
        type=Path,
        help="Metric config path (used for loading synthetic runner config).",
    )
    args = parser.parse_args()

    model_paths = args.models or [ROOT / "configs" / "models" / "synthetic_smoke.yaml"]
    model_configs = load_model_configs(model_paths)

    if args.source == "legacy":
        dataset_ids = args.datasets or list(LEGACY_HDDT_DATASET_REGISTRY)
        records = run_allocation_concentration_legacy(
            dataset_ids=dataset_ids,
            model_ids=list(model_configs),
            model_params=model_configs,
        )
    else:
        config = load_runner_config(args.config, model_paths, args.metrics)
        records = run_allocation_concentration_synthetic(config)

    output_path = write_jsonl(records, args.output)
    summary = summarize_allocation_records(records)
    csv_path = write_summary_csv(summary, args.summary_csv)
    md_path = write_allocation_summary_markdown(summary, args.summary_md)

    print(f"wrote {output_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
