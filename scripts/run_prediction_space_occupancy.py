"""Run prediction-space occupancy analysis and generate artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.datasets.legacy_registry import LEGACY_HDDT_DATASET_REGISTRY  # noqa: E402
from hib.occupancy_plots import plot_occupancy_artifacts  # noqa: E402
from hib.reporting import summarize_occupancy_records, write_occupancy_summary_markdown, write_summary_csv  # noqa: E402
from hib.runner import load_model_configs, run_prediction_space_occupancy_legacy, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["legacy"], default="legacy")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", action="append", default=None, type=Path)
    parser.add_argument("--output", default=ROOT / "results" / "prediction_space_occupancy.jsonl", type=Path)
    parser.add_argument("--summary-csv", default=ROOT / "reports" / "prediction_space_occupancy_summary.csv", type=Path)
    parser.add_argument("--summary-md", default=ROOT / "reports" / "prediction_space_occupancy_summary.md", type=Path)
    parser.add_argument("--plots-dir", default=ROOT / "reports" / "plots" / "occupancy", type=Path)
    args = parser.parse_args()

    model_paths = args.models or [ROOT / "configs" / "models" / "synthetic_smoke.yaml"]
    model_configs = load_model_configs(model_paths)
    dataset_ids = args.datasets or list(LEGACY_HDDT_DATASET_REGISTRY)

    records = run_prediction_space_occupancy_legacy(
        dataset_ids=dataset_ids,
        model_ids=list(model_configs),
        extracted_dir=ROOT / "data" / "extracted" / "legacy_hddt",
        model_params=model_configs,
    )
    output_path = write_jsonl(records, args.output)
    summary = summarize_occupancy_records(records)
    csv_path = write_summary_csv(summary, args.summary_csv)
    md_path = write_occupancy_summary_markdown(summary, args.summary_md)
    plot_paths = plot_occupancy_artifacts(records, args.plots_dir)

    print(f"wrote {output_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    for path in plot_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
