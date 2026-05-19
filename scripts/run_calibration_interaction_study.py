"""Run calibration interaction study for operational allocation geometry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.calibration_study import (  # noqa: E402
    plot_elasticity_shift,
    plot_pre_post_threshold_curves,
    plot_reliability_diagrams,
    run_calibration_interaction_legacy,
    write_calibration_study_artifacts,
)
from hib.datasets.legacy_registry import LEGACY_HDDT_DATASET_REGISTRY  # noqa: E402
from hib.runner import load_model_configs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["legacy"], default="legacy")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", action="append", default=None, type=Path)
    parser.add_argument("--output-dir", default=ROOT / "reports" / "calibration_interaction", type=Path)
    args = parser.parse_args()

    model_paths = args.models or [ROOT / "configs" / "models" / "synthetic_smoke.yaml"]
    model_configs = load_model_configs(model_paths)
    dataset_ids = args.datasets or list(LEGACY_HDDT_DATASET_REGISTRY)

    records = run_calibration_interaction_legacy(
        dataset_ids=dataset_ids,
        model_ids=list(model_configs),
        extracted_dir=ROOT / "data" / "extracted" / "legacy_hddt",
        model_params=model_configs,
    )
    artifact_paths = write_calibration_study_artifacts(records, args.output_dir)

    plot_dir = args.output_dir / "plots"
    reliability_paths = plot_reliability_diagrams(records, plot_dir / "reliability")
    threshold_paths = plot_pre_post_threshold_curves(records, plot_dir / "threshold_overlays")
    elasticity_paths = plot_elasticity_shift(records, plot_dir / "elasticity_shifts")

    for key, path in artifact_paths.items():
        print(f"wrote {key}: {path}")
    for path in reliability_paths + threshold_paths + elasticity_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
