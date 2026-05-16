"""Run the synthetic Gaussian skew smoke benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.runner import run_from_config  # noqa: E402


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
    args = parser.parse_args()
    model_paths = args.models or [ROOT / "configs" / "models" / "synthetic_smoke.yaml"]
    output_path = run_from_config(args.config, model_paths, args.metrics)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
