"""Generate posterior occupancy figure family for legacy datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.datasets.legacy_registry import LEGACY_HDDT_DATASET_REGISTRY  # noqa: E402
from hib.occupancy_figures import (  # noqa: E402
    collect_legacy_posterior_records,
    generate_posterior_occupancy_figure_family,
)
from hib.runner import load_model_configs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["legacy"], default="legacy")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", action="append", default=None, type=Path)
    parser.add_argument("--output-dir", default=ROOT / "reports" / "posterior_occupancy_figures", type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.50, 0.25, 0.10, 0.05, 0.01])
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--round-decimals", type=int, default=6)
    parser.add_argument("--max-plots-per-model", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_paths = args.models or [ROOT / "configs" / "models" / "synthetic_smoke.yaml"]
    model_configs = load_model_configs(model_paths)
    dataset_ids = args.datasets or list(LEGACY_HDDT_DATASET_REGISTRY)

    posterior_records = collect_legacy_posterior_records(
        dataset_ids=dataset_ids,
        model_ids=list(model_configs),
        extracted_dir=ROOT / "data" / "extracted" / "legacy_hddt",
        model_params=model_configs,
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        split_seed=args.split_seed,
        seed=args.seed,
    )
    outputs = generate_posterior_occupancy_figure_family(
        posterior_records=posterior_records,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
        grid_size=args.grid_size,
        round_decimals=args.round_decimals,
        max_plots_per_model=args.max_plots_per_model,
    )

    print(f"wrote {outputs['summary_csv']}")
    print(f"wrote {outputs['summary_md']}")
    print(f"generated {len(outputs['ecdf_paths'])} ecdf files")
    print(f"generated {len(outputs['reachability_paths'])} reachability files")
    print(f"generated {len(outputs['support_paths'])} support files")


if __name__ == "__main__":
    main()
