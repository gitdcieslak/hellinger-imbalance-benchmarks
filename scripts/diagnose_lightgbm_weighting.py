"""Diagnose LightGBM weighted vs unbalanced equivalence on synthetic data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.diagnostics import (  # noqa: E402
    lightgbm_weighting_diagnostic,
    write_lightgbm_diagnostic_json,
    write_lightgbm_diagnostic_markdown,
)
from hib.models import OptionalDependencyUnavailable  # noqa: E402
from hib.runner import load_runner_config  # noqa: E402
from hib.splits import generate_stratified_repeated_splits  # noqa: E402
from hib.synthetic import SyntheticSkewConfig, make_gaussian_skew_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=ROOT / "configs" / "experiments" / "synthetic_smoke.yaml",
        type=Path,
        help="Experiment config path.",
    )
    parser.add_argument(
        "--models",
        action="append",
        default=None,
        type=Path,
        help="Model config path (optional).",
    )
    parser.add_argument(
        "--metrics",
        default=ROOT / "configs" / "metrics" / "core_imbalance.yaml",
        type=Path,
        help="Metric config path.",
    )
    parser.add_argument(
        "--json-output",
        default=ROOT / "results" / "lightgbm_weighting_diagnostic.json",
        type=Path,
        help="JSON diagnostic output path.",
    )
    parser.add_argument(
        "--md-output",
        default=ROOT / "reports" / "lightgbm_weighting_diagnostic.md",
        type=Path,
        help="Markdown diagnostic output path.",
    )
    args = parser.parse_args()

    model_paths = args.models or [ROOT / "configs" / "models" / "synthetic_smoke.yaml"]
    config = load_runner_config(args.config, model_paths, args.metrics)

    skew_ratio = int(config["skew_ratios"][0])
    seed = int(config["seeds"][0])
    dataset_config = SyntheticSkewConfig(
        skew_ratio=skew_ratio,
        seed=seed,
        separation=float(config["separation"]),
        minority_count=int(config["minority_count"]),
        n_features=int(config["n_features"]),
        noise=float(config["noise"]),
        test_size=float(config["test_size"]),
    )
    X, y = make_gaussian_skew_dataset(dataset_config)
    split = generate_stratified_repeated_splits(
        y,
        n_repeats=int(config.get("n_repeats", 5)),
        test_size=float(config["test_size"]),
        random_seed=int(config.get("split_seed", 0)),
    )[0]

    X_train = X[split.train_idx]
    y_train = y[split.train_idx]
    X_test = X[split.test_idx]
    y_test = y[split.test_idx]

    try:
        report = lightgbm_weighting_diagnostic(X_train, y_train, X_test, y_test, seed=seed)
    except OptionalDependencyUnavailable:
        print("lightgbm not available; skipping diagnostic")
        return

    json_path = write_lightgbm_diagnostic_json(report, args.json_output)
    md_path = write_lightgbm_diagnostic_markdown(report, args.md_output)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
