# Design

## Package Boundary

The importable package is `hib`. It will contain benchmark infrastructure only, while HDDT estimators are imported from the external `hellinger-tree` package when experiments are implemented.

The initial package exposes only version metadata so imports are smoke-testable without loading optional modeling dependencies.

## Configuration Boundary

Benchmark behavior should be configured through files under `configs/`. Configs should describe datasets, model baselines, experiment suites, metrics, seeds, and output locations.

Configs must remain small, text-based, and suitable for code review. Raw datasets and generated outputs are outside the config boundary.

## Artifact Boundary

Generated benchmark outputs belong under `results/`. Generated report tables and figures belong under `reports/`. Both locations are ignored by default except for `.gitkeep` placeholders.

## Optional Dependencies

Optional baselines such as BalancedRandomForest, XGBoost, and LightGBM should be discovered at runtime and skipped gracefully when unavailable. The package skeleton must not import optional dependencies at module import time.

## Reproducibility

Future experiment runs must capture explicit seeds, package versions, timestamps, dataset identifiers, model identifiers, and split identifiers in structured result records.
