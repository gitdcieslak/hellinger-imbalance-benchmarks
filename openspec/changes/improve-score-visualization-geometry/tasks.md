# Tasks

## 1. Plot Geometry

- [x] Add full/zoom histogram variants.
- [x] Add full/zoom ECDF variants.
- [x] Add optional medium zoom histogram variant.
- [x] Add finer low-score bins for zoom variants.

## 2. Normalization

- [x] Add `class_fraction`, `count`, and `density` histogram normalization modes.
- [x] Make `class_fraction` the default.
- [x] Validate invalid normalization values with clear errors.

## 3. CLI

- [x] Add `--normalization`.
- [x] Add `--include-medium-zoom`.
- [x] Preserve `--no-threshold-overlays` behavior.

## 4. Tests and Verification

- [x] Add deterministic filename variant tests.
- [x] Add class-fraction normalization sum-to-one test.
- [x] Add count normalization count-preservation test.
- [x] Add zoomed histogram/ECDF generation tests.
- [x] Add invalid normalization error test.
- [x] Keep empty-score handling test.
- [x] Run `python -m pytest`.
- [x] Run `python scripts/run_score_plots.py --models configs/models/synthetic_smoke.yaml`.
