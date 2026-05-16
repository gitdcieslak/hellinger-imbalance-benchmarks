# Tasks

## 1. Model Registry

- [x] Add `xgboost` registry id.
- [x] Add `lightgbm` registry id.
- [x] Add lazy optional dependency handling.
- [x] Keep `hib` importable without boosted-tree packages.

## 2. Configs

- [x] Add `configs/models/boosted_trees.yaml`.

## 3. Tests and Verification

- [x] Add registry presence test.
- [x] Add graceful skip behavior tests for unavailable packages.
- [x] Add fit/predict smoke tests for installed packages.
- [x] Run `python -m pytest`.
- [x] Run `python scripts/run_synthetic_smoke.py`.
