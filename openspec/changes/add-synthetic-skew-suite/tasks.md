# Tasks

## 1. Synthetic Data

- [x] Add deterministic Gaussian skew generator.
- [x] Add configurable seeds, separation, noise, feature count, skew ratio, and train/test split.

## 2. Models and Metrics

- [x] Add model registry for HDDT, Bagged HDDT, CART, and RandomForest.
- [x] Add core imbalance metric computation.

## 3. Runner and Configs

- [x] Add YAML config examples under `configs/`.
- [x] Add JSONL-emitting synthetic suite runner.
- [x] Add `scripts/run_synthetic_smoke.py`.

## 4. Tests and Verification

- [x] Add deterministic generator test.
- [x] Add model registry instantiation test.
- [x] Add metrics computation test.
- [x] Add smoke JSONL output test.
- [x] Run `pip install -e ".[dev]"`.
- [x] Run `python -m pytest`.
- [x] Run `python scripts/run_synthetic_smoke.py`.
