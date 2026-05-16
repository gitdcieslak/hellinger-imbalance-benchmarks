# Tasks

## 1. Model Registry

- [x] Add `cart_balanced`.
- [x] Add `random_forest_balanced`.
- [x] Add `xgboost_weighted`.
- [x] Add `lightgbm_unbalanced`.
- [x] Add `lightgbm_weighted`.

## 2. Split-Dependent Weights

- [x] Add binary class ratio helper.
- [x] Apply split-dependent `scale_pos_weight` in runner before fit.

## 3. Config

- [x] Add `configs/models/class_weighted_trees.yaml`.

## 4. Tests and Verification

- [x] Add registry construction coverage for weighted models.
- [x] Add auto ratio resolution test.
- [x] Add optional weighted skip behavior tests.
- [x] Add weighted model fit/predict tests when available.
- [x] Add weighted runner execution coverage.
- [x] Add merge uniqueness coverage for smoke + weighted config.
- [x] Run `python -m pytest`.
- [x] Run smoke/summary/threshold/score/separation/plot scripts with both model configs.
