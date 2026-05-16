# Add Boosted Tree Baselines

## Summary

Add optional XGBoost and LightGBM baseline support to the model registry for synthetic benchmark runs.

## Motivation

Modern boosted-tree systems are important baselines for imbalance experiments, but they should not become hard dependencies of the benchmark package. Users should be able to import `hib` and run non-boosted smoke tests without installing XGBoost or LightGBM.

## Scope

- Add registry ids `xgboost` and `lightgbm`.
- Lazily import optional packages only when those models are instantiated.
- Skip unavailable optional models gracefully in the runner.
- Add config examples for boosted-tree defaults.
- Add smoke tests covering registry presence, unavailable-package behavior, and fit/predict behavior when packages are installed.

## Non-Goals

- Do not add neural models.
- Do not add real datasets.
- Do not add large benchmark runs.
- Do not commit generated results.
