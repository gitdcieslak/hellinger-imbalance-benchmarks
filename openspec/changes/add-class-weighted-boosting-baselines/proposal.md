# Add Class-Weighted Boosting Baselines

## Summary

Add imbalance-aware weighted baseline variants for CART, RandomForest, XGBoost, and LightGBM while preserving existing unweighted baselines.

## Motivation

Current synthetic experiments show strong threshold collapse and score compression for unweighted boosted trees under severe skew. Weighted counterparts are necessary for fair reviewer-facing comparisons and to test whether score geometry shifts under imbalance-aware training.

## Scope

- Add model ids: `cart_balanced`, `random_forest_balanced`, `xgboost_weighted`, `lightgbm_unbalanced`, `lightgbm_weighted`.
- Add split-dependent positive class weighting (`scale_pos_weight = n_negative / n_positive`) for weighted boosting models.
- Add class-weighted model config file.
- Add tests for registry construction, weighting ratio behavior, optional dependency skip behavior, and runner execution.

## Non-Goals

- Do not remove existing unweighted baselines.
- Do not add neural models or real datasets.
- Do not add calibration or threshold optimization.
