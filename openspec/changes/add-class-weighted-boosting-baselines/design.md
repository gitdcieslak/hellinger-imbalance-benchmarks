# Design

## Registry Additions

New ids are added to the existing model registry:

- `cart_balanced`: `DecisionTreeClassifier(class_weight="balanced")`
- `random_forest_balanced`: `RandomForestClassifier(class_weight="balanced_subsample")`
- `xgboost_weighted`: `XGBClassifier(..., scale_pos_weight=1.0)` with per-split override
- `lightgbm_unbalanced`: `LGBMClassifier(..., is_unbalance=True)`
- `lightgbm_weighted`: `LGBMClassifier(..., scale_pos_weight=1.0)` with per-split override

`balanced_subsample` is chosen for RandomForest to align class reweighting with each bootstrap sample.

## Split-Dependent Weighting

Add helper:

- `compute_binary_class_weight_ratio(y_train, positive_label=1) -> float`

Runner calls `apply_split_dependent_model_params(model, model_id, y_train)` before fit to set `scale_pos_weight` for weighted boosting variants.

If a split has zero positives or zero negatives, a clear `ValueError` is raised.

## Optional Dependencies

Weighted XGBoost and LightGBM variants use the existing optional dependency mechanism and skip gracefully when packages are unavailable.
