# Design

## Array Standardization

Add `ensure_numpy_array(X)` and `ensure_numpy_vector(y)` in `src/hib/arrays.py`.

Behavior:

- pass through `ndarray` without copy
- convert list/DataFrame-like containers via `np.asarray`
- enforce 2D features and 1D labels
- reject sparse and object-dtype arrays with clear errors

## Runner Integration

Normalize split outputs (`X_train`, `X_test`, `y_train`, `y_test`) before fit/predict operations in:

- synthetic suite
- threshold sweep suite
- score analysis suite
- score separation suite
- score distribution collection
- legacy HDDT benchmark suite

## LightGBM Warning Suppression

After `model.fit(...)` in the shared fit helper, remove `feature_names_in_` when present and runner input is `ndarray`.

This keeps container policy model-agnostic and prevents sklearn's feature-name warning for wrappers that may set names internally.

## Diagnostics

Normalize array inputs for LightGBM diagnostic fits/predicts to ensure consistency with benchmark paths.
