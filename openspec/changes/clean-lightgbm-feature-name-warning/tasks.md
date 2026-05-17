# Tasks

## 1. Normalization Helpers

- [x] Add `ensure_numpy_array(X)` helper.
- [x] Add `ensure_numpy_vector(y)` helper.
- [x] Add validation behavior for unsupported sparse/object structures.

## 2. Execution Path Integration

- [x] Normalize fit/predict inputs in synthetic benchmark runner.
- [x] Normalize fit/predict inputs in threshold, score, and separation runners.
- [x] Normalize fit/predict inputs in legacy HDDT benchmark runner.
- [x] Normalize inputs in LightGBM diagnostic utility path.

## 3. Tests and Verification

- [x] Add tests for DataFrame normalization and ndarray passthrough.
- [x] Add fit/predict warning capture test for LightGBM (when installed).
- [x] Run `python -m pytest`.
- [x] Run legacy benchmark CLI verification command for `oil`.
