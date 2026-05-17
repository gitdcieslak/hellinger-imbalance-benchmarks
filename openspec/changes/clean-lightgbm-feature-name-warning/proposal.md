# Clean LightGBM Feature Name Warning

## Summary

Standardize runner model execution inputs to `numpy.ndarray` and remove feature-name mismatch warnings observed with sklearn/LightGBM wrappers.

## Motivation

Current benchmark runs emit warnings about feature-name mismatch during prediction. While likely harmless, they indicate inconsistent container semantics and clutter benchmark output.

## Scope

- Add centralized array normalization helpers.
- Apply normalization in synthetic, threshold, score-analysis, score-separation, and legacy benchmark runner paths.
- Apply the same normalization in LightGBM diagnostic utilities.
- Add tests for normalization behavior and LightGBM warning absence (when installed).

## Non-Goals

- No benchmark logic refactor unrelated to container normalization.
- No model hyperparameter changes.
- No DataFrame-dependent execution paths.
