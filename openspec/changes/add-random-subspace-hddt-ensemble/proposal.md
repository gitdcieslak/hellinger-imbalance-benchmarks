# Add Random-Subspace HDDT Ensemble

## Summary

Add an HDDT-based random-subspace ensemble baseline using `BaggingClassifier` with feature subspace sampling.

## Motivation

This baseline tests whether Hellinger-based splitting remains strong when combined with random subspace behavior similar to parts of RandomForest.

## Scope

- Add `hddt_forest` model id.
- Add optional alias `random_subspace_hddt` mapped to the same implementation.
- Add split-aware `max_features` resolution for `sqrt`, `log2`, `null`, float, and int specs.
- Add `configs/models/hddt_ensembles.yaml`.
- Add tests and end-to-end runner verification.

## Non-Goals

- No changes to HDDT estimator internals.
- No custom ensemble class unless required.
