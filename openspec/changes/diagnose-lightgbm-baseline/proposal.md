# Diagnose LightGBM Baseline

## Summary

Investigate and fix LightGBM producing near-constant 0.5 probabilities in synthetic smoke runs.

## Motivation

Score and threshold analyses show cases where LightGBM emits uninformative probabilities on small synthetic settings, which obscures meaningful threshold-response comparisons with HDDT, CART, RandomForest, and other boosted baselines.

## Scope

- Add fit/predict diagnostics through regression tests on separable synthetic data.
- Adjust LightGBM smoke defaults for small datasets.
- Keep model registry and benchmark flow unchanged outside LightGBM parameter tuning.

## Non-Goals

- Do not add new models.
- Do not add real datasets.
- Do not add calibration methods.
