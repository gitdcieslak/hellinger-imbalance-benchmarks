# Add 5x2 Stratified Evaluation

## Summary

Add repeated 5x2 stratified train/test evaluation to the synthetic benchmark suite and threshold sweep pipeline.

## Motivation

The original HDDT evaluation used five repeated 50/50 train/test splits. For modern imbalance benchmarking, stratified 50/50 splits preserve minority examples better under high skew and provide more stable mean/std estimates than a single split.

## Scope

- Add reusable split generation with `StratifiedShuffleSplit`.
- Default to `n_repeats=5`, `test_size=0.5`, and deterministic split seeds.
- Add split metadata fields to all synthetic and threshold run records.
- Aggregate summaries across all repeats and include `n_runs`.
- Add tests for split generation, split metadata, repeated threshold sweeps, and summary aggregation.

## Non-Goals

- Do not add new models.
- Do not add real datasets.
- Do not add threshold optimization.
- Do not add calibration methods.
