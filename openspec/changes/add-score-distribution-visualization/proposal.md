# Add Score Distribution Visualization

## Summary

Add static score-distribution visualization artifacts for synthetic imbalance benchmarks, including class-overlaid histograms and ECDF curves with threshold overlays.

## Motivation

Recent score distribution and separation analyses suggest meaningful differences in score geometry across HDDT, CART, RandomForest, XGBoost, and LightGBM. Visual artifacts are needed for exploratory debugging and publication-ready iteration.

## Scope

- Add score histogram plotting with fixed bins and x-range `[0, 1]`.
- Add class-conditional ECDF plotting.
- Add optional threshold overlays for operational cutoffs.
- Add deterministic plot file naming under `reports/plots/`.
- Add CLI script for generating the plots from existing synthetic infrastructure.
- Add lightweight tests for path determinism, generation, empty-score behavior, and histogram normalization safety.

## Non-Goals

- Do not add interactive plotting.
- Do not add notebooks.
- Do not add new models or real datasets.
