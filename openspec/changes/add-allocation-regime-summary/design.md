# Design

## Inputs

Use existing report artifacts only:

- `reports/allocation_concentration_summary.csv`
- `reports/threshold_elasticity_summary.csv`

## Combination Strategy

1. Load allocation + elasticity summaries.
2. Join on `(dataset_id, model_id)`.
3. Aggregate to model-level means and interval-frequency fractions.

## Model-Level Regime Metrics

- mean entropy/support/gini
- mean fractions below 0.01 and 0.05
- mean recall and precision elasticity
- mean smoothness index
- mean max recall jump and max precision drop
- modal threshold interval of max recall jump
- fraction of datasets with max jump in each interval bucket

## Heuristic Regime Labels

Use explicit, interpretable heuristics (exploratory):

- `quantized_allocator`
- `conservative_allocator`
- `cliff_allocator`
- `smooth_allocator`
- `broad_allocator`

No unsupervised clustering is used.

## Outputs

- `reports/allocation_regime_summary.csv`
- `reports/allocation_regime_summary.md`
- `reports/allocation_regime_summary.json` (optional convenience output)

## Visuals

Matplotlib-only scatter plots:

- support vs recall elasticity
- entropy vs smoothness

under `reports/plots/regimes/`.
