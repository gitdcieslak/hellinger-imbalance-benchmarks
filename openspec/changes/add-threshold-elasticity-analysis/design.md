# Design

## Interval Metrics

For each dataset/model and adjacent threshold interval in relaxation order:

- `0.50->0.25`, `0.25->0.10`, `0.10->0.05`, `0.05->0.01`

compute start/end values, deltas, and elasticity (`delta / abs(threshold_delta)`) for:

- recall
- precision
- f1
- balanced accuracy

## Summary Metrics

Per dataset/model, aggregate:

- `max_recall_jump`
- `max_precision_drop`
- `max_f1_jump`
- mean absolute elasticities for recall/precision/f1
- `threshold_of_max_recall_jump`
- `operational_smoothness_index` (experimental):
  - `1 / (1 + mean_abs_recall_elasticity + mean_abs_precision_elasticity)`

This keeps the index interpretable while preserving raw elasticity metrics.

## Outputs

- `reports/threshold_elasticity_intervals.csv`
- `reports/threshold_elasticity_summary.csv`
- `reports/threshold_elasticity_summary.md`

## CLI

`scripts/analyze_threshold_elasticity.py` supports:

- `--summary-csv`
- `--source legacy|synthetic`
- `--datasets ...`
- `--output-intervals`
- `--output-summary-csv`
- `--output-summary-md`
