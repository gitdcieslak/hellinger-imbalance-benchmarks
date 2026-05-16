# Design

## Data Flow

The runner provides in-memory aggregated class-conditional score arrays by `(skew_ratio, model_id)` across repeated stratified splits. No raw per-row predictions are persisted.

## Plot Types

For each model and skew ratio:

- overlaid positive/negative score histogram (fixed bins)
- positive/negative ECDF curves

Both plot types use fixed x-limits `[0, 1]` and optional threshold overlays at:

- 0.01
- 0.05
- 0.10
- 0.25
- 0.50

## Output Naming

Deterministic file names:

- `score_histogram_{model_id}_skew_{ratio}.png`
- `score_ecdf_{model_id}_skew_{ratio}.png`

Outputs are written to `reports/plots/` and remain gitignored.
