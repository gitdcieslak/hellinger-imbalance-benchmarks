# Design

## Diagnostic Inputs

Use one deterministic synthetic split and train:

- `lightgbm`
- `lightgbm_unbalanced`
- `lightgbm_weighted`

## Captured Data

- resolved parameters (`is_unbalance`, `scale_pos_weight`, etc.)
- train class counts and computed ratio
- pairwise probability-difference metrics:
  - max absolute diff
  - correlation
  - identical prediction count

## Outputs

- `reports/lightgbm_weighting_diagnostic.md`
- `results/lightgbm_weighting_diagnostic.json`

If LightGBM is unavailable, the CLI skips gracefully.
