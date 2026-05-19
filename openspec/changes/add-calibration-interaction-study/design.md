# Design

## Protocol

Per dataset/model/split:

1. split train portion into fit/calibration subsets (stratified)
2. fit base model on fit subset
3. produce raw scores for calibration subset and test subset
4. derive calibrated test probabilities:
   - Platt scaling (logistic on calibration logits)
   - Isotonic regression (calibration scores -> labels)
5. evaluate threshold sweeps at fixed thresholds:
   - 0.50, 0.25, 0.10, 0.05, 0.01

No train/test leakage is introduced.

## Metrics

For each calibration mode (raw/platt/isotonic):

- threshold metrics (precision/recall/f1/balanced accuracy)
- allocation concentration metrics
- calibration metrics:
  - ECE
  - Brier score
  - calibration slope/intercept
  - reliability curve points

## Outputs

Under `reports/calibration_interaction/`:

- `calibration_study_records.jsonl`
- `calibration_summary_table.csv`
- `regime_persistence_table.csv`
- `calibration_interaction_metrics.csv`
- `calibration_interpretation_summary.md`

Plots under `reports/calibration_interaction/plots/`:

- reliability diagrams by model
- raw/platt/isotonic threshold overlays
- elasticity-shift scatter plots

## Interpretation Focus

Keep conclusions descriptive and empirical:

- does calibration reduce threshold collapse?
- do entropy/support/elasticity/smoothness shift materially?
- do inferred operational regimes persist across calibration modes?
