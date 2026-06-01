# Bagged HDDT Next Experiments

## Objective

Reduce uncertainty in the Bagged HDDT mechanism claim by testing whether the observed cliff-and-recovery profile is driven by ensemble vote concentration, calibration shape, or dataset-specific instability.

## Experiment A: Per-instance vote dispersion audit

- Question: Does Bagged HDDT produce concentrated ensemble votes near decision boundaries that release rapidly under threshold relaxation?
- Method:
  - For each dataset, extract per-instance tree vote proportions (or per-tree probability outputs) for `hddt_forest`, `random_forest`, `lightgbm`.
  - Compute vote entropy and margin distribution (`|p-0.5|`) by true class.
  - Compare vote-entropy quantiles in instances that flip from FN->TP between threshold bands.
- Expected discriminator:
  - If Bagged HDDT has lower vote entropy but dense mass just below strict thresholds, this supports the gate-release mechanism.
- Deliverable: `reports/neural_mlp/bagged_hddt_vote_dispersion.csv` + figure panel for margin densities.

## Experiment B: Threshold-local calibration diagnostics

- Question: Is the large recall jump caused by calibration misalignment in a narrow score interval?
- Method:
  - Build reliability curves and ECE/Brier by threshold band (`[0.50,0.25]`, `[0.25,0.10]`, `[0.10,0.05]`, `[0.05,0.01]`).
  - Evaluate class-conditional calibration errors in each band.
- Expected discriminator:
  - A localized calibration break around `0.10->0.05` would explain middle-band jumps.
- Deliverable: `reports/neural_mlp/bagged_hddt_band_calibration.csv` + calibration appendix plot.

## Experiment C: Bootstrap stability and dataset heterogeneity

- Question: Are Bagged HDDT dynamics stable or dominated by one dataset?
- Method:
  - Bootstrap datasets/runs to estimate confidence intervals for Recovery, Jump, Smoothness, Persistence deltas vs HDDT and LightGBM.
  - Leave-one-dataset-out sensitivity analysis for each metric.
- Expected discriminator:
  - If intervals remain separated from comparators under resampling, mechanism claim is robust.
- Deliverable: `reports/neural_mlp/bagged_hddt_stability_ci.csv` + robustness table.

## Prioritization

1. Run Experiment C first (fastest confidence upgrade for paper claims).
2. Run Experiment B second (directly tests threshold-local explanation).
3. Run Experiment A third (strongest mechanism evidence, highest implementation effort).

## Minimal Acceptance Criteria

- C: Recovery and Jump deltas vs HDDT remain positive with non-overlapping 95% CIs in most resamples.
- B: Bandwise calibration error identifies at least one interval aligned with observed jump concentration.
- A: Flip-set instances show distinct vote entropy/margin profile vs non-flip instances.

## If No New Runs Are Possible

Use existing artifacts to publish a constrained claim tier:

- Tier 1 (safe): operational non-equivalence and regime placement (already supported).
- Tier 2 (conditional): mechanism interpretation as plausible explanation requiring per-instance vote analysis.
