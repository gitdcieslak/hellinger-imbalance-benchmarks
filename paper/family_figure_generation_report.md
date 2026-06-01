# Family Figure Generation Report

## Generated Artifacts

- `paper/manuscript_figures/figure_family_reachability_comparison.png`
- `paper/manuscript_figures/figure_ranking_vs_accessibility_scatter.png`
- `paper/manuscript_tables/table_allocator_family_summary.md`

All assets were generated from existing `reports/neural_mlp/*.csv` artifacts. No new model training or experiments were run.

## Figure A — Model Family Reachability Comparison

Path:

- `paper/manuscript_figures/figure_family_reachability_comparison.png`

Data sources:

- `reports/neural_mlp/legacy_threshold_sweep_summary.csv`

Included families:

- CART
- HDDT
- Bagged HDDT (`hddt_forest` artifact ID)
- Random Forest
- XGBoost
- LightGBM
- MLP (included as optional reference)

Caption (manuscript-ready):

- Family-level reachability trajectories across thresholds (`0.50 -> 0.01`) show distinct operational accessibility morphologies under shared severe-imbalance evaluation, including quantized, cliff-like, conservative, and broader persistence behaviors; these differences appear despite overlapping ranking-quality ranges.

Primary claim:

- Operational accessibility trajectories differ substantially across model families despite comparable ranking performance.

## Figure B — Ranking vs Accessibility Scatter

Path:

- `paper/manuscript_figures/figure_ranking_vs_accessibility_scatter.png`

Data sources:

- `reports/neural_mlp/legacy_benchmark_summary.csv`
- `reports/neural_mlp/allocation_regime_summary.csv`

Axes used:

- X-axis: AUROC (dataset mean)
- Y-axis: Accessibility Persistence (`mean_fraction_below_0_01`)

Caption (manuscript-ready):

- AUROC versus accessibility persistence shows that models with similar ranking quality can exhibit materially different accessibility persistence and threshold controllability profiles, supporting partial non-equivalence between ranking and operational accessibility behavior.

Primary claim:

- Models with similar ranking quality can exhibit materially different accessibility persistence and threshold controllability.

## Table A — Allocator Family Summary

Path:

- `paper/manuscript_tables/table_allocator_family_summary.md`

Data sources:

- `reports/neural_mlp/legacy_benchmark_summary.csv`
- `reports/neural_mlp/legacy_threshold_sweep_summary.csv`
- `reports/neural_mlp/allocation_regime_summary.csv`

Purpose:

- Compact reviewer-facing family comparison anchor before neural perturbation analysis.

## Recommended Placement

### Figure A

Suggested section:

- Accessibility Trajectory Observations

Reason:

- Establishes recurring family-level accessibility morphology before neural perturbation results.

### Figure B

Suggested section:

- Accessibility Trajectory Observations

Reason:

- Provides direct visual support for ranking/accessibility non-equivalence.

### Table A

Suggested section:

- Accessibility Trajectory Observations

Reason:

- Gives a single cross-family numeric anchor table for rapid reviewer reference.

## Claim Mapping

| Artifact | Claim Supported |
| --- | --- |
| Figure A (`figure_family_reachability_comparison.png`) | Recurring accessibility structures exist across model families under shared protocol |
| Figure B (`figure_ranking_vs_accessibility_scatter.png`) | Ranking quality is partially non-equivalent to accessibility behavior |
| Table A (`table_allocator_family_summary.md`) | Family-level accessibility differences are numerically substantial and reviewer-verifiable |

## Narrative Effect

These additions support the intended evidence hierarchy:

1. Family-level accessibility phenomena first.
2. Reachability lens interpretation second.
3. Neural morphology transition as a follow-on explanation.
4. Calibration interaction as a later interaction layer.
