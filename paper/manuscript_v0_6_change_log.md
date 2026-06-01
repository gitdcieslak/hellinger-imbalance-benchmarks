# manuscript_v0_6 Change Log

Base file: `paper/manuscript_v0_5.md`
Output file: `paper/manuscript_v0_6.md`
Pass type: Experiment C stability integration

## Summary

This pass integrates Bagged HDDT Experiment C stability evidence into the manuscript while preserving the existing paper identity (empirical + conceptual framing, no new primary contribution).

## 1) Accessibility Trajectory Observations update

- Added subsection: **Stability of Accessibility Geometry** in Section 6.
- Integrated bootstrap and leave-one-dataset-out findings from:
  - `reports/neural_mlp/bagged_hddt_stability_summary.md`
  - `research/bagged_hddt_experiment_c.md`

Included conservative statements that:

- recovery uplift is stable under current resampling,
- max-jump uplift is stable under current resampling,
- lower smoothness is directionally stable,
- persistence uplift is smaller and qualified (especially vs HDDT).

No mechanism speculation was added in this subsection.

## 2) Bagged HDDT framing update

- Updated recurring-pattern discussion language so Bagged HDDT is framed as an **empirically stable phenomenon under current resampling** rather than an isolated anomaly.
- Kept recurring labels explicitly provisional and non-taxonomic.

## 3) Contribution statement strengthening (no new contribution)

- Strengthened Contribution 1 by adding a concrete ranking/accessibility non-equivalence anchor:
  - near-neighbor ranking case (Bagged HDDT vs LightGBM) with divergent accessibility geometry.

No additional contribution item was introduced.

## 4) Discussion stability sentence and caveat

- Added explicit sentence in Section 10:
  - effects are not driven by any single dataset in the current benchmark collection.
- Preserved bounded-generalization caveat:
  - additional datasets are needed for external validity.

## 5) Future work expansion

- Expanded Bagged HDDT mechanism future-work bullet to include:
  - vote dispersion,
  - bagging-effect decomposition,
  - ensemble accessibility geometry analysis.

## What did not change

- No new model training or benchmark reruns.
- No change to the manuscript's core thesis or scope delimitations.
- No mechanism-overclaim language was introduced.
