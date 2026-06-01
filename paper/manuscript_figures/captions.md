## Figure 0 — Operational Accessibility Lens Under Severe Imbalance

Source:

- Conceptual diagram generated for manuscript integration (not data-derived).

Primary claim:

- Operational accessibility should be evaluated as a threshold-trajectory object that links ranking/calibration summaries to deployment-facing persistence and smoothness.

Caption:

- Conceptual lens: ranking and calibration summaries are filtered through threshold evolution into reachability trajectories, which determine accessibility persistence and operational smoothness; cliff-like and smoother trajectory shapes imply different threshold-policy controllability under severe imbalance.

Manuscript placement:

- Section 3 (Operational Accessibility Under Severe Imbalance)

## Figure 1 — Mean Reachability Transition Across MLP Variants

Source:

- `reports/geometry_transition_analysis/plots/reachability_transition_mean.png`

Primary claim:

- Fixed-architecture imbalance-pressure perturbation reshapes minority accessibility trajectories.

Caption:

- Mean minority reachability curves across severe-imbalance datasets show that objective/sampling perturbation under fixed architecture flattens threshold response relative to BCE, reducing abrupt accessibility loss at operational cutoffs.

Manuscript placement:

- Section 4 and Section 8

## Figure 2 — Dataset-Level Reachability Panels

Source:

- `reports/geometry_transition_analysis/plots/reachability_transition_by_dataset.png`

Primary claim:

- Cliff-to-smooth transition is dataset-modulated rather than a mean-only artifact.

Caption:

- Per-dataset reachability panels reveal consistent but heterogeneous trajectory transition under oversampling/weighting: severe datasets show clear persistence gains, while transitional datasets remain structurally constrained.

Manuscript placement:

- Section 4 and Section 8

## Figure 3 — Elasticity Localization Heatmap

Source:

- `reports/geometry_transition_analysis/plots/elasticity_interval_heatmap.png`

Primary claim:

- Accessibility change concentrates differently across threshold intervals by variant.

Caption:

- Interval-wise reachability elasticity shows BCE concentration of recall change into sharp threshold zones, while oversampled/weighted variants distribute change more broadly with lower peak intensity.

Manuscript placement:

- Section 7 and Section 8

## Figure 4 — Support Breadth vs Occupancy Persistence

Source:

- `reports/geometry_transition_analysis/plots/support_vs_persistence.png`

Primary claim:

- Operational smoothness is not reducible to support breadth alone.

Caption:

- Effective support and occupancy persistence jointly indicate that smoother accessibility can emerge without maximal support broadening, consistent with threshold-morphology redistribution as a central transition mechanism.

Manuscript placement:

- Section 8

## Figure 5 — Calibration Geometry Deltas

Source:

- `reports/geometry_transition_analysis/plots/calibration_geometry_deltas.png`

Primary claim:

- Reliability improvement can coexist with degraded operational smoothness/jump behavior.

Caption:

- Average calibration deltas indicate that ECE/Brier improvements can coexist with negative smoothness shifts and positive jump shifts in this severe-imbalance setting, highlighting reliability-accessibility non-equivalence.

Manuscript placement:

- Section 9

## Table 1 — Ranking vs Accessibility Anchors

Source:

- `paper/manuscript_tables/table_1_ranking_vs_accessibility.md`

Primary claim:

- Ranking-quality summaries and threshold-mediated accessibility behavior are partially non-equivalent.

Caption:

- Dataset-mean anchors show that AUROC/AP quality can coexist with default-threshold accessibility collapse and large recovery under threshold relaxation, with distinct smoothness/jump profiles across model families and MLP perturbation variants.

Manuscript placement:

- Section 6
