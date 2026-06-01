# Figure Caption Drafts (manuscript_v0_2)

## Figure 1 — Mean Reachability Transition Across MLP Variants

Source:

- `reports/geometry_transition_analysis/plots/reachability_transition_mean.png`
- Supporting table: `results/geometry_transition_analysis/geometry_transition_model_means.csv`

Primary claim:

- Fixed-architecture imbalance-pressure perturbation reshapes minority accessibility trajectories.

Secondary claims:

- `mlp_bce` is cliff-like relative to smoother `mlp_oversampled` / `mlp_weighted` forms.
- Reachability is a trajectory object, not a single-point recall restatement.

Recommended manuscript section:

- Section 4 (Reachability) and Section 8 (Neural Perturbation and Morphology Transition)

Caption draft:

- Mean minority reachability curves across severe-imbalance datasets show that objective/sampling perturbation under fixed architecture flattens threshold response relative to BCE, reducing abrupt accessibility loss at operational cutoffs.

Interpretation notes:

- Lead transition narrative with this global view, then immediately qualify with dataset heterogeneity using Figure 2.

Reviewer risks:

- Mean aggregation can hide difficult datasets (`oil`-like constrained behavior).

Appendix candidate:

- No

## Figure 2 — Dataset-Level Reachability Panels

Source:

- `reports/geometry_transition_analysis/plots/reachability_transition_by_dataset.png`
- Supporting derivatives: `results/geometry_transition_analysis/reachability_derivatives.csv`

Primary claim:

- Cliff-to-smooth transition is real but dataset-modulated.

Secondary claims:

- Strongest transition appears in `boundary`, `cam`, and `compustat`.
- `oil` remains constrained, reinforcing bounded generalization.

Recommended manuscript section:

- Section 4 and Section 8

Caption draft:

- Per-dataset reachability panels reveal consistent but heterogeneous trajectory transition under oversampling/weighting: severe datasets show clear persistence gains, while transitional datasets remain structurally constrained.

Interpretation notes:

- Use this figure to defend against “mean-only artifact” critique.

Reviewer risks:

- Panel density and legends can be visually crowded in print.

Appendix candidate:

- No

## Figure 3 — Threshold Collapse and Recovery Anchors

Source:

- `reports/neural_mlp_allocation_geometry_summary.md`
- Optional plot companion from `reports/neural_mlp/plots/` threshold sweep outputs

Primary claim:

- AUROC/AP can coexist with default-threshold accessibility collapse and low-threshold recovery.

Secondary claims:

- Ranking-quality summaries alone underdescribe operational accessibility.

Recommended manuscript section:

- Section 6 (Accessibility Trajectory Observations)

Caption draft:

- Representative threshold anchors show that acceptable ranking quality can coincide with near-collapsed minority accessibility at `t=0.50`, with substantial recovery only under aggressive threshold relaxation.

Interpretation notes:

- Keep numeric anchor text in caption (e.g., `recall@0.50` vs `recall@0.01`) to make the claim self-contained.

Reviewer risks:

- If rendered only as a table, some reviewers may ask for trajectory context; pair with Figure 1 or Figure 2.

Appendix candidate:

- No

## Figure 4 — Elasticity Localization Heatmap

Source:

- `reports/geometry_transition_analysis/plots/elasticity_interval_heatmap.png`
- Supporting summary: `results/geometry_transition_analysis/elasticity_concentration.csv`

Primary claim:

- Smooth variants reduce concentration of accessibility change into narrow threshold intervals.

Secondary claims:

- Morphology transition is not only vertical shift; it redistributes sensitivity.
- Accessibility dynamics are localized and interval-specific.

Recommended manuscript section:

- Section 7 (Morphology) and Section 8 (Transition)

Caption draft:

- Interval-wise reachability elasticity shows BCE concentration of recall change into sharp threshold zones, while oversampled/weighted variants distribute change more broadly with lower peak intensity.

Interpretation notes:

- This is the main visual for elasticity localization and threshold-evolution dynamics.

Reviewer risks:

- Interval definitions are threshold-grid dependent; note this in caption or nearby text.

Appendix candidate:

- No

## Figure 5 — Recurring Morphology Regime Summary Scatter

Source:

- `reports/neural_mlp/plots/regimes/allocation_regime_scatter.png`
- Supporting labels: `reports/neural_mlp/allocation_regime_summary.csv`

Primary claim:

- Recurring empirical morphology patterns appear under shared protocol.

Secondary claims:

- Regime vocabulary is useful for summarization but provisional.

Recommended manuscript section:

- Section 7 (Recurring Morphology Patterns)

Caption draft:

- Model-level support and elasticity projections separate recurring empirical allocation patterns (quantized, cliff-like, smooth, broad, conservative), used here as provisional organizational shorthand rather than fixed taxonomy.

Interpretation notes:

- Keep caveat sentence in manuscript body to avoid ontology overclaim.

Reviewer risks:

- 2D projection may hide uncertainty and boundary overlap.

Appendix candidate:

- No

## Figure 6 — MLP Perturbation Transition Composite

Source:

- Compose from `reports/geometry_transition_analysis/plots/reachability_transition_mean.png` + selected dataset slices from `reports/geometry_transition_analysis/plots/reachability_transition_by_dataset.png`

Primary claim:

- `mlp_bce -> mlp_oversampled/mlp_weighted` transition is the manuscript’s central controlled result.

Secondary claims:

- Operational morphology is not architecture-only.

Recommended manuscript section:

- Section 8

Caption draft:

- With architecture held fixed, oversampling/weighting moves MLP accessibility trajectories from cliff-like BCE behavior toward smoother, more persistent profiles across threshold evolution.

Interpretation notes:

- If page budget is tight, this can replace separate Figure 1+2 in main text and push one panel to appendix.

Reviewer risks:

- Composite design must avoid visual overpacking and inconsistent axis scales.

Appendix candidate:

- Yes

## Figure 7 — Support Breadth vs Occupancy Persistence

Source:

- `reports/geometry_transition_analysis/plots/support_vs_persistence.png`

Primary claim:

- Smoothness is not reducible to support breadth alone.

Secondary claims:

- Persistence can improve via threshold-morphology redistribution, not only broad support expansion.

Recommended manuscript section:

- Section 8

Caption draft:

- Effective support and occupancy persistence jointly indicate that smoother accessibility can emerge without maximal support broadening, consistent with morphology redistribution as a central transition mechanism.

Interpretation notes:

- Key evidence for Claim 12; refer directly to weighted vs oversampled contrast.

Reviewer risks:

- Scatter-only read may be seen as correlational; pair with Figure 4 elasticity localization.

Appendix candidate:

- No

## Figure 8 — Occupancy Persistence and Support Breadth Profiles

Source:

- `reports/neural_mlp/prediction_space_occupancy_summary.csv`
- `reports/neural_mlp/plots/occupancy/` (final file to be selected)

Primary claim:

- Occupancy structure adds operational information beyond ranking metrics.

Secondary claims:

- Accessibility persistence differs by dataset and model regime.

Recommended manuscript section:

- Section 6 (Accessibility Observations) or Section 8 (as mechanism context)

Caption draft:

- Occupancy views show how minority score support and persistence survive (or collapse) under threshold tightening, adding structure not visible in AUROC/AP summaries.

Interpretation notes:

- Best used with one severe-case panel and one contrast panel to avoid redundancy.

Reviewer risks:

- Source plot path still needs final lock-in from occupancy outputs.

Appendix candidate:

- Yes

## Figure 9 — Calibration Geometry Deltas

Source:

- `reports/geometry_transition_analysis/plots/calibration_geometry_deltas.png`
- Supporting means: `results/geometry_transition_analysis/calibration_transition_model_means.csv`

Primary claim:

- Calibration can improve reliability while degrading trajectory smoothness/jump behavior.

Secondary claims:

- Reliability and accessibility geometry are partially non-equivalent axes.

Recommended manuscript section:

- Section 9 (Calibration vs Operational Accessibility)

Caption draft:

- Average calibration deltas indicate that ECE/Brier improvements can coexist with negative smoothness shifts and positive jump shifts in this severe-imbalance setting, highlighting reliability-accessibility non-equivalence.

Interpretation notes:

- Keep explicit bounded phrase “in this setting” to prevent anti-calibration overread.

Reviewer risks:

- Could be misread as general anti-calibration claim if caveat is omitted.

Appendix candidate:

- No

## Figure 10 — Reliability vs Accessibility Regime Shift Examples (Raw vs Calibrated)

Source:

- `reports/neural_mlp_objective_perturbation/calibration_interaction/regime_persistence_table.csv`
- `reports/neural_mlp_objective_perturbation/calibration_interaction/calibration_summary_table.csv`

Primary claim:

- Regime interpretation can change after calibration transforms even when reliability improves.

Secondary claims:

- Calibration interaction is deployment-relevant, not purely statistical.

Recommended manuscript section:

- Section 9

Caption draft:

- Raw-versus-calibrated regime summaries show that reliability correction can alter accessibility morphology class assignments, reinforcing that calibration quality alone does not determine threshold-policy controllability.

Interpretation notes:

- If no ready plot exists, include as compact table or mini-panel figure.

Reviewer risks:

- Tabular presentation may be less intuitive than geometric plots; consider concise schematic overlay.

Appendix candidate:

- Yes

## Figure 11 — Recall Evolution Panels at Operational Threshold Grid

Source:

- `reports/neural_mlp/plots/` (threshold response files to verify)
- Fallback data source: `results/geometry_transition_analysis/reachability_curves.csv`

Primary claim:

- Recall evolution across `0.50, 0.25, 0.10, 0.05, 0.01` is intrinsically dynamic and model-conditioned.

Secondary claims:

- Threshold collapse examples are not isolated to one dataset when severe subsets are considered.

Recommended manuscript section:

- Section 6 and Section 7

Caption draft:

- Threshold-grid recall evolution panels show that accessibility change concentrates differently by model and dataset, with cliff allocators exhibiting abrupt late recovery and smoother allocators showing gradual persistence.

Interpretation notes:

- This directly satisfies “recall evolution” and “threshold collapse examples” figure coverage.

Reviewer risks:

- Exact path and final rendering still require artifact verification.

Appendix candidate:

- Yes
