# Figure Realization Plan (manuscript_v0_4)

## Scope

This pass realizes figure/table assets for `paper/manuscript_v0_3.md` using existing `reports/` and `results/` artifacts only.

No new experiments, no new claims, and no scope expansion are introduced.

## Asset Realization Summary

- Figure 0 (conceptual) generated as manuscript-native diagram:
  - `paper/manuscript_figures/figure_0_operational_accessibility_lens.png`
  - `paper/manuscript_figures/figure_0_operational_accessibility_lens.pdf`
- Figures 1-5 copied from finalized transition-analysis plots:
  - `paper/manuscript_figures/figure_1_reachability_mean_transition.png`
  - `paper/manuscript_figures/figure_2_reachability_by_dataset.png`
  - `paper/manuscript_figures/figure_3_elasticity_interval_heatmap.png`
  - `paper/manuscript_figures/figure_4_support_vs_persistence.png`
  - `paper/manuscript_figures/figure_5_calibration_geometry_deltas.png`
- PDF versions produced for Figures 1-5:
  - `paper/manuscript_figures/figure_1_reachability_mean_transition.pdf`
  - `paper/manuscript_figures/figure_2_reachability_by_dataset.pdf`
  - `paper/manuscript_figures/figure_3_elasticity_interval_heatmap.pdf`
  - `paper/manuscript_figures/figure_4_support_vs_persistence.pdf`
  - `paper/manuscript_figures/figure_5_calibration_geometry_deltas.pdf`
- Table 1 generated with verified values:
  - `paper/manuscript_tables/table_1_ranking_vs_accessibility.md`
- Manuscript-ready caption pack generated:
  - `paper/manuscript_figures/captions.md`

## Source Mapping

- Figure 1: `reports/geometry_transition_analysis/plots/reachability_transition_mean.png`
- Figure 2: `reports/geometry_transition_analysis/plots/reachability_transition_by_dataset.png`
- Figure 3: `reports/geometry_transition_analysis/plots/elasticity_interval_heatmap.png`
- Figure 4: `reports/geometry_transition_analysis/plots/support_vs_persistence.png`
- Figure 5: `reports/geometry_transition_analysis/plots/calibration_geometry_deltas.png`
- Table 1 values:
  - `reports/neural_mlp_objective_perturbation/legacy_benchmark_summary.csv`
  - `reports/neural_mlp_objective_perturbation/legacy_threshold_sweep_summary.csv`
  - `results/geometry_transition_analysis/geometry_transition_model_means.csv`
  - `reports/neural_mlp/legacy_benchmark_summary.csv`
  - `reports/neural_mlp/legacy_threshold_sweep_summary.csv`
  - `reports/neural_mlp/allocation_regime_summary.csv`

## Standardization Choices

- Retained original plot styling from transition-analysis outputs for consistency with previously reviewed evidence.
- Kept model IDs in data-derived visuals unchanged to avoid relabeling ambiguity.
- Used consistent figure numbering and filename schema (`figure_<n>_<slug>.png`).
- Added PDF companions for all finalized figures.

## Verification Checklist

- Required figure files exist under `paper/manuscript_figures/`: Yes
- Required table exists under `paper/manuscript_tables/`: Yes
- Captions prepared in manuscript-ready format: Yes
- No unverified numeric values inserted: Yes
- Claims remain unchanged from v0.3 framing: Yes

## Follow-up (optional, not required for v0.4)

- Harmonize typography/colors across all data-derived figures if camera-ready formatting demands a single visual theme.
- Optionally regenerate selected plots from `results/geometry_transition_analysis/*.csv` for journal-specific style guides while preserving values.
