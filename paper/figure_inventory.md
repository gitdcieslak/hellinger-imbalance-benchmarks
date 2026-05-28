# Figure and Table Inventory (Operational Morphology Manuscript)

## Figure 1 — Threshold Collapse Under Severe Imbalance

**Candidate source artifact:**
- `reports/neural_mlp/legacy_threshold_sweep_summary.md`

**Primary claim supported:**
- AUROC/AP do not guarantee default-threshold accessibility.

**Secondary claims:**
- Ranking vs operational non-equivalence.

**Suggested manuscript location:**
- Section 5

**Caption draft:**
- *Threshold-sweep summaries show that models with acceptable ranking quality can exhibit near-collapsed minority accessibility at default threshold and recover only under aggressive relaxation.*

**Interpretation notes:**
- Pair with a numeric anchor table (recall@0.50 vs recall@0.01).

**Caveats:**
- Markdown summary may require a cleaner plotting view for manuscript quality.

## Figure 2 — Recall-vs-Threshold Response (Legacy Severe Set)

**Candidate source artifact:**
- `reports/neural_mlp/plots/` (threshold response plots; TODO: verify exact file names)

**Primary claim supported:**
- Accessibility collapse and recovery are trajectory phenomena.

**Secondary claims:**
- Threshold morphology varies by family.

**Suggested manuscript location:**
- Section 5

**Caption draft:**
- *Recall trajectories across operational thresholds reveal distinct transition morphologies that are not visible in single-threshold metrics.*

**Interpretation notes:**
- Prefer one severe dataset panel (`boundary` or `compustat`).

**Caveats:**
- TODO: verify exact artifact path and choose final panel.

## Figure 3 — Precision-Recall Trajectory Across Thresholds

**Candidate source artifact:**
- `reports/neural_mlp/plots/` trajectory outputs (TODO: verify exact path)

**Primary claim supported:**
- Operational tradeoff path differs across allocation regimes.

**Secondary claims:**
- Cliff allocators concentrate movement into narrow threshold segments.

**Suggested manuscript location:**
- Section 6

**Caption draft:**
- *Precision-recall trajectories parameterized by threshold illustrate family-specific operational tradeoff morphology under severe imbalance.*

**Interpretation notes:**
- Use consistent threshold ordering annotation.

**Caveats:**
- TODO: confirm final source file names.

## Figure 4 — Allocation Regime Summary Scatter

**Candidate source artifact:**
- `reports/neural_mlp/plots/regimes/allocation_regime_scatter.png`

**Primary claim supported:**
- Distinct empirical allocation regimes appear under shared protocol.

**Secondary claims:**
- Regime labels are useful but provisional.

**Suggested manuscript location:**
- Section 6

**Caption draft:**
- *Model-level support and elasticity summaries separate recurring empirical allocation regimes under severe imbalance.*

**Interpretation notes:**
- Pair with caution text about heuristic label boundaries.

**Caveats:**
- Single projection may hide multi-metric uncertainty.

## Figure 5 — Occupancy / Reachability Accessibility View

**Candidate source artifact:**
- `reports/neural_mlp/prediction_space_occupancy_summary.csv`
- `reports/neural_mlp/plots/occupancy/` (TODO: select one representative plot)

**Primary claim supported:**
- Accessibility persistence and occupancy structure add information beyond ranking.

**Secondary claims:**
- Compression behavior differs by dataset and model.

**Suggested manuscript location:**
- Section 6 or Section 3

**Caption draft:**
- *Occupancy and reachability artifacts reveal how minority accessibility survives (or collapses) under threshold tightening.*

**Interpretation notes:**
- Use one severe dataset plus `satimage` contrast.

**Caveats:**
- TODO: finalize exact occupancy plot variant.

## Figure 6 — Score Support / ECDF Morphology

**Candidate source artifact:**
- `results/neural_mlp_objective_perturbation/prediction_space_occupancy.jsonl` (ECDF payloads)
- existing occupancy ECDF plot outputs in `reports/neural_mlp_objective_perturbation/plots/occupancy/` (TODO verify names)

**Primary claim supported:**
- Posterior support continuity and accessibility shape differ by regime.

**Secondary claims:**
- Dead-zone and compression patterns are model- and dataset-conditioned.

**Suggested manuscript location:**
- Section 8

**Caption draft:**
- *Class-conditional ECDF/support views indicate that smooth transition can emerge from threshold-morphology redistribution, not only from broad support expansion.*

**Interpretation notes:**
- Pair with support-vs-persistence scatter.

**Caveats:**
- TODO: choose final ECDF figure path.

## Figure 7 — Calibration Interaction Overview

**Candidate source artifact:**
- `reports/neural_mlp_objective_perturbation/calibration_interaction/calibration_summary_table.csv`
- `reports/neural_mlp_objective_perturbation/calibration_interaction/regime_persistence_table.csv`

**Primary claim supported:**
- Calibration quality and operational smoothness are partially non-equivalent.

**Secondary claims:**
- Regime class can shift under calibration transforms.

**Suggested manuscript location:**
- Section 9

**Caption draft:**
- *Calibration improves ECE/Brier but can alter operational regime interpretation, indicating reliability and accessibility morphology are distinct axes.*

**Interpretation notes:**
- Show raw vs calibrated regime transitions for MLP variants.

**Caveats:**
- Requires careful wording to avoid anti-calibration overclaim.

## Table 1 — Baseline Neural Extension Summary

**Candidate source artifact:**
- `reports/neural_mlp_allocation_geometry_summary.md`

**Primary claim supported:**
- Baseline MLP integrates and is analyzable in the same framework.

**Secondary claims:**
- Baseline MLP cliff morphology and uneven recovery profile.

**Suggested manuscript location:**
- Section 7

**Caption draft:**
- *Baseline MLP under severe imbalance: ranking quality, default accessibility collapse, and threshold recovery in shared operational geometry analysis.*

**Interpretation notes:**
- Include anchor values from report (`auroc=0.7068`, `recall@0.50=0.1346`, `recall@0.01=0.7867`).

**Caveats:**
- Keep table compact; avoid leaderboard framing.

## Table 2 — MLP Objective Perturbation Comparison

**Candidate source artifact:**
- `reports/neural_mlp_objective_perturbation_summary.md`

**Primary claim supported:**
- Imbalance-pressure perturbation changes morphology under fixed architecture.

**Secondary claims:**
- Smoothness improves with oversampling/weighting.

**Suggested manuscript location:**
- Section 8

**Caption draft:**
- *Fixed-architecture MLP perturbation comparison shows cliff-to-smooth morphology transitions with reduced jump intensity and improved operational smoothness.*

**Interpretation notes:**
- Preserve numeric anchors already established.

**Caveats:**
- Clarify that this is not “best model” ranking.

## Figure 8 — Mean Reachability Transition

**Candidate source artifact:**
- `reports/geometry_transition_analysis/plots/reachability_transition_mean.png`

**Primary claim supported:**
- Objective/sampling perturbation reshapes accessibility trajectories under fixed architecture.

**Secondary claims:**
- Smooth variants reduce abrupt threshold collapse.

**Suggested manuscript location:**
- Section 8

**Caption draft:**
- *Mean minority reachability curves across datasets show flatter threshold response under oversampled/weighted MLP variants than BCE baseline.*

**Interpretation notes:**
- Directly from figure guide.

**Caveats:**
- Mean curves can hide dataset heterogeneity.

## Figure 9 — Dataset-level Reachability Panels

**Candidate source artifact:**
- `reports/geometry_transition_analysis/plots/reachability_transition_by_dataset.png`

**Primary claim supported:**
- Transition is real but dataset-modulated.

**Secondary claims:**
- Severe datasets show strongest transition signal; `oil` remains constrained.

**Suggested manuscript location:**
- Section 8

**Caption draft:**
- *Per-dataset reachability panels reveal consistent but heterogeneous cliff-to-smooth transition behavior across severe imbalance datasets.*

**Interpretation notes:**
- Mention `boundary/cam/compustat` vs `oil` contrast.

**Caveats:**
- Panel crowding; ensure readable legend.

## Figure 10 — Elasticity Interval Heatmap

**Candidate source artifact:**
- `reports/geometry_transition_analysis/plots/elasticity_interval_heatmap.png`

**Primary claim supported:**
- Smooth allocation corresponds to reduced elasticity concentration.

**Secondary claims:**
- Morphology transition reflects redistribution of sensitivity.

**Suggested manuscript location:**
- Section 8

**Caption draft:**
- *Interval-wise reachability elasticity shows BCE concentrates transition mass into sharp zones, while oversampled/weighted variants spread change more gradually.*

**Interpretation notes:**
- Pair with `results/geometry_transition_analysis/elasticity_concentration.csv`.

**Caveats:**
- Grid-dependent interval definitions.

## Figure 11 — Support vs Persistence Scatter

**Candidate source artifact:**
- `reports/geometry_transition_analysis/plots/support_vs_persistence.png`

**Primary claim supported:**
- Smoothness is not reducible to support breadth alone.

**Secondary claims:**
- Persistence can improve through threshold-morphology change even with modest support expansion.

**Suggested manuscript location:**
- Section 8

**Caption draft:**
- *Support breadth and occupancy persistence jointly indicate that smoother accessibility may emerge without maximal support broadening.*

**Interpretation notes:**
- Use as central evidence for Claim 12.

**Caveats:**
- Compression ratio instability should be noted.

## Figure 12 — Calibration Geometry Deltas

**Candidate source artifact:**
- `reports/geometry_transition_analysis/plots/calibration_geometry_deltas.png`

**Primary claim supported:**
- Calibration can improve reliability while worsening smoothness/jump morphology.

**Secondary claims:**
- Calibration-morphology interaction is non-trivial and deployment-relevant.

**Suggested manuscript location:**
- Section 9

**Caption draft:**
- *Average calibration deltas show reliability gains can coincide with smoothness decline and jump intensification, highlighting reliability-accessibility non-equivalence.*

**Interpretation notes:**
- Pair with regime persistence table for raw vs calibrated labels.

**Caveats:**
- Avoid interpreting as “calibration is harmful”; emphasize tradeoff axis.

---

## Missing or To-verify Artifacts

- TODO: verify exact threshold response and PR-trajectory plot paths under `reports/neural_mlp/plots/`.
- TODO: verify exact occupancy ECDF plot filenames for direct citation.
- TODO: if any figure is not publication quality, regenerate only the plotting layer (no experiment reruns).
