# Working Titles

- Operational Accessibility Under Severe Class Imbalance: An Empirical and Conceptual Morphology Framing
- Ranking Is Not Accessibility: Operational Morphology of Imbalanced Classifiers
- Threshold-Mediated Accessibility Morphology Under Severe Class Imbalance

# One-Paragraph Paper Thesis

Current evidence suggests that under severe class imbalance, ranking quality, calibration quality, and operational accessibility are partially non-equivalent. We observe repeatable threshold-collapse and reachability-morphology differences across classifier families and within a fixed MLP architecture under objective/sampling perturbations. We frame these observations as an empirical + conceptual operational morphology program, provide a provisional regime vocabulary, and show calibration can improve reliability while re-steepening operational accessibility.

# Abstract Skeleton

- **Context:** Severe imbalance systems are often deployed with threshold-mediated action policies.
- **Problem:** AUROC/AP and calibration alone can underdescribe deployment accessibility.
- **Approach:** Repeated-split legacy experiments with threshold sweeps, occupancy, elasticity, regime synthesis, and calibration interaction analyses.
- **Key observations:**
  - ranking vs accessibility non-equivalence,
  - distinct allocation regimes,
  - MLP `cliff -> smooth` transition under oversampling/weighting,
  - calibration-reliability gains with occasional morphology re-steepening.
- **Contribution type:** empirical + conceptual framing, not complete theory.
- **Implication:** motivates cautious development of reachability-aware operational analysis and objectives.

# 1. Introduction

## Section 1 — Introduction

**Purpose:** State core mismatch between ranking success and deployable accessibility under severe imbalance.

**Primary claims:** Claims 1, 2, 16.

**Evidence / figures:** Threshold-collapse examples; summary table anchors from `reports/neural_mlp_allocation_geometry_summary.md`.

**Key language to use:** “current evidence suggests,” “we observe,” “we do not claim universality.”

**Caveats:** Dataset family limited; threshold policy fixed.

**TODOs:** Add concise motivation example from one severe dataset panel.

# 2. Background and Motivation

## Section 2 — Background and Motivation

**Purpose:** Position work against leaderboard framing and define threshold-mediated deployment lens.

**Primary claims:** Claims 1, 2, 15, 16.

**Evidence / figures:** Study-journal trajectory; framework motivation excerpts.

**Key language to use:** “operational accessibility allocation systems.”

**Caveats:** Avoid overformalization of terminology.

**TODOs:** Add concise related-work matrix: ranking/calibration vs operational trajectory focus.

# 3. Operational Accessibility and Reachability

## Section 3 — Operational Accessibility and Reachability

**Purpose:** Define reachability and explain why trajectory metrics are required.

**Primary claims:** Claims 3, 14.

**Evidence / figures:** `reports/geometry_transition_analysis/plots/reachability_transition_mean.png`; derivative table.

**Key language to use:** “recall is a point on a curve, not the curve.”

**Caveats:** Reachability depends on threshold grid.

**TODOs:** Add formal notation box for `R(t)` and interval elasticity.

# 4. Experimental Framework

## Section 4 — Experimental Framework

**Purpose:** Document protocol and reproducibility boundaries.

**Primary claims:** Claims 8, 15, 16.

**Evidence / figures:** Config and report artifacts for legacy runs, perturbation, transition analysis.

**Key language to use:** “constrained model set,” “fixed architecture perturbation.”

**Caveats:** Legacy datasets only; no architecture zoo.

**TODOs:** Add compact methods table (datasets, models, thresholds, split protocol).

# 5. Empirical Observation: Ranking Does Not Ensure Accessibility

## Section 5 — Ranking Does Not Ensure Accessibility

**Purpose:** Demonstrate ranking-accessibility divergence with numeric anchors.

**Primary claims:** Claims 1, 2.

**Evidence / figures:** `reports/neural_mlp_allocation_geometry_summary.md`; threshold plots from `reports/neural_mlp/`.

**Key language to use:** “can obscure,” “operationally fragile at default threshold.”

**Caveats:** Threshold default context may vary by domain.

**TODOs:** Include one table with AUROC/AP + recall@0.50 + recall@0.01 across selected models.

# 6. Allocation Regimes Under Severe Imbalance

## Section 6 — Allocation Regimes Under Severe Imbalance

**Purpose:** Introduce provisional regime taxonomy and family-level patterns.

**Primary claims:** Claims 4, 5, 6, 7, 15.

**Evidence / figures:** allocation regime summaries and scatter plots from `reports/neural_mlp/plots/regimes/` and perturbation comparators.

**Key language to use:** “provisional empirical taxonomy.”

**Caveats:** Label boundaries are heuristic.

**TODOs:** Add reviewer note addressing “Are regime labels arbitrary?”

# 7. Neural Extension: MLPs as Operational Allocators

## Section 7 — Neural Extension: MLPs as Operational Allocators

**Purpose:** Show MLP integration without changing metric pipeline.

**Primary claims:** Claims 8, 9.

**Evidence / figures:** `reports/neural_mlp_allocation_geometry_summary.md`.

**Key language to use:** “MLP is analyzable in the same operational geometry framework.”

**Caveats:** Single architecture baseline.

**TODOs:** Add short appendix note on implementation constraints (sklearn-only).

# 8. Objective Perturbation and Morphology Transitions

## Section 8 — Objective Perturbation and Morphology Transitions

**Purpose:** Isolate imbalance-pressure effects under fixed architecture.

**Primary claims:** Claims 10, 11, 12.

**Evidence / figures:**
- `reports/neural_mlp_objective_perturbation_summary.md`
- `reports/geometry_transition_analysis_summary.md`
- figure guide plots (reachability, elasticity heatmap, support vs persistence)

**Key language to use:** “fixed architecture, varying imbalance pressure.”

**Caveats:** Limited perturbation family (BCE/oversampled/weighted only).

**TODOs:** Include compact transition table with anchored numbers:
- `mlp_bce` `smoothness=0.4178`, `max_jump=0.4061`
- `mlp_oversampled` `smoothness=0.5391`, `max_jump=0.0678`
- `mlp_weighted` `smoothness=0.6080`, `max_jump=0.1783`

# 9. Calibration vs Operational Morphology

## Section 9 — Calibration vs Operational Morphology

**Purpose:** Clarify reliability vs accessibility non-equivalence.

**Primary claims:** Claims 13, 14.

**Evidence / figures:**
- `results/geometry_transition_analysis/calibration_transition_model_means.csv`
- `reports/geometry_transition_analysis/plots/calibration_geometry_deltas.png`
- regime persistence table in perturbation calibration outputs

**Key language to use:** “calibration-reliability gains can coexist with accessibility re-steepening.”

**Caveats:** Calibrator-dependent behavior.

**TODOs:** Add focused objection-response box: “Is this just calibration failure?”

# 10. Discussion: Toward Operational Morphology

## Section 10 — Discussion

**Purpose:** Consolidate empirical narrative and conceptual framing contributions.

**Primary claims:** Claims 11, 12, 15, 16.

**Evidence / figures:** Synthesis table and figure guide narrative order.

**Key language to use:** “jointly emergent,” “we treat as provisional.”

**Caveats:** Avoid complete-theory framing.

**TODOs:** Add explicit paragraph on why this is not a benchmark paper.

# 11. Limitations

## Section 11 — Limitations

**Purpose:** Make boundaries explicit.

**Primary claims:** Claims 15, 16.

**Evidence / figures:** Scope documentation from configs and reports.

**Key language to use:** “we do not claim,” “current evidence is constrained by.”

**Caveats:** Dataset scope, threshold grid, heuristic taxonomy, limited neural families.

**TODOs:** Add checklist-style limitations for reviewer clarity.

# 12. Future Work

## Section 12 — Future Work

**Purpose:** Present bounded next steps without scope drift.

**Primary claims:** Claims 17, 18.

**Evidence / figures:** Study journal unresolved Bagged HDDT mechanism; perturbation transition signals.

**Key language to use:** “motivates,” “hypothesis-generating,” “cautious expansion.”

**Caveats:** Risk of architecture-zoo and objective overfitting.

**TODOs:** Prioritize:
1. Bagged HDDT mechanism analysis
2. calibrated-vs-raw morphology decomposition
3. cautiously scoped reachability-aware objective study

---

# Reviewer Objection Integration Plan

- **“Isn’t reachability just recall?”** Address in Section 3 with trajectory/derivative distinction.
- **“Isn’t this just threshold tuning?”** Address in Sections 5 and 8 via morphology concentration and regime movement evidence.
- **“Is this just calibration failure?”** Address in Section 9 with ECE/Brier gains plus smoothness decline evidence.
- **“Are regime labels arbitrary?”** Address in Section 6 and Limitations as provisional empirical taxonomy.
- **“Is this dataset-specific?”** Address with dataset panels and explicit scope boundary language.
- **“Does this generalize beyond tree models?”** Address via MLP extension and fixed-architecture perturbation evidence.
- **“Does oversampling merely shift thresholds?”** Address via interval elasticity redistribution and jump reduction.
- **“Is the framework overclaiming?”** Address via Section 10/11 with explicit non-theory positioning.
- **“Why not just a benchmark paper?”** Address in Intro/Discussion: emphasis is operational morphology, not SOTA ranking.
- **“Where does Bagged HDDT fit mechanistically?”** Address as unresolved core reference in Future Work.
