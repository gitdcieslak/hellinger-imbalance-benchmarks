# Working Titles

- Reachability Under Severe Class Imbalance: An Empirical + Conceptual Framing of Operational Accessibility
- Beyond AUROC and Calibration: Threshold-Mediated Accessibility Trajectories in Imbalanced Classification
- Operational Accessibility Trajectories Under Severe Imbalance: Evidence, Patterns, and Calibration Tensions

# One-Paragraph Paper Thesis

Current evidence suggests that under severe class imbalance, ranking quality, calibration quality, and operational accessibility are partially non-equivalent. We observe that threshold-mediated reachability trajectories reveal operational failure modes that static metrics can obscure; we further observe recurring trajectory patterns and fixed-architecture MLP morphology transitions under objective/sampling perturbation. We treat the regime vocabulary as provisional empirical shorthand, not final theory, and position the paper as an empirical + conceptual framing document.

# Abstract Skeleton

- **Problem framing:** Severe imbalance deployments are threshold-mediated; static ranking metrics can miss accessibility dynamics.
- **Primary analytical object:** minority reachability trajectory `R(t)=P(score>=t|y=1)` and its threshold elasticity structure.
- **Empirical basis:** repeated-split legacy dataset analyses, trajectory summaries, occupancy/accessibility metrics, neural perturbation transitions, calibration interaction.
- **Core observations:**
  - ranking/accessibility non-equivalence,
  - cliff vs smooth trajectory differences,
  - fixed-architecture MLP `cliff -> smooth` shifts under imbalance-pressure perturbation,
  - calibration reliability gains that can coincide with accessibility re-steepening.
- **Contribution posture:** empirical + conceptual framing; provisional taxonomy; no complete theory claim.

# Central Narrative Spine

1. Ranking metrics can obscure threshold-mediated accessibility behavior.
2. Reachability trajectories make accessibility evolution visible.
3. Recurring morphology patterns appear empirically, but labels are provisional.
4. Fixed-architecture neural perturbations show optimization/imbalance-pressure sensitivity.
5. Calibration can improve ECE/Brier while worsening trajectory smoothness.
6. Operational accessibility appears partially independent from both ranking and calibration quality.

# Explicit Non-Claims

- We do **not** claim a complete operational morphology theory.
- We do **not** claim a universal classifier taxonomy.
- We do **not** claim all boosted models behave identically (audit-supported distinction: XGBoost vs LightGBM).
- We do **not** claim a resolved causal mechanism for Bagged HDDT broad behavior.
- We do **not** claim a production-ready reachability-aware objective.
- We do **not** position this manuscript as a leaderboard benchmark or SOTA claim.

# Structural Reconsideration: Reachability-Centered Option

## Option A — Reachability-Centered Structure

Proposed flow:

1. Introduction
2. Operational Accessibility Under Severe Imbalance
3. Reachability and Threshold Accessibility
4. Experimental Framework
5. Accessibility Trajectory Observations
6. Recurring Morphology Patterns (provisional)
7. Neural Perturbation and Morphology Transition
8. Calibration vs Operational Accessibility
9. Toward Operational Morphology (framing synthesis)
10. Limitations
11. Future Work

**Advantages:**
- Aligns with strongest audited claim (reachability trajectory centrality).
- Reduces reviewer risk that paper reads as model-comparison benchmark.
- Places calibration interaction as a core, not peripheral, empirical finding.

**Risks:**
- Could appear concept-heavy if trajectory definitions are not tied quickly to concrete results.
- Requires careful de-emphasis of regime labels to avoid ontology framing.

**Rationale:**
- Adopt Option A. It best matches claim-strength calibration from `research/claim_audit_report.md` and the revised claim inventory.

---

# 1. Introduction

## Section 1 — Introduction

**Purpose:** Establish ranking/accessibility mismatch and position the paper as empirical + conceptual framing.

**Primary claims:** 1, 2, 16.

**Evidence basis:**
- `reports/neural_mlp_allocation_geometry_summary.md`
- `research/study_journal.md`

**Conceptual role:** Open the paper with operational question, not model-family contest.

**Reviewer attack surface:**
- “Is this just benchmark rhetoric?”

**Scope limitations:**
- Explicitly state: not a complete theory; severe-imbalance legacy scope.

**Key figures/tables:**
- Primary table: compact ranking vs accessibility anchor (recall@0.50 vs recall@0.01).

**Critical language constraints:**
- Use “current evidence suggests,” “we observe,” “we do not claim universality.”

**Open risks / TODOs:**
- TODO: add 2-3 concrete numeric anchors early (including MLP collapse/recovery example).

# 2. Operational Accessibility Under Severe Imbalance

## Section 2 — Operational Accessibility Under Severe Imbalance

**Purpose:** Define deployment context and explain why threshold-mediated analysis is necessary.

**Primary claims:** 1, 2, 16.

**Evidence basis:**
- `research/operational_morphology_framework.md`
- `research/terminology.md`

**Conceptual role:** Create shared conceptual ground before methods.

**Reviewer attack surface:**
- “Is this just threshold tuning?”

**Scope limitations:**
- Clarify: threshold analysis is a deployment lens, not post-hoc metric fishing.

**Key figures/tables:**
- Optional conceptual schematic (appendix if needed).

**Critical language constraints:**
- Avoid abstract topology language.

**Open risks / TODOs:**
- TODO: include short practical deployment examples (triage/queue systems) without over-expanding domain claims.

# 3. Reachability and Threshold Accessibility

## Section 3 — Reachability and Threshold Accessibility

**Purpose:** Elevate reachability as primary organizing analytical object.

**Primary claims:** 3, 14.

**Evidence basis:**
- `results/geometry_transition_analysis/reachability_curves.csv`
- `results/geometry_transition_analysis/reachability_derivatives.csv`
- `research/operational_morphology_framework.md`

**Conceptual role:** Distinguish trajectory from single-point recall and anchor later sections.

**Reviewer attack surface:**
- “Isn’t reachability just recall?”

**Scope limitations:**
- Grid-dependent thresholds and discrete interval derivatives.

**Key figures/tables:**
- `reports/geometry_transition_analysis/plots/reachability_transition_mean.png` (primary)
- `reports/geometry_transition_analysis/plots/reachability_transition_by_dataset.png` (primary)

**Critical language constraints:**
- Explicitly state: recall is one point on `R(t)`, not trajectory structure.

**Open risks / TODOs:**
- TODO: add concise notation box with `R(t)` and interval elasticity.

# 4. Experimental Framework

## Section 4 — Experimental Framework

**Purpose:** Provide reproducible protocol and comparability constraints.

**Primary claims:** 8, 15, 16.

**Evidence basis:**
- experiment/config/report artifacts across baseline, perturbation, and transition analyses.

**Conceptual role:** Ensure claims are interpreted as protocol-bound empirical observations.

**Reviewer attack surface:**
- “Is this secretly architecture benchmarking?”

**Scope limitations:**
- Constrained datasets, constrained thresholds, constrained model families.

**Key figures/tables:**
- Methods table: datasets, thresholds, splits, model IDs.

**Critical language constraints:**
- Avoid “comprehensive benchmark” wording.

**Open risks / TODOs:**
- TODO: include explicit note that no new training framework or architecture zoo is used.

# 5. Accessibility Trajectory Observations

## Section 5 — Accessibility Trajectory Observations

**Purpose:** Empirically show ranking/accessibility divergence via trajectory outcomes.

**Primary claims:** 1, 2, 3.

**Evidence basis:**
- threshold summary outputs in `reports/neural_mlp/` and perturbation reports.

**Conceptual role:** Convert abstract non-equivalence into concrete threshold-mediated evidence.

**Reviewer attack surface:**
- “Is this just cherry-picked thresholds?”

**Scope limitations:**
- Threshold set fixed; not continuous optimization.

**Key figures/tables:**
- Primary: recall-vs-threshold panels
- Supporting: PR trajectory plots

**Critical language constraints:**
- Emphasize trajectory evolution, not model winner claims.

**Open risks / TODOs:**
- TODO: include severe-dataset and contrast-dataset pair (`boundary` + `satimage`).

# 6. Recurring Morphology Patterns (Provisional)

## Section 6 — Recurring Morphology Patterns (Provisional)

**Purpose:** Present recurring empirical patterns as shorthand, de-emphasizing taxonomy finality.

**Primary claims:** 4, 5, 6, 7, 15.

**Evidence basis:**
- allocation regime summaries, threshold elasticity summaries, occupancy summaries.

**Conceptual role:** Provide interpretable pattern language after trajectory evidence.

**Reviewer attack surface:**
- “Are regime labels arbitrary?”

**Scope limitations:**
- Labels are heuristic and protocol-dependent.

**Key figures/tables:**
- Secondary narrative figure: regime scatter plot
- Supporting table: regime metrics by model

**Critical language constraints:**
- Use “empirically recurring,” “provisional,” “shorthand.”

**Open risks / TODOs:**
- TODO: include explicit paragraph that regime labels are not ontological classes.

# 7. Neural Perturbation and Morphology Transition

## Section 7 — Neural Perturbation and Morphology Transition

**Purpose:** Show architecture-fixed morphology shifts under imbalance-pressure perturbation.

**Primary claims:** 8, 9, 10, 11, 12.

**Evidence basis:**
- `reports/neural_mlp_objective_perturbation_summary.md`
- `reports/geometry_transition_analysis_summary.md`
- `results/geometry_transition_analysis/geometry_transition_model_means.csv`

**Conceptual role:** Provide strongest argument that morphology is not architecture-only.

**Reviewer attack surface:**
- “Does oversampling merely shift thresholds?”

**Scope limitations:**
- Only BCE/oversampling/weighting variants in sklearn MLP path.

**Key figures/tables:**
- Primary: reachability-by-dataset panels
- Primary: elasticity interval heatmap
- Primary: support vs persistence scatter
- Table: compact MLP transition anchors (`smoothness`, `max_jump`, `recall@0.50`, `recall@0.01`)

**Critical language constraints:**
- Say “appears consistent with optimization/imbalance-pressure sensitivity.”

**Open risks / TODOs:**
- TODO: ensure table includes uncertainty notes and not just means.

# 8. Calibration vs Operational Accessibility (Consolidated)

## Section 8 — Calibration vs Operational Accessibility (Consolidated)

**Purpose:** Consolidate claims 13 and 14 into one integrated empirical section.

**Primary claims:** 13, 14.

**Evidence basis:**
- `results/geometry_transition_analysis/calibration_transition_model_means.csv`
- `reports/neural_mlp_objective_perturbation/calibration_interaction/regime_persistence_table.csv`
- calibration summary tables in perturbation outputs

**Conceptual role:** Elevate calibration interaction as one of the strongest paper results.

**Reviewer attack surface:**
- “Is this just calibration pathology?”

**Scope limitations:**
- Calibrator and split-size dependence; no universal anti-calibration claim.

**Key figures/tables:**
- Primary: `reports/geometry_transition_analysis/plots/calibration_geometry_deltas.png`
- Supporting: raw vs calibrated regime persistence table

**Critical language constraints:**
- Distinguish two layers explicitly:
  - reliability improvements (ECE/Brier)
  - operational trajectory degradation (smoothness/jump)

**Open risks / TODOs:**
- TODO: add concise “calibration tradeoff” textbox for reviewer clarity.

# 9. Toward Operational Morphology (Framing Synthesis)

## Section 9 — Toward Operational Morphology (Framing Synthesis)

**Purpose:** Synthesize empirical findings into a bounded conceptual framing.

**Primary claims:** 11, 12, 15, 16.

**Evidence basis:**
- cross-section synthesis of trajectory + perturbation + calibration findings.

**Conceptual role:** Articulate what is now supported and what remains provisional.

**Reviewer attack surface:**
- “Is the framework overclaiming?”

**Scope limitations:**
- Not a complete morphology theory; not universal classifier map.

**Key figures/tables:**
- Conceptual summary table mapping evidence -> interpretation -> uncertainty.

**Critical language constraints:**
- Use “current evidence suggests,” “we treat as provisional.”

**Open risks / TODOs:**
- TODO: include explicit bridge paragraph to limitations to prevent overread.

# 10. Limitations

## Section 10 — Limitations

**Purpose:** Make inferential and scope boundaries explicit in manuscript body.

**Primary claims:** 15, 16.

**Evidence basis:**
- Protocol constraints and audit notes.

**Conceptual role:** Reduce reviewer attack surface by preemptive delimitation.

**Reviewer attack surface:**
- “Does this generalize beyond these datasets?”

**Scope limitations:**
- legacy dataset family, fixed threshold grid, heuristic regime inference, limited neural family.

**Key figures/tables:**
- None primary; checklist table recommended.

**Critical language constraints:**
- Explicit non-claim language.

**Open risks / TODOs:**
- TODO: include paragraph on adjacent-worktree evidence path consistency from audit.

# 11. Future Work

## Section 11 — Future Work

**Purpose:** Provide constrained next steps consistent with evidence strength.

**Primary claims:** 17, 18.

**Evidence basis:**
- unresolved Bagged HDDT mechanism notes
- perturbation and calibration interaction results

**Conceptual role:** Preserve momentum without scope drift.

**Reviewer attack surface:**
- “Why not move directly to architecture zoo?”

**Scope limitations:**
- Future work remains hypothesis-generating.

**Key figures/tables:**
- Optional roadmap table only.

**Critical language constraints:**
- “motivates cautious follow-up,” not “next definitive step.”

**Open risks / TODOs:**
- TODO priority order:
1. Bagged HDDT mechanism disambiguation
2. Raw-vs-calibrated morphology decomposition
3. Controlled reachability-aware objective exploration

---

# Primary vs Secondary Narrative Figures

**Primary narrative figures (main text):**
- `reports/geometry_transition_analysis/plots/reachability_transition_mean.png`
- `reports/geometry_transition_analysis/plots/reachability_transition_by_dataset.png`
- `reports/geometry_transition_analysis/plots/elasticity_interval_heatmap.png`
- `reports/geometry_transition_analysis/plots/calibration_geometry_deltas.png`
- MLP perturbation comparison table (from `reports/neural_mlp_objective_perturbation_summary.md` anchors)

**Secondary/supporting figures (main text if space; otherwise appendix):**
- `reports/geometry_transition_analysis/plots/support_vs_persistence.png`
- regime scatter summaries
- occupancy/ECDF supporting panels

**Appendix candidates:**
- expanded regime tables
- full dataset-by-dataset threshold/trajectory panels
- auxiliary occupancy and calibration plots.

---

# Reviewer-Simulation Integration by Theme

- **Reachability vs recall objection:** handled in Section 3 via trajectory and derivative evidence.
- **Threshold tuning objection:** handled in Sections 5 and 7 via morphology concentration and transition redistribution.
- **Regime arbitrariness objection:** handled in Section 6 + Section 10 as provisional shorthand.
- **Calibration pathology objection:** handled in Section 8 by jointly presenting reliability gains and morphology costs.
- **Dataset specificity objection:** handled in Sections 5, 7, 10 with dataset-panel heterogeneity and explicit scope limits.
- **Benchmark-paper objection:** handled in Sections 1, 2, 9 with narrative spine and non-claims.
- **Bagged HDDT mechanism objection:** handled in Sections 6 and 11 as unresolved but empirically relevant reference.
