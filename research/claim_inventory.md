# Claim Inventory (Operational Morphology Manuscript Scaffold)

## Claim 1 — Ranking Quality Is Not Operational Accessibility

**Claim:** Current evidence suggests ranking quality and operational accessibility are related but non-equivalent under severe imbalance.

**Status:** Core

**Confidence:** High

**Evidence:**
- `research/study_journal.md`
- `reports/neural_mlp_allocation_geometry_summary.md`
- `reports/neural_mlp_objective_perturbation_summary.md`

**Likely manuscript location:**
- Introduction; Empirical Observation section

**Caveats / threats:**
- Limited dataset family (legacy HDDT suite)
- Threshold set is fixed, not continuous

**Reviewer defense:**
- We explicitly do not claim universality; we claim repeated empirical non-equivalence in severe-imbalance settings.

## Claim 2 — AUROC/AP Can Obscure Default-Threshold Collapse

**Claim:** Models can maintain acceptable AUROC/AP while default-threshold recall collapses.

**Status:** Core

**Confidence:** High

**Evidence:**
- `reports/neural_mlp_allocation_geometry_summary.md` (MLP `recall@0.50=0.1346`, `recall@0.01=0.7867`)
- `research/study_journal.md` threshold-collapse notes

**Likely manuscript location:**
- Empirical Observation section

**Caveats / threats:**
- Depends on operational threshold conventions (0.50 may not be every deployment default)

**Reviewer defense:**
- We frame default threshold as a deployment-relevant reference point, not a normative optimum.

## Claim 3 — Reachability Adds Trajectory Information Beyond Single-Threshold Recall

**Claim:** Reachability curves capture accessibility evolution across thresholds, not only recall at one point.

**Status:** Core

**Confidence:** High

**Evidence:**
- `research/operational_morphology_framework.md`
- `reports/geometry_transition_analysis_summary.md`
- `results/geometry_transition_analysis/reachability_curves.csv`

**Likely manuscript location:**
- Operational Accessibility and Reachability section

**Caveats / threats:**
- Objection: “reachability is just recall”

**Reviewer defense:**
- We define recall as one point on `R(t)` and analyze trajectory shape, derivatives, and interval localization.

## Claim 4 — Classifier Families Show Distinct Allocation Regimes

**Claim:** Distinct recurring operational regimes (quantized, cliff, smooth, broad, conservative) appear across model families.

**Status:** Core

**Confidence:** Medium

**Evidence:**
- `reports/*allocation_regime_summary*.csv`
- `reports/neural_mlp_allocation_geometry_summary.md`

**Likely manuscript location:**
- Allocation Regimes section

**Caveats / threats:**
- Regime labels are heuristic and threshold-grid dependent

**Reviewer defense:**
- We treat taxonomy as provisional empirical organization, not a complete formal ontology.

## Claim 5 — CART-like Learners Can Exhibit Quantized Threshold Behavior

**Claim:** CART-like models can show near-threshold-invariant behavior consistent with quantized score support.

**Status:** Supporting

**Confidence:** Medium

**Evidence:**
- `research/study_journal.md`
- `reports/*threshold*summary*.md`

**Likely manuscript location:**
- Allocation Regimes section

**Caveats / threats:**
- Could vary with deeper trees or probability smoothing

**Reviewer defense:**
- We report observed behavior under fixed protocol; we do not claim all CART variants must behave this way.

## Claim 6 — XGBoost Exhibits Cliff-like Accessibility Collapse Under Severe Imbalance

**Claim:** XGBoost consistently exhibits cliff-like accessibility collapse under severe imbalance, with large abrupt recall jumps concentrated at aggressive threshold relaxation. This pattern does not generalize to all boosted learners — LightGBM exhibits a distinct conservative regime in the same benchmark conditions.

**Status:** Core

**Confidence:** High

**Evidence:**
- `reports/neural_mlp/allocation_regime_summary.csv` (xgboost=cliff_allocator)
- `reports/neural_mlp/threshold_elasticity_summary.csv` (boundary max_recall_jump=0.8129, consistent across all 5 datasets)

**Likely manuscript location:**
- Allocation Regimes section; Ranking vs Accessibility section

**Caveats / threats:**
- Hyperparameter sensitivity could change severity
- Claim is now XGBoost-specific; generalization to boosted learners as a class is explicitly not made

**Reviewer defense:**
- XGBoost and LightGBM are both gradient boosted but land in different regimes; we treat this as evidence that boosting family membership does not determine regime, and note it as an open mechanistic question.

## Claim 7 — HDDT and Bagged HDDT Exhibit Broad/Persistent Accessibility

**Claim:** Both hddt and bagged_hddt exhibit broad allocator behavior with lower concentration and more persistent minority accessibility than cliff or quantized alternatives. hddt_forest is a distinct model ID that should not be treated as synonymous — it exhibits cliff behavior in current-branch data and requires separate characterization.

**Status:** Supporting

**Confidence:** Medium

**Evidence:**
- `reports/neural_mlp/allocation_regime_summary.csv` (hddt=broad_allocator)
- `reports/neural_mlp/allocation_concentration_summary.csv` (low compression for hddt_forest occupancy, noting the regime label divergence)
- `reports/neural_mlp/prediction_space_occupancy_summary.csv`
- `../hellinger-imbalance-benchmarks/reports/allocation_regime_summary.csv` (bagged_hddt=broad_allocator)

**Likely manuscript location:**
- Allocation Regimes section; Discussion

**Caveats / threats:**
- `hddt_forest` is labeled `cliff_allocator` in current-branch data — the manuscript must not use "Bagged HDDT/HDDT-forest" slash notation as if these are synonyms
- Mechanism remains unresolved; broad behavior may reflect ensemble stabilization rather than Hellinger-specific effects
- Cross-branch naming drift should be resolved before submission

**Reviewer defense:**
- We restrict the broad allocator characterization to hddt and bagged_hddt specifically; hddt_forest behavior is noted as divergent and flagged for follow-up investigation.

## Claim 8 — Baseline MLP Is Analyzable in the Same Framework

**Claim:** MLP integrates cleanly into the existing operational geometry pipeline without neural-specific metric branches.

**Status:** Core

**Confidence:** High

**Evidence:**
- `reports/neural_mlp_allocation_geometry_summary.md`
- integration tests in `tests/test_models.py`, `tests/test_runner.py`

**Likely manuscript location:**
- Neural Extension section

**Caveats / threats:**
- Only sklearn-style MLP examined

**Reviewer defense:**
- The paper frames this as minimal neural extension, not comprehensive neural coverage.

## Claim 9 — BCE-trained MLP Exhibits Cliff-like Morphology

**Claim:** Under BCE/default training, MLP behaves as `cliff_allocator` with sharp threshold accessibility transitions.

**Status:** Core

**Confidence:** High

**Evidence:**
- `reports/neural_mlp_objective_perturbation_summary.md` (`mlp_bce` regime `cliff_allocator`, `max_recall_jump=0.4061`)

**Likely manuscript location:**
- Objective Perturbation section

**Caveats / threats:**
- Architecture depth/width fixed

**Reviewer defense:**
- We hold architecture constant by design to isolate imbalance-pressure effects.

## Claim 10 — Oversampling/Weighting Can Move MLP Cliff -> Smooth

**Claim:** Changing imbalance pressure (oversampling/weighting) can move MLP from cliff-like to smooth operational morphology.

**Status:** Core

**Confidence:** High

**Evidence:**
- `reports/neural_mlp_objective_perturbation_summary.md`
- `reports/geometry_transition_analysis_summary.md`
- `results/geometry_transition_analysis/geometry_transition_model_means.csv`

**Likely manuscript location:**
- Objective Perturbation and Morphology Transition section

**Caveats / threats:**
- Could be partly dataset-specific

**Reviewer defense:**
- We report repeated multi-dataset pattern and include dataset-level panels showing heterogeneity.

## Claim 11 — Operational Morphology Is Not Architecture-only

**Claim:** With architecture fixed, training pressure changes morphology, implying non-architectural control.

**Status:** Core

**Confidence:** High

**Evidence:**
- `mlp_bce` vs `mlp_oversampled` vs `mlp_weighted` comparisons in perturbation and transition reports

**Likely manuscript location:**
- Objective vs Architecture interpretation subsection

**Caveats / threats:**
- Still one architecture family

**Reviewer defense:**
- We state “not architecture-only,” not “architecture-irrelevant.”

## Claim 12 — Smoothness Is Not Reducible to Support Breadth Alone

**Claim:** Support broadening helps but does not fully explain smoothness; threshold-morphology redistribution is central.

**Status:** Core

**Confidence:** Medium

**Evidence:**
- `reports/neural_mlp_objective_perturbation_summary.md`
- `reports/geometry_transition_analysis_summary.md`

**Likely manuscript location:**
- Geometry Transition section

**Caveats / threats:**
- Some support/compression metrics unstable in edge cases

**Reviewer defense:**
- We use multi-metric convergence (elasticity, jumps, persistence) rather than a single support metric.

## Claim 13 — Calibration Can Improve ECE/Brier While Worsening Accessibility Morphology

**Claim:** Calibration improves reliability metrics yet can re-steepen threshold accessibility behavior.

**Status:** Core

**Confidence:** Medium

**Evidence:**
- `reports/geometry_transition_analysis_summary.md`
- `results/geometry_transition_analysis/calibration_transition_model_means.csv`

**Likely manuscript location:**
- Calibration vs Morphology section

**Caveats / threats:**
- Depends on calibrator family and split size

**Reviewer defense:**
- We report this as observed interaction, not a universal calibration theorem.

## Claim 14 — Calibration Quality and Operational Smoothness Are Partially Non-equivalent

**Claim:** Reliability improvement does not guarantee smoother operational trajectories.

**Status:** Core

**Confidence:** Medium

**Evidence:**
- Calibration regime persistence and smoothness/jump deltas

**Likely manuscript location:**
- Calibration vs Morphology section

**Caveats / threats:**
- Could change with alternative threshold policies

**Reviewer defense:**
- We frame non-equivalence as empirical under this deployment protocol.

## Claim 15 — Regime Taxonomy Is Provisional and Empirical

**Claim:** Regime labels are working empirical categories intended to guide analysis, not final theory classes.

**Status:** Core

**Confidence:** High

**Evidence:**
- `research/operational_morphology_framework.md`
- all regime summaries and transition notes

**Likely manuscript location:**
- Methods/Discussion framing note

**Caveats / threats:**
- Label boundary sensitivity

**Reviewer defense:**
- We include explicit qualifier language and ablation plans in future work.

## Claim 16 — This Is an Empirical + Conceptual Framing Paper, Not Complete Theory

**Claim:** TThe manuscript proposes an evidence-backed framing and vocabulary for operational accessibility geometry under severe imbalance. It is not a complete formal theory, a finalized algorithmic prescription, or a comprehensive model-zoo evaluation. Scope delimitation should be explicitly stated in the manuscript's abstract, introduction, and discussion — not only in internal research documents.

**Status:** Core

**Confidence:** Medium 

**Evidence:**
- `research/operational_morphology_framework.md` — Status section ("It is NOT: a finalized theory, a formal publication draft, or a complete mathematical treatment")
- Manuscript abstract and introduction — pending: explicit scope-delimiting language must be added before submission for this claim to be self-supporting

**Likely manuscript location:**
- Abstract, Introduction, Discussion

**Caveats / threats:**
- Until the manuscript contains its own scope statement, this claim depends on an internal document not visible to reviewers
- Reviewer pressure for stronger formalization remains a real risk; explicit delimitation is the primary defense

**Reviewer defense:**
- We explicitly frame the contribution as empirical and conceptual, position formal theory as future work, and delimit scope in the manuscript itself. The research program framework document provides the orienting rationale for this positioning.

## Claim 17 — Bagged HDDT Mechanism Remains Unresolved

**Claim:** Broad/persistent Bagged HDDT behavior is empirically visible, but causal mechanism remains open.

**Status:** Future Work

**Confidence:** Medium

**Evidence:**
- `research/study_journal.md` open question notes
- regime comparisons

**Likely manuscript location:**
- Discussion; Future Work

**Caveats / threats:**
- Confounding by ensemble size, feature subsampling, split randomness

**Reviewer defense:**
- We explicitly mark mechanism as unresolved and propose targeted follow-up analysis.

## Claim 18 — Reachability-aware Objectives Are a Cautious Future Direction

**Claim:** Current evidence motivates cautious investigation of reachability-aware objectives, without claiming immediate algorithmic superiority.

**Status:** Future Work

**Confidence:** Low

**Evidence:**
- Perturbation and transition evidence of morphology sensitivity

**Likely manuscript location:**
- Future Work

**Caveats / threats:**
- Risk of objective overfitting to specific threshold grid

**Reviewer defense:**
- We propose this as hypothesis-generating direction, not validated intervention.
