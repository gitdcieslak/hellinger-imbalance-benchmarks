# Claim Evidence Audit Report

Generated: 2026-05-28
Branch: research/neural-learners
Adjacent worktree checked: ../hellinger-imbalance-benchmarks/ (found — contains reports/ with top-level artifact files including allocation_regime_summary.csv, threshold_elasticity_summary.md, and legacy_hddt_threshold_sweep_summary.md; no results/ directory)

Note on glob resolution: the Glob tool does not search across worktrees. Claims 4 and 5 cite glob patterns that resolve correctly when evaluated against the adjacent worktree root; they return zero results when applied only to the current branch.

---

## Summary Table

| Claim | Title | Status | Stated Confidence | Audit Rating | Notes |
|-------|-------|--------|-------------------|--------------|-------|
| 1  | Ranking Quality Is Not Operational Accessibility | Core | High | STRONG | Numerical examples directly verified |
| 2  | AUROC/AP Can Obscure Default-Threshold Collapse | Core | High | STRONG | Exact quoted stats confirmed in source file |
| 3  | Reachability Adds Trajectory Information Beyond Single-Threshold Recall | Core | High | STRONG | Multi-threshold CSV + framework definition present |
| 4  | Classifier Families Show Distinct Allocation Regimes | Core | Medium | STRONG | `reports/allocation_regime_summary.csv` in adjacent worktree matches glob; 5 distinct regimes confirmed |
| 5  | CART-like Learners Can Exhibit Quantized Threshold Behavior | Supporting | Medium | STRONG | `reports/*threshold*summary*.md` matches 3 files in adjacent worktree; CART smoothness=1.0, jump=0.0 confirmed |
| 6  | XGBoost Exhibits Cliff-like Accessibility Collapse Under Severe Imbalance | Core | High | STRONG | Claim now XGBoost-specific; exact files cited; xgboost=cliff_allocator confirmed with per-dataset elasticity data |
| 7  | HDDT and Bagged HDDT Exhibit Broad/Persistent Accessibility | Supporting | Medium | STRONG | hddt=broad_allocator in current branch; bagged_hddt=broad_allocator in adjacent worktree; hddt_forest divergence explicitly scoped out in claim |
| 8  | Baseline MLP Is Analyzable in the Same Framework | Core | High | STRONG | Integration summary + test files confirm no neural-specific branches |
| 9  | BCE-trained MLP Exhibits Cliff-like Morphology | Core | High | STRONG | Exact cited values (cliff_allocator, max_recall_jump=0.4061) verified |
| 10 | Oversampling/Weighting Can Move MLP Cliff → Smooth | Core | High | STRONG | Regime shift confirmed quantitatively in two files + CSV |
| 11 | Operational Morphology Is Not Architecture-only | Core | High | STRONG | Fixed-architecture, variable-objective experiment design is airtight |
| 12 | Smoothness Is Not Reducible to Support Breadth Alone | Core | Medium | STRONG | mlp_oversampled dissociation case confirmed numerically |
| 13 | Calibration Can Improve ECE/Brier While Worsening Accessibility Morphology | Core | Medium | STRONG | calibration_transition_model_means.csv has explicit before/after deltas |
| 14 | Calibration Quality and Operational Smoothness Are Partially Non-equivalent | Core | Medium | PARTIAL | Supporting data exists but no specific file is cited in the inventory |
| 15 | Regime Taxonomy Is Provisional and Empirical | Core | High | STRONG | Framework document explicitly states provisional/empirical status |
| 16 | This Is an Empirical + Conceptual Framing Paper, Not Complete Theory | Core | High | STRONG | Framework status section explicitly delimits scope |
| 17 | Bagged HDDT Mechanism Remains Unresolved | Future Work | Medium | (see §Future Work) | Consistent with existing evidence; appropriately hedged |
| 18 | Reachability-aware Objectives Are a Cautious Future Direction | Future Work | Low | (see §Future Work) | Consistent with perturbation evidence; appropriately hedged |

---

## Missing Evidence Files

The following references from the claim inventory could not be resolved to files in the **current branch**. All of them resolve in the **adjacent worktree** (`../hellinger-imbalance-benchmarks/`) or are vague:

| Claim | Cited Pattern | Resolution |
|-------|---------------|------------|
| 4 | `reports/*allocation_regime_summary*.csv` | **Resolved in adjacent worktree** — `../hellinger-imbalance-benchmarks/reports/allocation_regime_summary.csv` matches; includes `bagged_hddt`, `cart`, `hddt`, `lightgbm`, `random_forest`, `xgboost` |
| 5 | `reports/*threshold*summary*.md` | **Resolved in adjacent worktree** — matches `reports/legacy_hddt_threshold_sweep_summary.md`, `reports/threshold_elasticity_summary.md`, and `reports/synthetic_threshold_sweep_summary.md` |
| 6, 7 | `reports/neural_mlp/` (directory reference) | **Resolved** — claims now cite specific files: `reports/neural_mlp/allocation_regime_summary.csv`, `reports/neural_mlp/threshold_elasticity_summary.csv`, `reports/neural_mlp/allocation_concentration_summary.csv`, `reports/neural_mlp/prediction_space_occupancy_summary.csv` |
| 14 | "Calibration regime persistence and smoothness/jump deltas" | **No file path cited** — evidence lives in `results/geometry_transition_analysis/calibration_transition_model_means.csv` and `reports/neural_mlp_objective_perturbation/calibration_interaction/regime_persistence_table.csv` |

All other cited files were located in the current worktree. The adjacent worktree (`../hellinger-imbalance-benchmarks/`) contains a populated `reports/` directory at its root with the original pre-neural baseline analysis. It does not contain a `results/` directory.

---

## Detailed Findings

---

### Claim 1 — Ranking Quality Is Not Operational Accessibility

**Audit rating:** STRONG

**Files found:**
- `research/study_journal.md` ✓
- `reports/neural_mlp_allocation_geometry_summary.md` ✓
- `reports/neural_mlp_objective_perturbation_summary.md` ✓

**Files missing:** None

**Key evidence:**
From `neural_mlp_allocation_geometry_summary.md` §2: "MLP: `auroc_mean=0.7068`, `average_precision_mean=0.2410`, default-threshold `recall_mean=0.1346`... Recall recovery under threshold relaxation (0.50 → 0.01, dataset-mean): MLP recall recovery: +0.6522 (0.1346 → 0.7867)."
From `study_journal.md` (2026-05-16): "XGBoost and RandomForest achieve good AUROC/AP, but default-threshold recall collapses under severe imbalance. Example: AUROC > 0.90, recall near 0.0 at threshold 0.50."

**Confidence calibration:** Stated High — agrees. Numerical examples are concrete and cross-validated across multiple datasets and model families.

**Notes:** The geometry summary also notes MLP boundary=0.0000 and compustat=0.0000 recall@0.50 alongside reasonable AUROC, providing multiple independent instances of the phenomenon.

---

### Claim 2 — AUROC/AP Can Obscure Default-Threshold Collapse

**Audit rating:** STRONG

**Files found:**
- `reports/neural_mlp_allocation_geometry_summary.md` ✓ (contains the exact quoted statistics)
- `research/study_journal.md` ✓

**Files missing:** None

**Key evidence:**
The exact values cited in the claim — `recall@0.50=0.1346`, `recall@0.01=0.7867` — are confirmed in `neural_mlp_allocation_geometry_summary.md` §2. The journal adds a sharper XGBoost case (AUROC > 0.90, recall ≈ 0.0 at threshold 0.50).

**Confidence calibration:** Stated High — agrees. The gap between AUROC and deployment-threshold recall is documented with exact numbers, not inference.

**Notes:** Dataset-level breakdown (boundary=0.0000, cam=0.0310, compustat=0.0000 at threshold 0.50) makes this a multi-instance claim, not a single anecdote.

---

### Claim 3 — Reachability Adds Trajectory Information Beyond Single-Threshold Recall

**Audit rating:** STRONG

**Files found:**
- `research/operational_morphology_framework.md` ✓
- `reports/geometry_transition_analysis_summary.md` ✓
- `results/geometry_transition_analysis/reachability_curves.csv` ✓

**Files missing:** None

**Key evidence:**
`reachability_curves.csv` provides per-threshold recall for all three MLP variants × 5 datasets × 5 threshold levels, making explicit the trajectory shape that would be invisible from any single-threshold statistic. The framework formally defines R(t) = P(ŷ(x) ≥ t | y = 1). The transition summary describes how mlp_bce concentrates recall gain in a single interval while mlp_oversampled redistributes it, a distinction invisible from single-threshold recall.

**Confidence calibration:** Stated High — agrees. The trajectory/derivative decomposition in the CSV is direct proof of concept.

---

### Claim 4 — Classifier Families Show Distinct Allocation Regimes

**Audit rating:** STRONG

**Files found:**
- `../hellinger-imbalance-benchmarks/reports/allocation_regime_summary.csv` ✓ (adjacent worktree; matches cited glob `reports/*allocation_regime_summary*.csv`)
- `reports/neural_mlp/allocation_regime_summary.csv` ✓ (current branch; not matched by cited glob but corroborates)
- `reports/neural_mlp_allocation_geometry_summary.md` ✓

**Files missing:** None — glob resolves in the adjacent worktree.

**Key evidence:**
From `../hellinger-imbalance-benchmarks/reports/allocation_regime_summary.csv` (the canonical baseline analysis):

| model_id | inferred_regime | smoothness | max_recall_jump |
|---|---|---:|---:|
| bagged_hddt | broad_allocator | 0.2486 | 0.3288 |
| cart | quantized_allocator | 1.0000 | 0.0000 |
| hddt | broad_allocator | 0.2353 | 0.2690 |
| lightgbm | conservative_allocator | 0.2624 | 0.2218 |
| random_forest | broad_allocator | 0.2249 | 0.3458 |
| xgboost | cliff_allocator | 0.1753 | 0.4887 |

Five distinct regime labels (quantized, cliff, broad, conservative, and smooth — added by the neural perturbation analysis) are evidenced across the corpus.

**Confidence calibration:** Stated Medium — agrees. The regime taxonomy is heuristic and the label boundaries are threshold-grid dependent as noted. The data is clear but the provisional framing is appropriate.

**Notes:** The current branch's allocation_regime_summary.csv (at nested paths) also confirms the same baseline regime labels, adding mlp_bce/oversampled/weighted from the neural perturbation experiments.

---

### Claim 5 — CART-like Learners Can Exhibit Quantized Threshold Behavior

**Audit rating:** STRONG

**Files found:**
- `research/study_journal.md` ✓
- `../hellinger-imbalance-benchmarks/reports/threshold_elasticity_summary.md` ✓ (adjacent worktree; matches cited glob `reports/*threshold*summary*.md`)
- `../hellinger-imbalance-benchmarks/reports/legacy_hddt_threshold_sweep_summary.md` ✓ (adjacent worktree; also matches glob)
- `../hellinger-imbalance-benchmarks/reports/synthetic_threshold_sweep_summary.md` ✓ (adjacent worktree; also matches glob)

**Files missing:** None — glob resolves to three files in the adjacent worktree.

**Key evidence:**
From `../hellinger-imbalance-benchmarks/reports/threshold_elasticity_summary.md` across all 5 datasets:

| dataset | model_id | max_recall_jump | elasticity | smoothness |
|---|---|---:|---:|---:|
| boundary | cart | 0.0000 | 0.0000 | 1.0000 |
| cam | cart | 0.0000 | 0.0000 | 1.0000 |
| compustat | cart | 0.0000 | 0.0000 | 1.0000 |
| oil | cart | 0.0000 | 0.0000 | 1.0000 |
| satimage | cart | 0.0000 | 0.0000 | 1.0000 |

CART achieves perfect smoothness (1.0) and zero recall jump on every dataset. The legacy threshold sweep table further confirms that all precision/recall/F1 values are numerically identical across every threshold level for CART. The study journal records the same observation qualitatively.

**Confidence calibration:** Stated Medium — agrees. Evidence for CART is unambiguous; the "CART-like learners" generalization is not tested beyond a single decision-tree family in this dataset suite.

**Notes:** Three files in the adjacent worktree match the cited glob, all of which independently confirm the quantized behavior. No evidence gap exists.

---

### Claim 6 — XGBoost Exhibits Cliff-like Accessibility Collapse Under Severe Imbalance

**Audit rating:** STRONG

**Files found:**
- `reports/neural_mlp/allocation_regime_summary.csv` ✓
- `reports/neural_mlp/threshold_elasticity_summary.csv` ✓
- `research/study_journal.md` ✓
- `reports/neural_mlp_allocation_geometry_summary.md` ✓

**Files missing:** None. Both specifically cited files exist and contain the claimed data.

**Key evidence:**
From `reports/neural_mlp/allocation_regime_summary.csv`: xgboost — `inferred_regime=cliff_allocator`, `mean_max_recall_jump=0.4887`, `mean_operational_smoothness_index=0.1753`. From `reports/neural_mlp/threshold_elasticity_summary.csv`, per-dataset max_recall_jump: boundary=0.8129, cam=0.5864, compustat=0.4462, oil=0.3143, satimage=0.2838 — cliff behavior is consistent across all 5 datasets, not isolated to one. Journal (2026-05-18): "XGBoost: moderate entropy, but extremely high recall elasticity, strongest cliff behavior observed so far." LightGBM is confirmed as `conservative_allocator` in both worktrees, which the updated claim explicitly acknowledges and contrasts.

**Confidence calibration:** Stated High (upgraded from Medium in the updated inventory) — agrees. The claim is now precisely scoped to XGBoost, the LightGBM distinction is built into the claim text, and the quantitative data across all 5 datasets gives this a level of replication warranting High confidence.

---

### Claim 7 — HDDT and Bagged HDDT Exhibit Broad/Persistent Accessibility

**Audit rating:** STRONG

**Files found:**
- `reports/neural_mlp/allocation_regime_summary.csv` ✓ (hddt=broad_allocator)
- `reports/neural_mlp/allocation_concentration_summary.csv` ✓
- `reports/neural_mlp/prediction_space_occupancy_summary.csv` ✓
- `../hellinger-imbalance-benchmarks/reports/allocation_regime_summary.csv` ✓ (bagged_hddt=broad_allocator)
- `research/study_journal.md` ✓

**Files missing:** None. All four specifically cited evidence files exist and contain the claimed data.

**Key evidence — hddt (current branch):**
From `reports/neural_mlp/allocation_regime_summary.csv`: `hddt` — `inferred_regime=broad_allocator`, `mean_max_recall_jump=0.2690`, `mean_operational_smoothness_index=0.2353`, `mean_effective_support_size=3.2119`. The concentration and occupancy summaries confirm hddt has lower compression and broader persistent support than cliff-allocating comparators across datasets.

**Key evidence — bagged_hddt (adjacent worktree):**
From `../hellinger-imbalance-benchmarks/reports/allocation_regime_summary.csv`: `bagged_hddt` — `inferred_regime=broad_allocator`, mean_max_recall_jump=0.3288, smoothness=0.2486, effective_support_size=3.5315. Per-dataset max_recall_jump values (0.452, 0.354, 0.333, 0.257, 0.248) consistently lower than XGBoost (0.813, 0.586, 0.446, 0.314, 0.284). Journal (2026-05-18): "Bagged HDDT in particular appears: operationally stable, relatively controllable."

**On hddt_forest:**
The updated claim explicitly scopes out `hddt_forest` and notes it as a divergent model_id requiring separate characterization. The inventory's caveat ("The manuscript must not use 'Bagged HDDT/HDDT-forest' slash notation") directly addresses the audit's prior concern. The `allocation_concentration_summary.csv` is cited specifically to document hddt_forest's occupancy properties in context of the divergence, not to support the broad_allocator characterization.

**Confidence calibration:** Stated Medium — agrees. Both `hddt` and `bagged_hddt` are directly confirmed as broad_allocator in separate worktrees. Medium is appropriate because the mechanism is unresolved (Claim 17) and it is not yet clear whether the broad behavior reflects Hellinger splitting, ensemble stabilization, or both.

---

### Claim 8 — Baseline MLP Is Analyzable in the Same Framework

**Audit rating:** STRONG

**Files found:**
- `reports/neural_mlp_allocation_geometry_summary.md` ✓
- `tests/test_models.py` ✓
- `tests/test_runner.py` ✓

**Files missing:** None

**Key evidence:**
`neural_mlp_allocation_geometry_summary.md` §1: "Runner model-family validation was extended with `MLP` in `src/hib/runner.py`; no neural-specific branch logic was introduced." `test_models.py` lines 12–33 confirm `mlp`, `mlp_bce`, `mlp_oversampled`, `mlp_weighted` all appear in the required model set and pass `make_model()` instantiation. The allocation geometry summary shows MLP runs through the same regime synthesis pipeline producing the same output columns as all other models.

**Confidence calibration:** Stated High — agrees. The integration claim is a structural/technical one that is unambiguously confirmed.

---

### Claim 9 — BCE-trained MLP Exhibits Cliff-like Morphology

**Audit rating:** STRONG

**Files found:**
- `reports/neural_mlp_objective_perturbation_summary.md` ✓

**Files missing:** None

**Key evidence:**
From `neural_mlp_objective_perturbation_summary.md` §2, table row: `mlp_bce | 0.7068 | 0.2410 | 0.1346 | 0.7867 | 0.6522 | 0.4178 | 0.4061 | cliff_allocator`. The exact figures cited in the claim (`cliff_allocator`, `max_recall_jump=0.4061`) are directly confirmed. The `reports/neural_mlp_objective_perturbation/allocation_regime_summary.csv` independently confirms: mlp_bce — `inferred_regime=cliff_allocator`, `mean_max_recall_jump=0.4061`.

**Confidence calibration:** Stated High — agrees. Two independent artifacts confirm the same numbers.

---

### Claim 10 — Oversampling/Weighting Can Move MLP Cliff → Smooth

**Audit rating:** STRONG

**Files found:**
- `reports/neural_mlp_objective_perturbation_summary.md` ✓
- `reports/geometry_transition_analysis_summary.md` ✓
- `results/geometry_transition_analysis/geometry_transition_model_means.csv` ✓

**Files missing:** None

**Key evidence:**
From `geometry_transition_model_means.csv`:
- mlp_bce: smoothness=0.4178, max_recall_jump=0.4061, inferred_regime=cliff_allocator
- mlp_oversampled: smoothness=0.5391, max_recall_jump=0.0678, inferred_regime=smooth_allocator
- mlp_weighted: smoothness=0.6080, max_recall_jump=0.1783, inferred_regime=smooth_allocator

Regime shift is explicit in the data. The perturbation summary narrates the transition and the geometry transition summary provides mechanism analysis.

**Confidence calibration:** Stated High — agrees. Three independent artifacts (narrative, analysis, quantitative CSV) all confirm the same finding.

---

### Claim 11 — Operational Morphology Is Not Architecture-only

**Audit rating:** STRONG

**Files found:**
- `reports/neural_mlp_objective_perturbation_summary.md` ✓ (primary)
- `reports/geometry_transition_analysis_summary.md` ✓

**Files missing:** None. The inventory's reference to "mlp_bce vs mlp_oversampled vs mlp_weighted comparisons" is informal but both summary files contain this comparison explicitly.

**Key evidence:**
From `neural_mlp_objective_perturbation_summary.md` §4: "Behavior is not architecture-only: changing imbalance pressure within the same MLP architecture materially changes operational geometry." The architecture is held constant at `hidden_layer_sizes=(64, 32), max_iter=300` across all three variants; only the training objective / sampling strategy differs. The regime shift from cliff to smooth under this controlled perturbation directly supports the claim.

**Confidence calibration:** Stated High — agrees. The experimental design (fixed architecture, variable objective) is methodologically sound for this specific claim.

---

### Claim 12 — Smoothness Is Not Reducible to Support Breadth Alone

**Audit rating:** STRONG

**Files found:**
- `reports/neural_mlp_objective_perturbation_summary.md` ✓
- `reports/geometry_transition_analysis_summary.md` ✓

**Files missing:** None

**Key evidence — dissociation case:**
From `geometry_transition_model_means.csv`:
- mlp_oversampled: smoothness=0.5391, effective_support_size=2.1528
- mlp_bce: smoothness=0.4178, effective_support_size=2.3282

mlp_oversampled achieves *higher* smoothness than mlp_bce while having *lower* effective support size. This is a direct dissociation case: smoothness improved without support broadening (support actually narrowed). The geometry transition summary explicitly narrates this: "mlp_oversampled improves threshold behavior strongly but keeps concentrated low-score mass... implying smoothness is not purely a broad-support effect."

**Confidence calibration:** Stated Medium — *understated*. The dissociation case is quantitatively explicit and confirmed in both the CSV and the narrative summary. The data would support High confidence. Medium may reflect the authors' caution about "some support/compression metrics unstable in edge cases," which is a methodological caveat rather than an evidentiary limitation.

---

### Claim 13 — Calibration Can Improve ECE/Brier While Worsening Accessibility Morphology

**Audit rating:** STRONG

**Files found:**
- `reports/geometry_transition_analysis_summary.md` ✓
- `results/geometry_transition_analysis/calibration_transition_model_means.csv` ✓

**Files missing:** None

**Key evidence — explicit before/after accessibility metrics:**
From `calibration_transition_model_means.csv` (negative delta = improvement; positive jump_delta = worsening):

| model_id | ece_delta | brier_delta | smoothness_delta | jump_delta |
|---|---:|---:|---:|---:|
| mlp_bce | -0.0853 | -0.0210 | **-0.262** | **+0.149** |
| mlp_oversampled | -0.0845 | -0.0466 | **-0.321** | **+0.236** |
| mlp_weighted | -0.2234 | -0.1049 | **-0.440** | **+0.293** |

All three variants show ECE and Brier improvement (reliability improves) alongside smoothness degradation and jump magnitude increase (accessibility morphology worsens). The `regime_persistence_table.csv` files further confirm that mlp_oversampled and mlp_weighted shift from `smooth_allocator` (raw) to `cliff_allocator` (platt/isotonic) — a qualitative regime reversal.

**Confidence calibration:** Stated Medium — *understated*. The calibration_transition_model_means.csv contains precisely the "before/after accessibility metrics showing degradation" that the special-scrutiny instruction asked to verify. The degradation is consistent across all three models and all calibration methods. Data would support High confidence.

**Notes:** This finding is scientifically important precisely because it is counterintuitive. The evidence is unusually clean for such a counterintuitive claim.

---

### Claim 14 — Calibration Quality and Operational Smoothness Are Partially Non-equivalent

**Audit rating:** PARTIAL

**Files found (by inference from related claims):**
- `results/geometry_transition_analysis/calibration_transition_model_means.csv` ✓ (supports claim but not cited)
- `reports/neural_mlp_objective_perturbation/calibration_interaction/regime_persistence_table.csv` ✓ (supports claim but not cited)

**Files missing:**
- No specific file is cited in the inventory. The evidence reference reads only: "Calibration regime persistence and smoothness/jump deltas."

**Key evidence:**
The same CSV that strongly supports Claim 13 also directly supports Claim 14: ECE/Brier improve while smoothness worsens — reliability improvement does not guarantee smoother operational trajectories. The regime persistence tables confirm that calibrated versions of smooth variants revert to cliff_allocator labels.

**Confidence calibration:** Stated Medium — agrees given the citation gap. The underlying data is strong, but a reader following the inventory cannot locate it. Claims 13 and 14 are making closely related assertions from overlapping evidence, and Claim 14's evidence section is effectively empty as written.

**Notes:** Claims 13 and 14 share all primary evidence files (see Cross-cutting Observations). The inventory should cite `results/geometry_transition_analysis/calibration_transition_model_means.csv` and at least one regime persistence table explicitly.

---

### Claim 15 — Regime Taxonomy Is Provisional and Empirical

**Audit rating:** STRONG

**Files found:**
- `research/operational_morphology_framework.md` ✓

**Files missing:** None. "All regime summaries and transition notes" is vague but the framework document is sufficient.

**Key evidence:**
From `operational_morphology_framework.md`, Current Regime Taxonomy section: "The current taxonomy is: provisional, empirical, and revisable." The status section at the top of the document also states: "This document is: a working conceptual framework... It is NOT: a finalized theory."

**Confidence calibration:** Stated High — agrees. This is a meta-claim about framing that is explicitly stated in the primary framework document.

---

### Claim 16 — This Is an Empirical + Conceptual Framing Paper, Not Complete Theory

**Audit rating:** STRONG

**Files found:**
- `research/operational_morphology_framework.md` ✓

**Files missing:** None

**Key evidence:**
From `operational_morphology_framework.md` status section: "This document is: a working conceptual framework, an evolving synthesis artifact, and a research coordination document. It is NOT: a finalized theory, a formal publication draft, or a complete mathematical treatment."

**Confidence calibration:** Stated High — agrees. The scope delimitation is explicit and unambiguous.

---

### Claims 17 & 18 — Future Work Assessment

**Claim 17 — Bagged HDDT Mechanism Remains Unresolved**

Existing evidence is *consistent with* this future direction. The study journal lists Bagged HDDT under open questions: "Does bagging stabilize HDDT posterior allocation? Does it reduce fragmentation? Is this the real modern form of HDDT?" The regime analysis adds complexity — hddt_forest is labeled cliff_allocator while hddt is labeled broad_allocator — leaving mechanism even more opaque than the journal entry suggests. The claim is appropriately hedged; the open-question framing is well-calibrated.

**Claim 18 — Reachability-aware Objectives Are a Cautious Future Direction**

The objective perturbation evidence (cliff → smooth under oversampling/weighting) demonstrates that operational morphology is responsive to training-time objective changes, which is the necessary motivating condition for this direction. The claim's Low confidence and "cautious investigation" framing are appropriate given that no reachability-aware objective has been implemented or tested. The caveat regarding "risk of objective overfitting to specific threshold grid" is specific and honest.

---

## Cross-cutting Observations

### Claims sharing evidence files (potential circularity)

| Claims | Shared file | Risk |
|--------|-------------|------|
| 13 and 14 | `calibration_transition_model_means.csv`, regime persistence tables | High circularity: Claim 14 has no independent citation and is essentially a restatement of Claim 13's conclusion. Consider merging or providing a distinct evidence hook for Claim 14. |
| 10 and 11 | `neural_mlp_objective_perturbation_summary.md`, `geometry_transition_analysis_summary.md` | Moderate — Claim 11 is a causal interpretation of Claim 10's observation; shared evidence is appropriate but should be explicitly acknowledged as drawing from the same experiment. |
| 1 and 2 | `neural_mlp_allocation_geometry_summary.md`, `study_journal.md` | Low — Claims 1 and 2 make distinct assertions (non-equivalence vs. AUROC masking); shared evidence is incidental. |

### Clusters of weak/missing evidence affecting manuscript sections

**Allocation Regimes section (Claims 4, 5, 6, 7):** All four claims are now STRONG. Evidence quality improved in two rounds: (a) Claims 4 and 5 were resolved by discovering the glob patterns in the adjacent worktree; (b) Claims 6 and 7 were revised in the inventory to cite specific files and explicitly scope out the `bagged_hddt`/`hddt_forest` naming ambiguity. No remaining evidence gaps in this section.

**Calibration vs Morphology section (Claims 13, 14):** Claims 13 and 14 are both Core/Medium confidence with overlapping evidence and no independent citation for Claim 14. Claim 13 is actually very strongly supported; Claim 14 needs its own citation.

### Claims where stated confidence seems miscalibrated

| Claim | Stated | Audit Suggests | Reason |
|-------|--------|----------------|--------|
| 12 | Medium | High | The mlp_oversampled dissociation case (higher smoothness, lower support than mlp_bce) is a direct, numerically clean counterexample. |
| 13 | Medium | High | calibration_transition_model_means.csv provides consistent before/after degradation across all models with explicit delta columns. |

### Evidence found in adjacent worktree but not referenced in the inventory

The adjacent worktree (`../hellinger-imbalance-benchmarks/`) contains a full pre-neural baseline analysis at `reports/`. Files present there that are not cited anywhere in the claim inventory but are relevant:

- `reports/allocation_regime_summary.md` — narrative summary of regime taxonomy (supports Claims 4, 15)
- `reports/paper_operational_synthesis.md` — synthesis document (may overlap with Claims 15, 16)
- `reports/lightgbm_weighting_diagnostic.md` — directly relevant to the LightGBM conservative behavior in Claim 6
- `reports/allocation_concentration_summary.csv` / `.md` — occupancy metrics for baseline models (supports Claims 4, 7)
- `reports/prediction_space_occupancy_summary.csv` / `.md` — supports Claim 12 (support breadth vs smoothness distinction exists in baseline data too)

These are additional corroborating artifacts the inventory currently does not cite. They would strengthen several claims if referenced.

### Evidence files present in current branch but not cited in inventory

The following files contain relevant data but are not explicitly cited in any claim:

- `reports/neural_mlp/threshold_elasticity_summary.csv` — per-dataset elasticity intervals; directly relevant to Claims 4, 5, 6
- `reports/neural_mlp/calibration_interaction/calibration_summary_table.csv` — per-model ECE/Brier before/after; directly relevant to Claims 13, 14
- `results/geometry_transition_analysis/elasticity_concentration.csv` — elasticity distribution shift; relevant to Claims 12, 13
- `results/geometry_transition_analysis/occupancy_support_transition.csv` — support breadth vs smoothness; directly relevant to Claim 12
- `reports/neural_mlp_objective_perturbation/allocation_regime_summary.csv` — same data as neural_mlp version but includes mlp_bce/oversampled/weighted rows with baseline comparators; relevant to Claims 4, 9, 10

---

## Top 3 Most Urgent Gaps Before Manuscript Submission

**1. Add specific file citations to Claim 14** *(primary remaining gap)*
Claim 14 has no file path cited anywhere in the inventory. The evidence reference reads only: "Calibration regime persistence and smoothness/jump deltas." The supporting data exists and is strong — it lives in `results/geometry_transition_analysis/calibration_transition_model_means.csv` and `reports/neural_mlp_objective_perturbation/calibration_interaction/regime_persistence_table.csv` — but a reader following the inventory cannot locate it. The inventory should cite both files explicitly (or at minimum the delta CSV, which is the primary quantitative hook).

**2. Differentiate Claim 14 from Claim 13 or merge them**
Claims 13 and 14 currently share all primary evidence and make overlapping assertions (calibration worsens morphology; reliability ≠ smoothness). Either elevate Claim 14 to have its own evidence hook — for example, citing the regime persistence tables (which show qualitative regime reversals, `smooth_allocator → cliff_allocator`, as distinct from the delta metrics in the CSV) — or explicitly frame Claim 14 as a derived corollary of Claim 13 and reduce it to a sub-claim. As written, it gives the appearance of two separately corroborated Core claims when they rest on identical evidence.

**3. Verify cross-branch naming consistency for hddt_forest before submission**
Claim 7 has been revised to explicitly distinguish `hddt` and `bagged_hddt` (both `broad_allocator`) from `hddt_forest` (`cliff_allocator`), which is appropriate. However, the manuscript text, figure labels, and section headings have not yet been audited for stray "HDDT-forest" or "Bagged HDDT/HDDT-forest" slash notation. A reviewer examining both worktrees will immediately notice if any manuscript section uses these terms interchangeably. A targeted text-search of the paper draft before submission is warranted.

---

## Re-audit: Claims 3, 15, 16 — Deep Content Verification

Re-audit date: 2026-05-28
Standard: Direct quotation or statistic required for STRONG rating

---

### Claim 3 — Reachability Adds Trajectory Information Beyond Single-Threshold Recall

**Re-audit rating:** STRONG

**Files read:**
- `results/geometry_transition_analysis/reachability_curves.csv`: fully read (75 data rows, fresh read)
- `reports/geometry_transition_analysis_summary.md`: fully read (in session context, confirmed unchanged)
- `research/operational_morphology_framework.md`: key sections re-read directly (lines 1–35, 96–130, 240–252, 591–601)

**reachability_curves.csv structure:**
- Header: `dataset_id, model_id, threshold, recall_mean, precision_mean, f1_mean`
- 75 data rows: 5 datasets (`boundary`, `cam`, `compustat`, `oil`, `satimage`) × 3 models (`mlp_bce`, `mlp_oversampled`, `mlp_weighted`) × 5 thresholds (`0.5`, `0.25`, `0.1`, `0.05`, `0.01`)
- Recall range: 0.0 to 0.9955

**Direct evidence — CSV trajectory dissociation:**

The following two rows from the CSV have identical recall at threshold 0.5 (`recall_mean = 0.0` for both) but completely different trajectory shapes:

```
boundary,mlp_bce,0.5,0.0,0.0,0.0
boundary,mlp_bce,0.25,0.8838709677419354,...
boundary,mlp_bce,0.1,1.0,...
```
(cliff: 0 → 0.884 in a single threshold step; fully recovered by threshold 0.1)

```
compustat,mlp_bce,0.5,0.0,0.0,0.0
compustat,mlp_bce,0.25,0.0023076923076923,...
compustat,mlp_bce,0.1,0.0138461538461538,...
compustat,mlp_bce,0.05,0.3246153846153846,...
compustat,mlp_bce,0.01,0.946153846153846,...
```
(cliff concentrated at a different interval: near-zero through threshold 0.1, then 0.014 → 0.325 → 0.946 in the 0.10→0.05→0.01 range)

Single-threshold recall at 0.5 returns `0.0` for both. The trajectory reveals these are fundamentally different operational situations: on `boundary` a threshold of 0.25 recovers 88% minority accessibility; on `compustat` a threshold of 0.25 recovers less than 0.3% and a threshold of 0.10 recovers less than 1.5%.

Additional trajectory contrast (same dataset, different models):

```
boundary,mlp_bce,t=[0.5,0.25,0.1,0.05,0.01]:  recall = [0.000, 0.884, 1.000, 1.000, 1.000]
boundary,mlp_oversampled,t=[0.5,...]:            recall = [0.139, 0.219, 0.326, 0.403, 0.529]
```

At threshold 0.01, mlp_bce (1.0) outperforms mlp_oversampled (0.529). At threshold 0.5, mlp_oversampled (0.139) outperforms mlp_bce (0.0). Model ranking reverses depending on threshold. The trajectory is required to know this.

**Direct evidence — geometry_transition_analysis_summary.md:**

> "mlp_bce shows abrupt jumps concentrated at one interval (notably 0.50→0.25 on boundary, 0.05→0.01 on compustat). mlp_oversampled redistributes reachability gain across intervals, with smaller per-interval cliffs."

This directly confirms that (a) the trajectory data reveals *where* collapse occurs, which varies by dataset for the same model, and (b) this information is invisible from any single threshold measurement.

**Direct evidence — operational_morphology_framework.md:**

> "Candidate Reachability Features: ... decay rate (accessibility steepness) ... derivative localization (where collapse occurs) ... curvature (smooth vs phase-like decay)"

The framework explicitly lists features of the trajectory that cannot be derived from a single recall value, providing conceptual grounding for the claim.

**Corroboration:** The CSV is not merely topically related — it is the artifact that directly demonstrates the claim. The two rows above (boundary vs. compustat, both mlp_bce, both recall=0.0 at threshold 0.5) constitute a concrete counterexample to single-threshold sufficiency.

**Confidence calibration:** High → **confirmed**. The dissociation is numerically explicit, present in multiple dataset pairs, and independently narrated in the geometry transition summary. The trajectory data demonstrates accessibility evolution that is demonstrably invisible from single-threshold recall.

---

### Claim 15 — Regime Taxonomy Is Provisional and Empirical

**Re-audit rating:** STRONG

**Files treated as "regime summaries and transition notes":**

The vague citation "all regime summaries and transition notes" was resolved to the following specific files, all of which have been read:

1. `reports/neural_mlp/allocation_regime_summary.csv` — current-branch model regime assignments (fully read)
2. `reports/neural_mlp_objective_perturbation/allocation_regime_summary.csv` — perturbation experiment regime assignments (read in prior session, confirmed in audit)
3. `../hellinger-imbalance-benchmarks/reports/allocation_regime_summary.csv` — pre-neural baseline regime assignments (read in prior session)
4. `reports/geometry_transition_analysis_summary.md` — regime transition analysis narrative (read in session)
5. `reports/neural_mlp_allocation_geometry_summary.md` — MLP integration regime summary (read in session)
6. `reports/neural_mlp_objective_perturbation_summary.md` — perturbation regime comparison narrative (read in session)

**Direct evidence — operational_morphology_framework.md, "Current Regime Taxonomy" section (lines 243–249):**

> "The current taxonomy is:
> * provisional,
> * empirical,
> * and revisable."

This is the primary and most direct evidence. The quote is unambiguous.

**Supporting evidence from regime summary CSVs:**

All three `allocation_regime_summary.csv` files use `inferred_regime` as the column name — not `regime`, `assigned_regime`, or `classified_regime`. The word "inferred" in the column header is structural evidence that the regime assignments are treated as heuristic outputs of an inference procedure, not as definitive category memberships. From `reports/neural_mlp/allocation_regime_summary.csv`, header row:

> `model_id,...,inferred_regime`

This naming convention is consistent across the current branch and the adjacent worktree. It corroborates the framework's provisional framing at the artifact level.

**Corroboration:** The claim's primary assertion ("regime labels are working empirical categories, not final theory classes") is directly and explicitly supported by the framework document. The regime CSV column naming provides structural corroboration. No file in the corpus uses language suggesting finality or completeness for the taxonomy.

**Confidence calibration:** High → **confirmed**. The framework quote is direct, precise, and intentional. The claim is a meta-claim about the taxonomy's status, and the framework document is the correct and authoritative source for that status.

**Note on the vague citation:** "All regime summaries and transition notes" cannot be independently verified by a reader — it requires knowing which files to enumerate. This is a documentation gap in the inventory (Claim 15 would benefit from listing specific file names) but it does not weaken the evidence quality, since the primary evidence hook is the framework document's explicit quote.

---

### Claim 16 — This Is an Empirical + Conceptual Framing Paper, Not Complete Theory

**Re-audit rating:** STRONG

**"Status section" location:** Section heading `# Status of This Document`, first major section after the document title and subtitle. Lines 7–32 of `research/operational_morphology_framework.md`.

**Direct evidence — lines 9–19:**

> "This document is:
> * a working conceptual framework,
> * an evolving synthesis artifact,
> * and a research coordination document.
>
> It is NOT:
> * a finalized theory,
> * a formal publication draft,
> * or a complete mathematical treatment."

**Corroborating evidence — "Current Position of the Research Program" section (lines 591–601):**

> "The project no longer appears primarily concerned with: 'Which classifier performs best?' Instead, it increasingly concerns: how classifiers allocate operationally accessible posterior probability mass under severe class imbalance. This now appears to be the central organizing principle of the research direction."

This second quote establishes that the research program is organized around a conceptual reframing of the problem, not around deriving or proving a theory.

**Corroboration:** The Status section quote directly supports the claim's substance. One structural nuance deserves noting: the Status section says "This document is NOT: a finalized theory, a formal publication draft" — the subject is the *framework document itself*, not the manuscript. The claim (Claim 16) is about the *manuscript's* scope. This is one inference step removed: the framework document establishes the research program's conceptual orientation, and the manuscript is expected to carry that orientation forward.

In context this is the correct level of evidence — a research program's scope is established in its coordination documents, and the framework document is explicitly that. No file in the corpus asserts that the manuscript aims to prove a complete formal theory; all content is consistent with an empirical/framing paper.

**Confidence calibration:** High → **confirmed**. The Status section quote is direct. The single-step inference from "framework orientation" to "manuscript positioning" is well-grounded and consistent with every other document in the corpus.

---

### Re-audit Summary

| Claim | Original Rating | Re-audit Rating | Confidence Change |
|-------|----------------|-----------------|-------------------|
| 3  | STRONG | STRONG | High → confirmed |
| 15 | STRONG | STRONG | High → confirmed |
| 16 | STRONG | STRONG | High → confirmed |

**Overall assessment:** The original STRONG ratings were correctly assigned. The re-audit confirms that direct quotations and concrete statistics support all three claims. The original audit was not overconfident on these claims.

Two observations that have implications for the manuscript:

1. **Claim 3's strongest evidence is a numerical dissociation in the CSV that the original audit did not quote explicitly.** The `boundary`/`compustat` mlp_bce comparison (both recall=0.0 at threshold 0.5, but cliff location at completely different threshold intervals) is the clearest available illustration of why single-threshold recall is insufficient. This example should appear in the manuscript body or a figure caption, not just be implied by the reachability curve plots.

2. **Claim 16 rests on evidence about the framework document's status, not the manuscript's.** This is appropriate — the framework document is the correct source for research program positioning — but the manuscript itself should contain explicit scope-delimiting language ("this paper proposes a vocabulary and empirical framing; formal theory is future work") to make the claim self-supporting once published. Currently the evidence lives only in an internal research coordination document.
