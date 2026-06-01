# Contribution Audit (manuscript_v0_2)

## Scope

This audit reviews contribution statements in the **title, abstract, introduction, discussion (Section 10), and conclusion (Section 13)** of `paper/manuscript_v0_2.md`, and maps each to available evidence from:

- `research/claim_inventory.md`
- `research/claim_audit_report.md`
- `paper/literature_deep_dive.md`

The guiding question is: **Are we claiming exactly what the evidence supports?**

---

## Contribution Statement Table

| Location | Statement | Evidence | Confidence | Overclaim Risk |
| --- | --- | --- | --- | --- |
| Title | “Operational Accessibility Trajectories Under Severe Class Imbalance” | Claims 1–3, 10–13; strong multi-report trajectory evidence | High | Safe |
| Abstract | AUROC/AP/calibration can be partially non-equivalent to operational accessibility | Claims 1, 2, 13, 14 (audit: strong for 1/2/13; partial for 14 path specificity) | Medium-High | Borderline (needs explicit “in this setting”) |
| Abstract | Reachability trajectories are a central analytical object | Claim 3 audit strong; deep-dive positions as reframing of class-conditional coverage | High | Borderline (if read as wholly new object) |
| Abstract | Fixed-architecture MLP perturbations move cliff-like to smoother behavior | Claims 9–11, audit strong | High | Safe |
| Abstract | Calibration tension: ECE/Brier improve while smoothness declines/jumps rise | Claim 13 audit strong; transition delta CSV explicit | High | Safe |
| Abstract | “empirical + conceptual framing paper” and non-claims | Claims 15–16 audit strong; framework status explicit | High | Safe |
| Introduction | Ranking quality does not reliably determine threshold-level accessibility | Claims 1–2 audit strong, numeric anchors provided | High | Safe |
| Introduction | Threshold-mediated accessibility trajectories are operationally relevant objects | Claims 3, 16 + operational framing docs | Medium-High | Safe |
| Introduction | Fixed-architecture neural perturbation tests imbalance-pressure sensitivity | Claims 10–11 audit strong | High | Safe |
| Introduction | “operational accessibility appears partially independent from ranking and calibration quality” | Claims 1, 13, 14 + transition/calibration evidence | Medium | Borderline (should stay “appears” and setting-bounded) |
| Discussion (Sec 10) | Recurring morphology patterns summarize behavior as empirical shorthand | Claims 4, 15 audit strong/provisional | Medium-High | Safe |
| Discussion (Sec 10) | Jointly emergent view: ranking, calibration, accessibility are partially independent axes | Claims 1, 13, 14 + deep-dive framing | Medium | Borderline (needs explicit non-universal scope) |
| Discussion (Sec 10) | Operational relevance for threshold-policy control in queue/triage contexts | Deep-dive operational ML alignment; conceptual extrapolation | Medium | Borderline (application extrapolation) |
| Conclusion | Accessibility should be analyzed as a first-class empirical object | Claims 1–3 + audit strong | High | Safe |
| Conclusion | Reachability + elasticity localization + persistence provide practical lens | Claims 3, 12, 13 supported across transition artifacts | High | Safe |
| Conclusion | MLP perturbations support imbalance-pressure sensitivity | Claims 9–11 audit strong | High | Safe |
| Conclusion | Calibration tension as meaningful finding | Claim 13 audit strong; claim 14 partial support path noted | Medium-High | Safe-Borderline |
| Conclusion | Framing contribution, not complete theory/benchmark endpoint | Claims 15–16 audit strong | High | Safe |

---

## Contribution Classification

## A) Empirical Findings

1. Ranking quality can coexist with default-threshold accessibility collapse under severe imbalance.
2. Reachability trajectories expose threshold-evolution behavior not captured by single-threshold recall.
3. `mlp_bce -> mlp_oversampled/mlp_weighted` morphology transition under fixed architecture.
4. Calibration can improve ECE/Brier while worsening smoothness/jump behavior.

Assessment: **well supported** (mostly STRONG in claim audit).

## B) Analytical Contributions

1. Reachability-centered analysis of minority accessibility trajectories.
2. Elasticity localization as threshold sensitivity decomposition.
3. Accessibility persistence + smoothness lens for threshold-policy robustness.

Assessment: **supported as framing/analysis layer**, not as wholly new mathematical formalism.

## C) Conceptual Framing Contributions

1. Operational accessibility framing for threshold-mediated deployment.
2. Partial non-equivalence framing among ranking quality, reliability quality, and accessibility behavior.
3. Provisional morphology vocabulary as empirical shorthand.

Assessment: **supported with caveats**; must remain explicitly bounded and provisional.

## D) Speculative / Future Directions

1. Reachability-aware objective exploration.
2. Mechanistic explanation for Bagged HDDT broad behavior.
3. Broader generalization beyond current dataset/model scope.

Assessment: **appropriately speculative** if retained as future work only.

---

## Overclaim Analysis

### Safe

- Ranking/accessibility non-equivalence in this severe-imbalance experimental setting.
- MLP perturbation transition (`cliff -> smooth`) under fixed architecture.
- Calibration reliability gains coexisting with morphology degradation in observed runs.
- Empirical + conceptual framing (not complete theory).

### Borderline

- “Operational accessibility partially independent from ranking and calibration” (strong as setting-specific; weak as broad general law).
- Operational deployment implications (queue/triage) when not directly validated on deployment datasets.
- Reachability wording if interpreted as claiming fundamentally new object instead of reframing.

### Risky

- Any wording implying universal regime taxonomy.
- Any implication that calibration generally harms deployment behavior.
- Any language suggesting formal theory completion.

---

# Recommended Contribution Statement

## Primary Contributions

1. **Empirical non-equivalence evidence:** Under severe class imbalance, ranking and calibration summaries can underdescribe threshold-mediated minority accessibility behavior.
2. **Trajectory-centric analysis:** Reachability and elasticity localization provide an operationally relevant view of accessibility evolution across thresholds.
3. **Fixed-architecture perturbation result:** Objective/sampling pressure can move MLP accessibility behavior from cliff-like to smoother patterns.
4. **Calibration interaction finding:** Reliability improvements can coexist with degraded operational smoothness in this setting.

## Secondary Contributions

1. Provisional morphology vocabulary for recurring empirical patterns.
2. Deployment-oriented interpretation connecting threshold-policy adjustments to accessibility controllability.

## Explicit Non-Claims

- No new learner or optimization method is proposed.
- No complete operational morphology theory is claimed.
- No universal classifier taxonomy is claimed.
- No definitive causal mechanism for Bagged HDDT is claimed.
- No universal anti-calibration claim is made.

---

## Abstract Audit

Current abstract assessment:

- Strong and mostly aligned with evidence.
- Correctly includes non-claims and bounded language.
- Main residual risk: reachability phrasing may be interpreted as a wholly new object unless overlap with class-conditional coverage is acknowledged.

Recommended abstract revisions:

1. Add one clause to reachability sentence: “closely related to class-conditional coverage trajectories, but used here as an operational deployment object under severe imbalance.”
2. Keep calibration tension sentence but append “in this constrained severe-imbalance setting.”
3. Preserve current explicit non-claims verbatim.

---

## Conclusion Audit

Current conclusion assessment:

- Coherent and evidence-aligned.
- Correctly reiterates framing contribution and non-theory posture.
- Mild risk in broader interpretive phrasing (“first-class empirical object”) if read as universal.

Recommended conclusion revisions:

1. Qualify opening line: “in threshold-mediated severe-imbalance settings such as those studied here.”
2. Add one sentence reinforcing overlap-aware positioning: “This trajectory lens is closely related to class-conditional coverage analysis but emphasizes operational accessibility persistence and smoothness under threshold policy change.”
3. Retain explicit boundedness and future-work language.

---

## Final Assessment

The manuscript’s contribution profile is largely well calibrated to available evidence. Most risk sits in **interpretive breadth**, not empirical claims. With minor phrasing adjustments (especially abstract/conclusion scope qualifiers and overlap acknowledgement), the contribution framing should be reviewer-resilient.
