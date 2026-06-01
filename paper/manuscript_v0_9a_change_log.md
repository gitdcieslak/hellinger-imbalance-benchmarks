# manuscript_v0_9a Change Log

Base file: `paper/manuscript_v0_9.md`
Output file: `paper/manuscript_v0_9a.md`
Pass type: reviewer-flow polish and sequencing cleanup

## Summary

This pass improves flow, sequencing, and readability without changing scientific claims, experiments, metrics, contributions, or conclusions.

## 1) Abstract reordering

- Reordered abstract logic to mirror manuscript argument flow:
  1. severe-imbalance deployment problem,
  2. reachability lens,
  3. family-level non-equivalence,
  4. fixed-architecture MLP perturbation finding,
  5. calibration-accessibility interaction,
  6. non-claims.

## 2) Related Work consolidation

- Merged former Section 2.5 into Section 2.4.
- Renamed Section 2.4 to **Calibration, Decision Analysis, and Operational ML**.
- Renumbered summary subsection from `2.6` to `2.5`.
- Preserved calibration, decision-theoretic, and operational-ML positioning and citations.

## 3) Oil dataset callout after Figure 2

- Added short observational note acknowledging near-flat `oil` reachability behavior and limited threshold leverage in that regime.
- Framed conservatively as dataset-dependent accessibility geometry.

## 4) Oversampling objection response strengthened

- Replaced high-level wording with more concrete interpretation using existing evidence:
  - dominant jump interval changes,
  - accessibility redistribution across threshold bands,
  - reduced elasticity localization,
  - not consistent with pure threshold translation alone.

## 5) Metric-semantics relocation

- Moved persistence clarification from Section 10 to Section 4 near reachability/persistence definitions.
- Removed duplicate clarification from synthesis section to improve narrative continuity.

## 6) Figure hierarchy confirmation

- Confirmed main-text emphasis on jump-vs-persistence regime map (Figure 7).
- Kept smoothness-vs-persistence and recovery-vs-persistence as appendix-oriented complements.
- Updated `paper/figure_main_vs_appendix_recommendation.md` header/decision label to v0.9a.

## 7) Readability cleanup

- Reduced transition friction and improved section lead-ins where reviewers flagged sequencing.
- Preserved terminology consistency across operational accessibility, reachability, persistence, and smoothness.

## Notes

- A final micro-pass was applied after `paper/reviewer_feedback_round2.md` became available, tightening phrasing in the abstract, Section 2.4 transition, and the Section 8 oversampling-objection response while preserving all claims and evidence scope.
