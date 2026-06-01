# Manuscript v0.2 Change Log

## Scope of Revision

This pass integrates related-work positioning and terminology consistency improvements into `paper/manuscript_v0_1.md` without introducing new scientific claims.

## Major Revisions

1. Added full Related Work section
- Inserted a complete `## 2. Related Work` section with six subsections:
  - severe class imbalance learning
  - evaluation under severe imbalance
  - selective prediction/coverage/threshold analysis
  - calibration and decision-focused probability estimation
  - decision-theoretic and operational perspectives
  - summary positioning
- Integrated language from `paper/related_work_draft.md` and aligned to deep-dive conclusions.

2. Strengthened introduction framing
- Reinforced threshold-mediated deployment motivation.
- Explicitly preserved empirical + conceptual framing posture.
- Avoided algorithmic novelty or formal-theory framing.

3. Reachability section alignment
- Harmonized terminology with `research/terminology.md`:
  - reachability
  - accessibility persistence
  - operational smoothness
  - elasticity localization
- Added explicit acknowledgment of overlap with class-conditional coverage/selective prediction.

4. Morphology section de-risking
- Replaced ontology-like language with:
  - recurring empirical patterns
  - heuristic/provisional shorthand
- Kept taxonomy bounded and non-formal.

5. Calibration section tightening
- Elevated reliability vs accessibility non-equivalence clarity.
- Explicitly states calibration remains valuable.
- Reduced anti-calibration interpretation risk.

6. Discussion and deployment relevance
- Expanded operational ML relevance:
  - threshold policy as control mechanism
  - queue/triage controllability implications

7. Limitations expansion
- Added explicit limitations on:
  - dataset scope
  - literature overlap risks
  - provisional terminology
  - absence of formal theory claims

8. Structural updates
- Renumbered manuscript sections after adding Related Work.
- Updated placeholder citation checklist to include selective prediction anchor.

## Claims Impact

- No new claims introduced.
- Existing claims were reframed for reviewer clarity and overlap acknowledgment.

## Reviewer Attack Surface Reduction

Addressed likely concerns more explicitly:
- “reachability is just coverage” (acknowledged overlap + differentiated emphasis)
- “anti-calibration” interpretation (explicitly rejected)
- “taxonomy overclaim” risk (provisional language reinforced)
- “benchmark paper” misclassification (positioning reiterated)
