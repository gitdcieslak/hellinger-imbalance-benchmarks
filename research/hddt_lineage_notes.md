# HDDT Lineage Notes

## Sources Reviewed

- `Hellinger distance decision trees are robust and skew-insensitive` (Cieslak, Hoens, Chawla, Kegelmeyer; DMKD 2012; DOI `10.1007/s10618-011-0222-1`)
- Existing project context docs:
  - `research/bagged_hddt_mechanism_study.md`
  - `research/bagged_hddt_experiment_c.md`

## Original Motivation (HDDT)

- The HDDT line targets a core imbalance problem: standard tree split criteria (for example entropy/gain-ratio style criteria) can be sensitive to class skew, which can degrade minority-class treatment.
- HDDT proposes using Hellinger distance as the split criterion to reduce skew sensitivity and reduce dependence on sampling complexity.

## Core Contribution

- HDDT introduces Hellinger-distance-based split selection for decision trees in imbalanced settings.
- The DMKD paper emphasizes analytical and empirical skew-insensitivity of the Hellinger criterion relative to common alternatives.
- The paper reports broad empirical evaluation under imbalance and argues practical competitiveness, including balanced-data behavior.

## Relationship to Severe Imbalance Framing

- The HDDT program explicitly frames imbalance as a first-order design target, not an afterthought.
- Minority preservation and robustness to skew are central motivations.
- This supports using HDDT-family models as principled comparators in severe-imbalance studies where minority accessibility is operationally important.

## Ensemble / Bagged HDDT Notes

- The DMKD HDDT paper includes ensemble context and reports that HDDT with bagging is effective in imbalanced regimes.
- A practical takeaway stated in the paper is that bagged HDDT can be used without requiring additional sampling methods in many imbalanced scenarios.
- This is directly relevant to current manuscript framing: bagged HDDT is not an ad hoc model choice, but part of prior imbalance-oriented HDDT practice.

## Manuscript-Relevant Takeaways

1. **Why include HDDT-family models:** They were designed for skew-insensitive learning under imbalance, making them natural anchors for accessibility-oriented evaluation.
2. **What not to claim:** Prior HDDT papers do not present the same accessibility-geometry framework used here; current work is a reinterpretation lens, not a claim of historical continuity of identical metrics.
3. **How to connect safely:**
   - Prior HDDT: skew-insensitive split design + imbalance robustness.
   - Current manuscript: threshold-mediated operational accessibility analysis across model families, including HDDT variants.
4. **Bagged HDDT guardrail:** Current findings show stable directional recovery/jump effects under resampling, while mechanism remains inferential and should stay in future-work space.

## Candidate Citation Use in Manuscript

- `CieslakChawla2008` for original HDDT introduction.
- `CieslakEtAl2012` for skew-insensitive robustness framing and bagged-HDDT practical guidance.

## Safe Wording Snippets

- "The HDDT line was introduced to improve skew-insensitive tree induction under severe imbalance; we use HDDT-family models as principled imbalance-aware comparators."
- "Our manuscript revisits these families through an operational accessibility lens; it does not claim that prior HDDT work studied accessibility geometry in this form."
- "Bagged HDDT behavior in our benchmark is consistent with prior imbalance-oriented HDDT motivation, while causal mechanism attribution remains future work."
