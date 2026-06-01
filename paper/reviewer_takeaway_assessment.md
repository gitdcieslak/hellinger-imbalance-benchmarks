# Reviewer Takeaway Assessment

## Core Question

What will reviewers most likely remember first after reading the current manuscript stack (`manuscript_v0_4` + new family figures + regime analysis + pairwise exposition + metric audit)?

Candidates:

1. Reachability
2. Calibration-accessibility tension
3. Bagged HDDT vs LightGBM
4. Neural morphology transition

## Likely Recall Ranking (Current State)

1. **Ranking-quality similarity with operational divergence** (best represented by Bagged HDDT vs LightGBM)
2. **Family-level accessibility trajectory structure** (reachability + regime map)
3. **Neural morphology transition (`mlp_bce -> oversampled/weighted`)**
4. **Calibration-accessibility tension**

Interpretation:

- "Reachability" as a term may not be the first remembered noun.
- The remembered claim is likely the **non-equivalence pattern** (similar ranking, different operational behavior), with reachability as the analytical vehicle.

## Candidate-by-Candidate Assessment

### 1) Reachability

Strength:

- High analytical coherence across manuscript.

Recall profile:

- Medium-high among technical reviewers; medium among broad reviewers.

Risk:

- May be mentally filed as class-conditional coverage unless framed repeatedly as operational lens.

### 2) Calibration-Accessibility Tension

Strength:

- Distinctive and interesting.

Recall profile:

- Medium; often remembered as caveat/interaction, not main thesis.

Risk:

- Can trigger anti-calibration misread if overemphasized without boundedness language.

### 3) Bagged HDDT vs LightGBM

Strength:

- Very high pedagogical clarity; immediate contradiction of "AUROC closeness implies similar deployment behavior."

Recall profile:

- High; likely the most quotable example in reviews/discussion.

Risk:

- Needs metric-semantics cleanup to avoid undermining trust in persistence wording.

### 4) Neural Morphology Transition

Strength:

- Strong controlled result and mechanism-oriented bridge.

Recall profile:

- Medium-high, but now less likely to dominate first impression because family-level evidence comes earlier.

Risk:

- If over-positioned, paper can drift back to "MLP case study" perception.

## Dominant Contribution Now

Most likely dominant remembered contribution:

**A family-level demonstration that ranking-quality similarity can mask large threshold-mediated operational-accessibility differences, with the Bagged HDDT vs LightGBM contrast as the cleanest anchor example.**

Reachability remains the core analytical framing, but the pairwise contrast now carries stronger memory salience.

## Messaging Recommendation for Submission

Use a two-layer headline:

1. **Memory hook:** "Similar AUROC, different operational accessibility behavior" (Bagged HDDT vs LightGBM example).
2. **Method framing:** "Reachability trajectories and threshold-sensitivity metrics explain why."

Then position neural transition and calibration interaction as extensions that reinforce generality of the non-equivalence framing.

## Dependency on Metric Semantics Fix

Current caveat:

- The persistence-label mismatch identified in `paper/metric_semantics_audit.md` can dilute this takeaway if not corrected.

If fixed:

- Reviewer memory and confidence should both improve, and the paper’s center-of-gravity shift (family-level first, neural second) becomes more robust.
