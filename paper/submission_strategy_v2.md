# Executive Summary (v2)

Current paper identity:

An empirical severe-imbalance evaluation/framing manuscript on **threshold-mediated operational accessibility**, now strengthened by family-level evidence (pre-neural), a high-clarity pairwise example (Bagged HDDT vs LightGBM), and an explicit metric-semantics audit.

Center-of-gravity shift since prior strategy:

- **Before:** MLP transition and calibration interaction were carrying most of the narrative weight.
- **Now:** family-level accessibility non-equivalence is the primary foundation; neural and calibration sections read as explanatory extensions.

## Identity Re-Evaluation

### Is this a severe-imbalance paper?

Yes, primarily. The datasets, protocol, and core claims remain severe-imbalance-first.

### Is this a threshold-analysis paper?

Yes, strongly. Threshold-evolution behavior is now visibly central and no longer MLP-dependent.

### Is this selective-prediction-adjacent?

Yes, secondarily. Reachability is adjacent to class-conditional coverage ideas and should continue to be framed as an operational reinterpretation.

### Is this an operational-ML paper?

Yes, as framing/application layer. It offers deployment-relevant diagnostics (controllability, jump risk), though still without explicit queue simulation.

### Is this a deployment-behavior paper?

Yes, in evaluation framing terms. It is not yet a production case-study paper.

## Updated Paper Type Statement

Best single-line positioning:

**A severe-imbalance threshold-behavior evaluation paper with operational-ML framing and selective-prediction-adjacent semantics.**

## Venue Reassessment

### JMLR

Fit:

- Improved relative to v1 due to stronger family-level evidence hierarchy.

Risk:

- Metric semantics issue (if unresolved) and limited external breadth/theory can still block.

Probability:

- Medium-Low

### MLJ

Fit:

- Good fit for empirical evaluation framing with explicit methodological caveats.

Risk:

- Will likely request either broader data families or stronger formalization.

Probability:

- Medium

### ECML PKDD

Fit:

- Stronger than v1: family-level phenomenon -> neural transition -> calibration interaction is a cleaner conference story.

Risk:

- Must proactively neutralize "rebranded coverage" and metric-label ambiguity.

Probability:

- Medium-High

### KDD Applied Data Science

Fit:

- Better than v1 if framed as threshold-policy risk diagnostics for deployment teams.

Risk:

- Lack of explicit operational simulation/KPI validation may reduce applied impact score.

Probability:

- Medium

### AAAI Workshops

Fit:

- Very strong for cross-community positioning and fast feedback.

Risk:

- Lower archival weight; variable reviewer depth.

Probability:

- High

### NeurIPS Workshops

Fit:

- Strong for selective prediction + calibration + operational behavior intersection.

Risk:

- Novelty can still be judged as framing unless pairwise and regime-map evidence are foregrounded.

Probability:

- High-

### Operational ML venues

Fit:

- Strong conceptual fit now that family-level diagnostics are explicit.

Risk:

- Need at least one concrete KPI bridge paragraph (or appendix) for best reception.

Probability:

- Medium-High

## Strategy Change Recommendation

Recommended sequence (updated):

1. **Immediate fix before submission:** resolve metric-semantics mismatch (`Accessibility Persistence` label).
2. **Primary target lane:** ECML PKDD / strong ML conference route.
3. **Parallel insurance lane:** NeurIPS/AAAI workshop submission for framing feedback and citation foothold.
4. **Journal route after reinforcement:** MLJ (or JMLR if dataset/calibration replication is expanded).

## Updated Risk Register

### Reduced Risks (vs v1)

- "MLP-centric evidence" risk: reduced from High -> Medium.
- "No clear motivating example" risk: reduced from Medium -> Low (Bagged HDDT vs LightGBM).

### Persistent/Heightened Risks

- Metric naming/semantics mismatch: **High until fixed**.
- Selective-prediction overlap interpretation: **High**.
- External validity breadth: **Medium-High**.

## Final Assessment

The manuscript center has shifted in a favorable way: it now reads first as a family-level severe-imbalance threshold-behavior paper, with neural and calibration sections as follow-on structure. Venue strategy should shift slightly toward **ECML PKDD/main-track viability**, provided semantic cleanup is completed before submission.
