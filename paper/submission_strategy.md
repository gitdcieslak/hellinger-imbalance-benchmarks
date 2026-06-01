# Executive Summary

Current paper identity:

An empirical + conceptual framing manuscript on **operational accessibility under severe class imbalance**, centered on threshold-mediated trajectory analysis (reachability, accessibility persistence, operational smoothness, elasticity localization), with a controlled MLP perturbation result and a bounded calibration-accessibility interaction finding.

### What is this paper?

- A severe-imbalance evaluation/analysis paper with deployment-facing interpretation.
- A trajectory-centric framing that complements (not replaces) ranking and reliability metrics.
- A bounded evidence paper: strong within-scope empirical claims, explicit non-claims.

### What is it not?

- Not a new learner, loss, or optimization method.
- Not a complete theory of accessibility morphology.
- Not a universal classifier taxonomy or benchmark leaderboard paper.
- Not an anti-calibration manifesto.

### Who will appreciate it?

- Researchers in imbalance evaluation who care about threshold behavior.
- Operational ML reviewers focused on policy controllability and deployment stability.
- Applied ML practitioners managing review queues/triage thresholds.

### Who may misunderstand it?

- Selective prediction reviewers who may read reachability as rebranded class-conditional coverage.
- Calibration reviewers who may infer anti-calibration claims.
- Algorithm-centric reviewers expecting method novelty over framing/value-of-analysis novelty.

# Contribution Classification

## Empirical

- Ranking/accessibility non-equivalence under severe imbalance.
- Default-threshold collapse with low-threshold recovery patterns.
- Fixed-architecture MLP transition (`mlp_bce -> mlp_oversampled`/`mlp_weighted`).
- Calibration reliability gains coexisting with degraded operational smoothness/jump behavior.

## Analytical

- Reachability trajectory lens across threshold evolution.
- Elasticity localization for where sensitivity concentrates.
- Accessibility persistence + operational smoothness for threshold-policy robustness interpretation.

## Conceptual

- Operational accessibility framing as a first-class deployment analysis target.
- Three-view synthesis: ranking, calibration, accessibility as separate but interacting views.
- Provisional recurring-pattern vocabulary as empirical shorthand.

## Speculative

- Reachability-aware objectives as future direction.
- Mechanistic explanation for bagged HDDT behavior.
- Broader cross-domain generalization claims.

| Contribution | Strength |
| ------------ | -------- |
| Ranking/accessibility non-equivalence evidence | High |
| MLP fixed-architecture morphology transition | High |
| Calibration-accessibility interaction (bounded setting) | Medium-High |
| Reachability + elasticity + persistence analytical bundle | Medium-High |
| Operational accessibility conceptual framing | Medium |
| Recurring-pattern vocabulary | Medium |
| Bagged HDDT mechanism explanation | Low (currently unresolved) |
| Reachability-aware objective direction | Low (speculative) |

# Reviewer Archetypes

## Imbalance Researcher

Reaction:

- Likely praise: strong severe-imbalance empirical framing; clear fixed-architecture perturbation result.
- Likely criticism: limited dataset/model breadth; no new algorithmic method.
- Strongest defense: the contribution is evaluation/framing precision under deployment thresholds, with explicit non-claims and repeated-split evidence.

## Selective Prediction Researcher

Reaction:

- Likely praise: explicit threshold trajectory focus and overlap acknowledgment.
- Likely criticism: reachability may appear equivalent to class-conditional coverage/risk-coverage ideas.
- Strongest defense: explicitly position reachability as an operational reinterpretation with minority accessibility persistence, elasticity localization, and calibration interaction integration.

## Calibration Researcher

Reaction:

- Likely praise: nuanced reliability-vs-deployment behavior analysis.
- Likely criticism: concern effect is calibrator/split artifact; anti-calibration misread risk.
- Strongest defense: manuscript repeatedly states calibration remains valuable and claims only partial non-equivalence in this severe-imbalance setting.

## Operational ML Researcher

Reaction:

- Likely praise: threshold-policy controllability framing is practical and actionable.
- Likely criticism: limited direct KPI/queue simulation linkage.
- Strongest defense: paper is intentionally a diagnostic framing layer; simulations are a clear next step, not implicit claim.

## Applied ML Reviewer

Reaction:

- Likely praise: interpretable narrative around why AUROC/AP alone can miss deployment risk.
- Likely criticism: terminology may still feel abstract without cookbook workflow.
- Strongest defense: figures and compact contribution framing provide adoption path without requiring new modeling stack.

# Venue Analysis

### JMLR

Fit:

- Moderate fit for a careful empirical + conceptual framing paper if positioning stays conservative.

Risk:

- High expectations for breadth/theory could challenge acceptance without expanded replication or formalism.

Probability:

- Medium-Low

### Machine Learning Journal

Fit:

- Good fit for evaluation/framing contributions with strong methodological transparency.

Risk:

- May request broader benchmarks or stronger methodological novelty.

Probability:

- Medium

### ECML PKDD

Fit:

- Strong fit for empirical analysis + operational evaluation framing in classical ML ecosystem.

Risk:

- Novelty framing must be explicit to avoid “known thresholding behavior” dismissal.

Probability:

- Medium-High

### KDD Applied Data Science

Fit:

- Strong if framed around deployment diagnostics and decision-threshold control relevance.

Risk:

- Practical impact expectations may push for real-system case study or queue KPI simulation.

Probability:

- Medium

### AAAI Workshop Track

Fit:

- Very good for introducing framing and gathering cross-community feedback.

Risk:

- Less archival weight; feedback quality depends on workshop audience alignment.

Probability:

- High

### NeurIPS Workshop

Fit:

- Good for selective prediction/calibration/operational ML intersection framing.

Risk:

- If novelty is interpreted as only renaming coverage objects, reception may be mixed.

Probability:

- Medium-High

### Operational ML / MLOps-oriented venues

Fit:

- Strong conceptual fit for threshold-policy robustness and accessibility controllability narrative.

Risk:

- May require stronger production-KPI translation and case-study grounding.

Probability:

- Medium-High

### Domain-specific imbalance venues

Fit:

- Good fit when emphasizing severe-imbalance evaluation limitations and operational consequences.

Risk:

- Audience may still prioritize algorithmic novelty over evaluation framing.

Probability:

- Medium-High

# Recommended Path

# Recommended Submission Sequence

### Phase 1

Internal review

- Run a targeted pre-submission review with one reader each from imbalance, selective prediction, calibration, and operational ML.
- Goal: test wording resilience around overlap and anti-calibration misread risk.

### Phase 2

Workshop venue

- Submit to a NeurIPS/AAAI/ECML-aligned workshop with evaluation, trustworthy ML, or operational ML focus.
- Goal: stress-test paper identity, sharpen rebuttal language, and collect reviewer-language priors.

### Phase 3

Journal or conference main track

- Preferred next step: ECML PKDD-style conference route or ML journal route after minor reinforcement (calibration replication or additional dataset family).
- Journal-first route is viable if team prefers slower cycle and can add one reinforcement experiment pre-submission.

# Remaining Risks

### Scientific Risks

- Scope-limited external validity across datasets/model families: **Medium**.
- Calibration interaction replication depth (calibrator/split dependence): **Medium**.
- Mechanism interpretation for bagged HDDT unresolved: **Medium**.

### Positioning Risks

- “This is just threshold tuning/coverage rebranding” critique: **High**.
- “No algorithmic novelty” critique in method-centric venues: **Medium-High**.

### Terminology Risks

- Reachability interpreted as claiming wholly new mathematical object: **Medium**.
- Recurring-pattern language drifting into ontology/taxonomy overclaim: **Medium**.

### Evidence Risks

- Some claims rely on aggregate summaries; per-dataset clarity must stay visible: **Medium**.
- Operational implications without explicit queue simulation: **Medium**.

### Reviewer Risks

- Cross-community mismatch (different novelty priors): **High**.
- Calibration-community overread as anti-calibration: **Medium-High**.

# Missing Work Assessment

Would another major experiment materially improve the paper?

- Neural extensions (beyond sklearn-style MLP): **Helpful**.
  - Improves breadth, but not strictly required for current core framing.
- Bagged HDDT mechanism disambiguation: **Helpful**.
  - Valuable for depth, but not critical if framed as unresolved.
- Operational simulations (queue/KPI policy simulation): **Helpful to Critical** for applied venues; **Helpful** for research venues.
- Calibration replication (additional calibrators/splits): **Critical** for strongest reviewer resilience on Claim 13/14.
- Additional datasets (new severe-imbalance families): **Critical** for broader-scope venues/journals; **Helpful** for workshop launch.

# Elevator Pitch

### 50-word version

Under severe class imbalance, ranking and calibration metrics can look acceptable while minority accessibility collapses at operational thresholds. We propose a trajectory-centric analysis (reachability, elasticity localization, persistence/smoothness), show fixed-architecture MLP cliff-to-smooth transitions under imbalance-pressure perturbation, and document bounded calibration-accessibility tension, framed as empirical and conceptual—not universal theory.

### 150-word version

This paper asks a deployment-focused question: how should we reason about minority accessibility as decision thresholds move under severe class imbalance? Using repeated-split experiments, we show that strong AUROC/AP and improved ECE/Brier can coexist with poor threshold-mediated accessibility behavior. We analyze reachability trajectories (closely related to class-conditional coverage but used here as an operational object), plus elasticity localization and accessibility persistence/operational smoothness. In a controlled fixed-architecture MLP setting, objective/sampling perturbations (`mlp_bce` vs oversampled/weighted) produce a clear cliff-to-smoother transition, indicating morphology sensitivity to imbalance pressure rather than architecture alone. We also observe a bounded calibration interaction: reliability gains can coincide with degraded smoothness and sharper threshold jumps. The contribution is an empirical + conceptual framing for deployment-relevant evaluation under severe imbalance, with explicit non-claims (no new learner, no universal taxonomy, no anti-calibration general claim).

### Reviewer-facing version

This manuscript is best read as an evidence-bounded evaluation/framing contribution at the intersection of imbalance learning, selective prediction-style threshold analysis, calibration, and operational ML. We do not claim a new algorithm or a new foundational mathematical object. We show, in a constrained severe-imbalance setting, that ranking quality, reliability quality, and threshold-mediated minority accessibility can be partially non-equivalent. The core empirical anchors are (i) default-threshold collapse with low-threshold recovery, (ii) fixed-architecture MLP morphology transition under imbalance-pressure perturbation, and (iii) calibration-reliability gains coexisting with accessibility smoothness degradation. Reachability is positioned as an operational reinterpretation of class-conditional coverage trajectories, integrated with elasticity localization and persistence/smoothness diagnostics for threshold-policy controllability analysis.
