# Executive Summary

This manuscript appears to sit at the intersection of:

- severe class imbalance learning,
- classifier evaluation under threshold-mediated deployment,
- probability calibration,
- and operational ML decision practice.

The closest adjacent literatures appear to be:

- imbalance-focused modeling and evaluation traditions,
- threshold/utility-sensitive decision analysis,
- calibration and reliability assessment,
- selective prediction / risk-coverage analysis,
- and deployment-oriented ML monitoring/operations work.

The manuscript’s most distinctive contribution appears to be:

- reframing minority accessibility as a trajectory object (reachability under threshold evolution),
- empirically separating ranking quality, reliability quality, and operational accessibility behavior,
- and documenting a calibration tension (ECE/Brier improvements coexisting with accessibility smoothness degradation) in a constrained severe-imbalance setting.

What is familiar:

- severe imbalance metrics, threshold sweeps, calibration methods, and model-family comparisons.

What appears less familiar in this combination:

- explicit trajectory-centric operational accessibility framing,
- morphology-transition analysis under fixed architecture perturbation,
- and reliability-vs-accessibility non-equivalence framing.

What reviewers may initially misunderstand:

- They may read the work as a benchmark paper, taxonomy proposal, or anti-calibration argument.
- The manuscript should instead be positioned as an empirical + conceptual framing study with explicit non-claims.

---

## Literature Family: Severe Class Imbalance Learning

### What this community studies

This community studies learning under skewed class priors, with emphasis on rare-event detection performance and mitigation via sampling, weighting, specialized split criteria, and cost-sensitive objectives.

### Why this manuscript is relevant

The manuscript directly uses severe-imbalance datasets and compares behavior under objective/sampling perturbations (BCE vs oversampled vs weighted MLP) while preserving fixed architecture.

### Where this manuscript agrees

- Imbalance pressure materially affects classifier behavior.
- Sampling/weighting changes minority handling properties.

### Where this manuscript extends or reframes

- Extends focus from static endpoint metrics to threshold-evolution accessibility trajectories.
- Reframes perturbation effects as operational morphology transition rather than only performance optimization.

### Reviewer expectations likely to emerge

- Expectation of broader model zoo and stronger algorithmic novelty.
- Expectation that “improvement” be framed as AUROC/AP gains.

### Citation targets to investigate

- HDDT lineage and Hellinger-based imbalance work.
- Cost-sensitive and class-weighted learning for rare events.
- Oversampling traditions and critiques (e.g., distribution distortion).

---

## Literature Family: Classifier Evaluation Under Imbalance

### What this community studies

Evaluation protocols emphasizing AUROC, AP/PR, thresholded confusion metrics, and model ranking comparisons.

### Why this manuscript is relevant

The manuscript challenges an implicit assumption that strong ranking metrics are sufficient proxies for operational accessibility under threshold policies.

### Where this manuscript agrees

- AUROC/AP remain useful for ranking discrimination.
- PR/threshold analyses are established tools.

### Where this manuscript extends or reframes

- Highlights trajectory-level non-equivalence: same or acceptable ranking can coexist with default-threshold collapse.
- Introduces reachability trajectory interpretation as an organizing lens.

### Reviewer expectations likely to emerge

- “Is this just threshold tuning?”
- “Is reachability just rebranded recall?”

### Citation targets to investigate

- Threshold selection methodology under class imbalance.
- Critiques of AUROC/AP sufficiency in operational settings.
- Evaluation frameworks separating discrimination and actionability.

---

## Literature Family: Probability Calibration

### What this community studies

Probability reliability, post-hoc calibration methods (Platt/isotonic), and diagnostics (ECE, Brier, reliability diagrams).

### Why this manuscript is relevant

Calibration analysis is central to this manuscript’s strongest tension finding: reliability gains can coincide with degraded operational smoothness.

### Where this manuscript agrees

- Calibration is valuable and often improves reliability metrics.

### Where this manuscript extends or reframes

- Reframes calibration outcomes through accessibility trajectories, not reliability alone.
- Emphasizes partial non-equivalence between calibrated correctness and threshold controllability.

### Reviewer expectations likely to emerge

- “Is this anti-calibration rhetoric?”
- “Are these effects calibrator-specific artifacts?”

### Citation targets to investigate

- Foundational calibration papers.
- Work discussing downstream decision impact of calibration.
- Any literature on calibration effects over threshold trajectories (if available).

---

## Literature Family: Decision-Theoretic Classification

### What this community studies

Decision rules under asymmetric costs, threshold selection under utility functions, and risk-sensitive policy choices.

### Why this manuscript is relevant

The paper’s threshold-mediated deployment framing aligns naturally with utility-sensitive decision analysis.

### Where this manuscript agrees

- Threshold choice is policy-dependent and operationally meaningful.

### Where this manuscript extends or reframes

- Moves from static threshold choice to trajectory robustness under threshold variation.
- Emphasizes accessibility persistence as an operational property.

### Reviewer expectations likely to emerge

- “Where is explicit utility modeling?”
- “How is reachability different from standard decision-theoretic risk curves?”

### Citation targets to investigate

- Cost-sensitive threshold optimization frameworks.
- Decision-analytic classification under class imbalance.
- Robust threshold policy literature.

---

## Literature Family: Selective Prediction / Risk-Coverage

### What this community studies

Abstention/reject-option systems, coverage-risk tradeoffs, confidence-threshold mechanisms.

### Why this manuscript is relevant

Reachability resembles coverage-style trajectory analysis over threshold changes, especially in how accessibility mass survives stricter policies.

### Where this manuscript agrees

- Thresholded confidence controls effective action set size.
- Tradeoff trajectories can be more informative than single points.

### Where this manuscript extends or reframes

- Condition is minority accessibility (`y=1`) rather than aggregate acceptance coverage.
- Focus is imbalance-specific operational accessibility rather than generic abstention quality.

### Reviewer expectations likely to emerge

- “Is reachability equivalent to class-conditional coverage?”
- “Why not directly use risk-coverage formalism?”

### Citation targets to investigate

- Selective classification and reject option literature.
- Coverage curve definitions and decision guarantees.
- Class-conditional coverage work relevant to imbalance.

---

## Literature Family: Operational Machine Learning

### What this community studies

Deployment behavior, monitoring, queue constraints, human-in-the-loop workflows, and operational reliability.

### Why this manuscript is relevant

The manuscript is explicitly threshold-mediated and deployment-oriented; accessibility trajectories map naturally to practical queue and review-capacity concerns.

### Where this manuscript agrees

- Model usefulness depends on deployment behavior, not only offline score summaries.

### Where this manuscript extends or reframes

- Provides a concrete trajectory language (reachability/smoothness/elasticity localization) for imbalance deployment behavior.

### Reviewer expectations likely to emerge

- “Can this connect to concrete operational KPIs?”
- “Is this too abstract for applied teams?”

### Citation targets to investigate

- Human-in-the-loop triage/fraud operations papers.
- Queue-aware decision thresholding in ML systems.
- Monitoring frameworks for threshold policy drift.

---

## Literature Family: Explainable / Interpretable Evaluation

### What this community studies

Interpretable metrics, model behavior visualization, and stakeholder-facing evaluation abstractions.

### Why this manuscript is relevant

The paper introduces a perspective shift: from static endpoint metrics to trajectory-centric accessibility interpretation.

### Where this manuscript agrees

- Visual and interpretable summaries are essential for policy-facing model decisions.

### Where this manuscript extends or reframes

- Positions reachability less as a single new metric and more as a perspective combining threshold evolution, elasticity, and occupancy persistence.

### Reviewer expectations likely to emerge

- “Is reachability a metric, visualization, or framework?”
- “What is minimally required for practical adoption?”

### Citation targets to investigate

- Interpretable evaluation frameworks.
- Threshold response visualization traditions.
- Reliability-vs-actionability interpretability papers.

---

# Claim-to-Literature Mapping

| Claim | Relevant Literature | Relationship |
|---|---|---|
| ranking quality != operational accessibility | classifier evaluation under imbalance | extends |
| AUROC/AP can obscure default-threshold collapse | metric interpretation literature | challenges assumptions |
| reachability trajectories as primary object | threshold analysis, selective prediction | reframes |
| recurring morphology patterns | imbalance behavior characterization | operationalizes |
| CART-like quantized behavior | tree probability/discretization literature | aligns + contextualizes |
| XGBoost cliff-like collapse (and LGBM distinction) | boosted imbalance analysis | refines family-level assumptions |
| HDDT/bagged HDDT broad persistence pattern | imbalance tree methods | extends with trajectory framing |
| baseline MLP analyzable in same pipeline | cross-family evaluation methodology | extends |
| BCE MLP cliff behavior | neural imbalance evaluation | aligns with conservative-allocation concerns |
| oversampling/weighting induce cliff->smooth transitions | cost-sensitive/sampling literature | extends with morphology transition lens |
| morphology not architecture-only | optimization-sensitive behavior literature | supports joint-emergence framing |
| smoothness not reducible to support breadth | score-distribution analysis | refines explanatory assumptions |
| calibration can improve ECE/Brier while worsening accessibility morphology | calibration literature | challenges single-axis reliability interpretation |
| calibration quality vs accessibility smoothness non-equivalence | calibration + decision-theoretic intersections | extends/reframes |
| taxonomy is provisional empirical shorthand | interpretability/evaluation frameworks | bounds overclaiming |
| paper is empirical + conceptual framing, not complete theory | methodology-positioning literature | clarifies scope |
| Bagged HDDT mechanism unresolved | ensemble mechanism literature | open question |
| reachability-aware objectives as cautious future direction | objective design under imbalance | motivates future work |

---

# Reviewer Simulation

## Reviewer Type: Imbalance Researcher

Potential reaction:
- “Interesting, but this feels like known class-weighting/sampling behavior.”

Likely criticism:
- “Where is the algorithmic novelty?”
- “Is this just another imbalance benchmark?”

Suggested manuscript response:
- Emphasize contribution type: trajectory-centric empirical + conceptual framing, not new learner.
- Keep fixed-architecture perturbation argument central.

## Reviewer Type: Calibration Researcher

Potential reaction:
- “Calibration is being interpreted too negatively.”

Likely criticism:
- “Are smoothness degradations just method/split artifacts?”

Suggested manuscript response:
- Explicitly state calibration remains valuable.
- Frame result as partial non-equivalence under severe imbalance and threshold-mediated policy, not anti-calibration claim.

## Reviewer Type: Applied ML Researcher

Potential reaction:
- “This is operationally plausible, but can I map it to deployment knobs?”

Likely criticism:
- “Too conceptual; unclear operational KPI linkage.”

Suggested manuscript response:
- Strengthen queue/triage interpretation and include threshold-policy examples.
- Keep figures tied to actionable threshold sensitivity insights.

## Reviewer Type: Statistical Learning Researcher

Potential reaction:
- “Interesting decomposition, but formal guarantees are missing.”

Likely criticism:
- “Taxonomy appears heuristic/arbitrary.”

Suggested manuscript response:
- Proactively bound claims and mark taxonomy as provisional shorthand.
- Position formalization as future work and focus on robust empirical regularities.

---

# What This Paper Is

- An empirical + conceptual framing manuscript.
- A threshold-aware analysis of operational accessibility trajectories under severe imbalance.
- A study of partial non-equivalence among ranking quality, calibration quality, and accessibility morphology.
- A constrained fixed-architecture perturbation analysis of morphology transition.

# What This Paper Is Not

- Not a complete operational morphology theory.
- Not a universal classifier taxonomy.
- Not a new production learner or objective validated at scale.
- Not a leaderboard benchmark competition.
- Not a definitive causal explanation of Bagged HDDT mechanism.

---

# Open Literature Questions

- Is reachability already formalized under another name in class-conditional coverage traditions?
- What is the closest mathematical relative of the reachability trajectory/elasticity decomposition?
- Are there calibration papers explicitly discussing threshold smoothness or accessibility persistence degradation?
- What queue-theoretic deployment literatures best map threshold-policy sensitivity to operational load?
- Are there medical triage/fraud-review studies that separate probability reliability from intervention accessibility?
- How should selective prediction and reject-option frameworks be mapped to minority-conditioned reachability?
- Is there prior work on “morphology transitions” under fixed architecture with objective perturbations?
- Which literature best supports distinctions among support breadth, occupancy persistence, and smoothness?
- Where does HDDT/bagged-HDDT mechanistic interpretation fit relative to ensemble stability literature?

---

## Notes on Inputs and Gaps

- `research/claim_audit_report.md` was used as the primary audit source.
- `claim_inventory(1).md` was not found in the current workspace; this document assumes the revised inventory in `research/claim_inventory.md` is authoritative. If `claim_inventory(1).md` exists in another worktree, merge deltas before final Related Work drafting.
