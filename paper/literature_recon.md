# Literature Reconnaissance: Operational Accessibility Manuscript

## Scope and Intent

This document maps the manuscript's closest conceptual neighbors and reviewer interpretation risks. It is not a broad bibliography dump; it is a positioning artifact for likely reviewer priors.

Primary manuscript anchor: `paper/manuscript_v0_1.md`.

---

## 1) Reachability Recon (Highest Priority)

### Candidate Concept A

Concept: **Class-conditional threshold coverage / survival function**

Closest papers:

- Chow (1970) reject-option foundations.
- El-Yaniv and Wiener (2010) selective classification formalism.
- Geifman and El-Yaniv (2017) modern selective deep classification.

Similarity:

- The manuscript object `R(t)=P(\hat p(x)\ge t\mid y=1)` is mathematically a class-conditional survival/coverage trajectory over thresholds.

Difference:

- Manuscript emphasis is minority-conditioned operational accessibility under severe imbalance, with trajectory diagnostics (jump localization, smoothness, persistence), not only accept/reject risk guarantees.

Potential reviewer interpretation:

- "This is class-conditional coverage with new language."

Risk of accidental reinvention:

- **High** at the object-definition level; lower at the operational synthesis layer.

### Candidate Concept B

Concept: **Selective prediction risk-coverage curves**

Closest papers:

- El-Yaniv and Wiener (2010).
- Geifman and El-Yaniv (2017).
- Follow-on selective prediction calibration/risk papers (modern variants).

Similarity:

- Threshold controls effective action set; trajectory shape across thresholds matters.

Difference:

- Selective prediction usually optimizes global risk/coverage trade-offs; manuscript centers minority accessibility persistence under skew and policy controllability.

Potential reviewer interpretation:

- "Why not report standard risk-coverage directly?"

Risk of accidental reinvention:

- **High** if overlap is not explicitly acknowledged in Related Work.

### Candidate Concept C

Concept: **Reject-option / abstention classification**

Closest papers:

- Chow (1970).
- Classical reject-option decision rules.

Similarity:

- Shared threshold-mediated decision gate.

Difference:

- Manuscript is less about abstain decision policy design and more about minority accessibility dynamics as thresholds move operationally.

Potential reviewer interpretation:

- "This is abstention analysis without explicit abstain utility modeling."

Risk of accidental reinvention:

- **Medium-High**.

### Reachability Recon Summary

- Is reachability already formalized? **Yes, adjacent formal objects are established.**
- Closest mathematical objects: class-conditional coverage/survival trajectories, selective risk-coverage curves.
- Terminology overlap: coverage, acceptance set, reject option, abstention threshold.
- Distinction that remains: minority-conditioned deployment accessibility + smoothness/jump/persistence bundle under severe imbalance.

Key Papers

Foundational:

- Chow (1970)
- El-Yaniv and Wiener (2010)

Important modern:

- Geifman and El-Yaniv (2017)
- Contemporary selective prediction/risk-control literature

Why relevant:

- These are the most likely reviewer priors for `R(t)`-style trajectory claims.

Potential citation priority:

- **High**

---

## 2) Calibration Recon

Questions addressed:

- Have others observed reliability improvement with worse downstream deployment behavior?
- Does literature assume calibration improves threshold control?

Findings:

- Calibration literature (Platt 1999; Zadrozny and Elkan 2002; Niculescu-Mizil and Caruana 2005; Guo et al. 2017) is strong on probabilistic reliability, but less unified on minority operational accessibility smoothness under severe imbalance.
- Decision-focused discussions often imply better reliability should help downstream policy, but this is typically not evaluated as minority accessibility trajectory controllability.
- Partial overlap exists with decision calibration/utility literature; direct overlap with "reliability improves while threshold smoothness worsens" appears limited and context-dependent.

Direct overlap:

- Reliability diagnostics and post-hoc calibration methods.

Partial overlap:

- Calibration-for-decision and thresholding work.

Novelty opportunities (conservative):

- Setting-specific empirical tension framing between reliability and accessibility smoothness/jump behavior under severe imbalance.

Key Papers

Foundational:

- Platt (1999)
- Zadrozny and Elkan (2002)
- Niculescu-Mizil and Caruana (2005)

Important modern:

- Guo et al. (2017)

Why relevant:

- Core priors for claims about calibration interaction.

Potential citation priority:

- **High**

---

## 3) Decision-Theoretic Classification Recon

Questions addressed:

- How close is operational accessibility to decision-theoretic threshold concepts?

Findings:

- Very close in deployment motivation (threshold choice under asymmetric cost/risk).
- Established: threshold policy as utility decision variable (Elkan 2001; Fawcett 2006; cost-curve traditions incl. Drummond and Holte 2006).
- Manuscript extension: shift from selecting one threshold to analyzing trajectory controllability as policy changes.

What is new vs established:

- Established: utility-sensitive threshold optimization.
- Distinct manuscript emphasis: threshold-evolution accessibility persistence/smoothness diagnostics for minority actionability.

Key Papers

Foundational:

- Elkan (2001)
- Fawcett (2006)

Important modern:

- Decision-analytic thresholding and cost-sensitive deployment papers

Why relevant:

- Reviewers may ask for explicit utility models; this literature sets that expectation.

Potential citation priority:

- **High**

---

## 4) Operational ML Recon

Questions addressed:

- Are there operational metrics similar to accessibility persistence?
- How are threshold-policy effects studied?

Findings:

- Operational ML literature emphasizes maintenance, monitoring, policy drift, and human workflow constraints rather than explicit minority accessibility persistence metrics.
- Relevant neighbors include technical debt/readiness frameworks (Sculley et al. 2015; Breck et al. 2017) and human-AI workflow guidance (Amershi et al. 2019).
- Threshold policy management in triage/review contexts exists, but often with domain-specific KPIs rather than a generalized trajectory morphology vocabulary.

Where this paper fits:

- Natural fit as a deployment-interpretation paper introducing a compact threshold-trajectory language for imbalance settings.

Key Papers

Foundational:

- Sculley et al. (2015)

Important modern:

- Breck et al. (2017)
- Amershi et al. (2019)

Why relevant:

- Supports positioning beyond pure metric paper toward operations-aware evaluation.

Potential citation priority:

- **Medium-High**

---

## 5) Score Distribution / Margin Geometry Recon

Questions addressed:

- Are support breadth and occupancy persistence already studied?
- Could morphology language conflict with existing terminology?

Findings:

- Related traditions exist: margin distributions, confidence score distributions, calibration error decomposition, uncertainty concentration.
- Direct "occupancy persistence" terminology appears non-standard; likely interpreted as a derived distributional stability descriptor rather than a canonical metric family.
- Morphology vocabulary (cliff/smooth/broad/conservative) may collide with informal terms in score-shape discussions, so repeated "provisional empirical shorthand" caveat is important.

Closest existing ideas:

- Margin/confidence distribution analysis.
- Score concentration/sparsity diagnostics.

Key Papers

Foundational:

- Margin/distribution-oriented classifier analysis traditions.

Important modern:

- Probability-distribution diagnostics in calibration/uncertainty work.

Why relevant:

- Prevents overclaim that occupancy constructs are entirely new object classes.

Potential citation priority:

- **Medium**

---

# Reviewer Risk Matrix

| Literature | Risk Level | Why |
| --- | --- | --- |
| Selective Prediction / Coverage | High | Reachability is mathematically close to class-conditional coverage trajectories. |
| Calibration | High | Calibration-tension claims invite scrutiny for setup-dependence and anti-calibration misread. |
| Decision-Theoretic Classification | Medium-High | Reviewers may expect explicit utility formulations if threshold claims are strong. |
| Severe Imbalance Learning | Medium | Reviewers may expect learner novelty instead of framing novelty. |
| Operational ML | Medium | Strong fit, but reviewers may ask for clearer KPI linkage and deployment examples. |
| Score/Margin Geometry | Medium | Terminology could be read as rebranding known confidence-distribution analysis. |

---

# Potential Novel Contributions (Conservative)

- Current evidence suggests a setting-specific reliability-accessibility tension under severe imbalance (reliability gains can coincide with lower operational smoothness / higher jump intensity).
- Fixed-architecture morphology transitions under objective/sampling perturbation appear distinct as a controlled empirical result in this pipeline.
- Family-level non-equivalence examples (e.g., near-neighbor ranking with divergent accessibility geometry) appear strong as deployment-facing evidence.

# Potential Reframing Contributions

- Reachability trajectories as central operational accessibility framing.
- Accessibility morphology vocabulary as provisional synthesis shorthand.
- Joint interpretation of ranking, reliability, and accessibility as related but partially non-equivalent deployment axes.

---

## Recommended Positioning Posture

- Treat object-level reachability novelty as **low**; treat empirical synthesis novelty as **moderate**.
- Lead with overlap acknowledgment in selective prediction/coverage to preempt reinvention critiques.
- Emphasize bounded claims: "current evidence suggests," "in this constrained severe-imbalance setting," and "requires broader replication." 
