# Related Work Expansion Draft (Positioning-Oriented)

This is a manuscript-oriented synthesis artifact, not final prose.

## Severe Class Imbalance Learning

### What reviewers will know

- Cost-sensitive learning, sampling/reweighting, and specialized imbalance learners are established.
- HDDT lineage is recognized as imbalance-oriented tree design.

### How our manuscript relates

- Uses established imbalance tools and baselines.
- Focuses on threshold-mediated minority accessibility trajectories rather than proposing a new learner.

### What we should cite

- Elkan (2001), Chawla et al. (2002), He and Garcia (2009), Krawczyk (2016), Buda et al. (2018), Johnson and Khoshgoftaar (2019), Cieslak and Chawla (2008), Cieslak et al. (2012).

### What we should avoid claiming

- Avoid implying algorithmic novelty in imbalance learning.
- Avoid implying universal taxonomy across all model families.

### Suggested manuscript language

"Our setup is directly within established severe-imbalance learning practice (weighting, oversampling, and imbalance-aware baselines). We do not propose a new learner; we analyze threshold-mediated minority accessibility behavior under deployment-relevant policy variation."

---

## Selective Prediction / Coverage / Reject Option

### What reviewers will know

- Coverage-risk trajectories, reject-option, and abstention are mature topics.
- Thresholded confidence sets are standard formal objects.

### How our manuscript relates

- Reachability is closely related to class-conditional coverage trajectories.
- Manuscript emphasis is minority-conditioned accessibility persistence/smoothness/jump localization under severe imbalance.

### What we should cite

- Chow (1970), El-Yaniv and Wiener (2010), Geifman and El-Yaniv (2017).

### What we should avoid claiming

- Avoid "new mathematical object" claims for reachability.
- Avoid dismissing risk-coverage traditions.

### Suggested manuscript language

"Reachability is closely related to class-conditional threshold coverage. Our contribution is an operational reinterpretation for minority accessibility under severe imbalance, coupled with threshold-sensitivity diagnostics (persistence, smoothness, and jump localization)."

---

## Calibration, Decision Analysis, and Operational ML

### What reviewers will know

- Calibration methods and reliability metrics are established and valuable.
- Decision theory treats threshold choice under asymmetric costs.
- Operational ML emphasizes deployment risk, policy drift, and human workflow integration.

### How our manuscript relates

- Uses standard calibration methods/metrics.
- Does not challenge calibration value.
- Investigates a bounded interaction: reliability improvements can coexist with less controllable accessibility trajectories in severe imbalance.

### What we should cite

- Platt (1999), Zadrozny and Elkan (2002), Niculescu-Mizil and Caruana (2005), Guo et al. (2017).
- Elkan (2001), Fawcett (2006), Drummond and Holte (2006).
- Sculley et al. (2015), Breck et al. (2017), Amershi et al. (2019).

### What we should avoid claiming

- Avoid anti-calibration rhetoric.
- Avoid implying utility-theory replacement.
- Avoid broad operational generalization beyond current slice.

### Suggested manuscript language

"This work is complementary to calibration and decision-theoretic threshold analysis. We retain standard reliability framing while showing that, in this constrained severe-imbalance setting, reliability gains do not always imply smoother minority accessibility control as thresholds shift."

---

## Classifier Evaluation Under Imbalance

### What reviewers will know

- AUROC/AP and thresholded confusion metrics are standard reporting anchors.

### How our manuscript relates

- Agrees these metrics are useful.
- Shows they can underdescribe threshold-mediated accessibility behavior in deployment-like settings.

### What we should cite

- Davis and Goadrich (2006), Saito and Rehmsmeier (2015), Fawcett (2006), Drummond and Holte (2006).

### What we should avoid claiming

- Avoid "AUROC/AP are invalid" framing.
- Avoid one-threshold normative conclusions.

### Suggested manuscript language

"We do not dispute AUROC/AP utility for discrimination. Our narrower claim is that these summaries can be insufficient for threshold-mediated minority accessibility interpretation under severe imbalance."

---

## Score Distribution / Occupancy Geometry

### What reviewers will know

- Confidence/margin/score-distribution analysis is established.

### How our manuscript relates

- Occupancy/support/persistence metrics are used as operational descriptors.
- Framework is empirical and provisional, not a complete geometric theory.

### What we should cite

- Calibration/uncertainty score-distribution references adjacent to manuscript metrics.

### What we should avoid claiming

- Avoid formal geometry-theory novelty claims.
- Avoid presenting morphology labels as fixed ontology.

### Suggested manuscript language

"Occupancy and support descriptors are used here as practical diagnostics for threshold-mediated accessibility behavior; they are not presented as a complete geometric formalism."

---

# Reviewer Risk Matrix

| Literature | Risk Level | Why |
| --- | --- | --- |
| Selective Prediction | High | Reachability may be read as class-conditional coverage under new naming. |
| Calibration | High | Tension claims can be overread as anti-calibration without careful wording. |
| Decision-Theoretic | Medium-High | Threshold-policy claims can trigger requests for explicit utility formalization. |
| Imbalance Learning | Medium | Reviewers may expect learner novelty if framing is unclear. |
| Operational ML | Medium | Good fit, but reviewers may ask for explicit deployment KPI linkage. |

---

# Potential Novel Contributions (Conservative)

- Current evidence suggests a distinct calibration-accessibility interaction in this severe-imbalance protocol.
- Controlled fixed-architecture morphology transitions under objective/sampling perturbation appear strong within current scope.
- Family-level examples of ranking-accessibility non-equivalence appear operationally salient.

# Potential Reframing Contributions

- Reachability-centered operational accessibility framing.
- Morphology vocabulary as provisional empirical shorthand.
- Joint axis framing: ranking, reliability, and accessibility as partially non-equivalent deployment views.
