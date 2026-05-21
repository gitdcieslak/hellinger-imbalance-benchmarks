# Ranking Is Not Deployment:
## Operational Allocation Geometry Under Severe Class Imbalance

Optional subtitle: *A Modern Reassessment of Probability Allocation Behavior in Imbalanced Classification*

---

## Abstract (Working Draft)

Under severe class imbalance, ranking quality does not directly determine deployment behavior. We argue that the missing link is **posterior occupancy geometry**: how predicted positive-class probability mass is distributed relative to operational thresholds, especially for minority examples. We introduce **threshold reachability** as an operational accessibility function,
\[
R(t)=P(\hat p(x)\ge t\mid y=1),
\]
and use interval elasticity to characterize whether recall recovery is smooth, staged, or cliff-like as thresholds relax.

Across synthetic and modernized legacy HDDT benchmarks, we observe recurring occupancy signatures: quantized behavior (CART), cliff-like accessibility recovery (XGBoost), broader occupancy and smoother accessibility (HDDT/Bagged HDDT), and a less stable conservative pattern (LightGBM in this benchmark setting). Calibration improves reliability metrics in many cases, but often preserves occupancy-regime signatures, suggesting that accessibility geometry is not reducible to reliability correction alone. We treat regimes as empirical archetypes rather than fixed class identities and highlight dataset-conditional expression across boundary/cam/compustat/oil/satimage.

The central contribution is a mechanistic occupancy lens that explains when strong ranking fails to become deployable threshold behavior under severe imbalance.

---

## 1. Introduction

Severe imbalance is an operational allocation problem before it is a ranking problem. In fraud detection, SOC triage, medical screening, and rare-event industrial monitoring, deployed systems consume thresholded alerts under resource constraints. A model may rank well globally while still placing minority probability mass below actionable thresholds.

This manuscript reframes that failure mode using **posterior occupancy geometry**: the distribution of predicted probabilities and, crucially, minority mass accessibility across thresholds. Our claim is not only that AUROC/AP are insufficient. The stronger claim is that occupancy geometry explains when ranking is translated into action and when it is not.

To operationalize this view, we front-load three objects:
- **Posterior occupancy**: where model score mass resides in \([0,1]\), class-conditionally.
- **Threshold reachability**: minority survival above threshold, \(R(t)\), as deployable accessibility.
- **Operational trajectories**: threshold-indexed motion through precision-recall space, induced by reachability structure.

Across synthetic and legacy data, models with similar ranking can produce qualitatively different reachability landscapes: staircase-invariant, cliff-like, or broadly recoverable. These differences are directly deployment-relevant.

### Contributions (Revised)

1. We present **posterior occupancy geometry** as a formal evaluation lens for severe-imbalance deployment behavior.
2. We define and use **threshold reachability** \(R(t)\) as an operational accessibility function linking posterior mass to recall realization.
3. We provide **mechanistic regime interpretations** (quantized, cliff, broad, conservative) grounded in ECDFs, support concentration, and elasticity summaries.
4. We show **calibration persistence**: calibration can improve reliability while frequently preserving occupancy accessibility structure and regime signatures.
5. We characterize **dataset-conditional regime expression**, showing recurring signatures with nontrivial sensitivity to skew, boundary complexity, separability, and sample structure.

[Figure Placeholder: Posterior Occupancy Lens and Evaluation Pipeline]

---

## 2. Related Work (Initial Draft)

Imbalance research spans class weighting, resampling, cost-sensitive objectives, and tree/ensemble adaptations including Hellinger-based splitting. Ranking-centric evaluation under imbalance emphasizes AUROC/AP and PR curves. Calibration work emphasizes reliability correction and probability alignment.

Our gap is deployment translation: how posterior allocation structure governs threshold accessibility. This manuscript is an operational-geometry contribution, not a model-zoo superiority claim.

---

## 3. Benchmark Infrastructure

We use a reproducible pipeline combining synthetic skew scenarios and a curated legacy HDDT suite. Severe-imbalance analyses emphasize boundary, cam, compustat, oil, and satimage, with fixed threshold ladder \(0.50,0.25,0.10,0.05,0.01\) and repeated stratified evaluation.

Infrastructure includes dataset modernization, benchmark runners, threshold sweeps, occupancy figure generation, elasticity summaries, regime synthesis, calibration-interaction artifacts, and integrated synthesis reporting.

---

## 4. Experimental Design

### 4.1 Model families

Tree-centric families evaluated here are CART, HDDT, Bagged HDDT, Random Forest, XGBoost, and LightGBM. The scope is intentional and bounded.

### 4.2 Metrics and geometry descriptors

- Ranking: AUROC, average precision.
- Operational: precision, recall, F1, balanced accuracy, Brier.
- Occupancy: support size/discreteness, concentration, entropy-like spread, threshold persistence.

### 4.3 Reachability and elasticity

Minority reachability is:
\[
R(t)=P(\hat p(x)\ge t\mid y=1).
\]
It is the accessible minority mass above threshold and directly tracks recall potential under thresholding.

Threshold elasticity is treated as:
\[
E(t)=\left|\frac{dR(t)}{dt}\right|,
\]
estimated over threshold intervals. High localized elasticity indicates operational cliffs; low elasticity indicates threshold invariance.

### 4.4 Calibration interaction protocol

Raw, Platt, and isotonic outputs are compared with leakage-safe held-out calibration subsets per split. We then re-evaluate occupancy/reachability structure to separate reliability correction from accessibility redistribution.

---

## 5. Ranking Performance Under Imbalance

Ranking metrics remain informative but incomplete. In these benchmark conditions, strong AUROC/AP frequently coexists with weak default-threshold minority recall, reproducing the ranking-deployment gap observed in both the study journal and legacy threshold sweeps.

This section is setup, not endpoint. The explanatory core is occupancy and reachability mechanism.

[Figure Placeholder: AUROC/AP vs Default-Threshold Recall Divergence]

---

## 6. Threshold Dynamics Under Imbalance

Threshold sweeps expose operationally distinct responses:
- near-invariant response (CART),
- cliff recovery (often XGBoost),
- broader staged recovery (HDDT-family and some ensembles).

`reports/threshold_elasticity_summary.md` shows high XGBoost recall elasticity and large recall jumps on harder datasets (for example, boundary), while CART remains near-zero elasticity across the threshold ladder. These dynamics motivate regime archetypes but do not, by themselves, identify mechanism; mechanism is developed through occupancy geometry.

[Figure Placeholder: Elasticity Intervals and Jump Locations]

---

## 7. Posterior Occupancy Geometry (Core Section)

### 7.1 Occupancy phenomenon and operational meaning

**Phenomenon.** Models with similar ranking can distribute minority mass very differently relative to thresholds.

**Mechanistic interpretation.** Occupancy structure determines accessibility bottlenecks before a policy threshold is chosen.

**Evidence channel.** ECDFs, reachability curves, support histograms, concentration summaries.

**Deployment consequence.** Accessibility geometry determines whether threshold adjustment yields controllable recall recovery or brittle failure.

[Figure Placeholder: Occupancy Figure Family Overview]

### 7.2 Quantized allocators (CART)

**Phenomenon.** CART exhibits extreme discrete support in these runs (`unique_score_count_mean=2.0`, with very high mass concentration in `reports/posterior_occupancy_figures/posterior_occupancy_figure_summary.md`).

**Plausible mechanism.** Finite leaf posterior alphabets and empirical class-frequency outputs under severe skew create sparse threshold crossing points.

**Supporting evidence.** Staircase ECDFs, dominant support mass points, and near-zero elasticity in threshold summaries.

**Deployment consequence.** Threshold invariance over wide ranges: policy adjustments often produce little change until a discrete crossing is reached.

[Figure Placeholder: CART ECDF + Support Histogram + Reachability Staircase]

### 7.3 Compressed minority occupancy and cliff behavior (XGBoost)

**Phenomenon.** XGBoost can show many unique posterior values while still exhibiting poor minority accessibility at higher thresholds, with abrupt recovery only after aggressive threshold relaxation.

**Mechanistic conjecture (careful).** One plausible mechanism is additive logit-space scoring under severe-skew objective pressure, yielding strong ranking separation but compressed minority mass in low-probability regions. This interpretation is consistent with high recall elasticity and cliff-like reachability transitions; it is not presented as formal causal proof.

**Supporting evidence.** High mean recall elasticity and large jump behavior in `reports/threshold_elasticity_summary.md`, plus occupancy/reachability collapse patterns in the figure family.

**Deployment consequence.** Operational fragility: small threshold changes near cliff regions can induce large recall shifts and unstable queue load.

**Key scientific distinction.** **Support diversity is not accessibility.** Many unique posterior values are neither necessary nor sufficient for deployable minority accessibility. CART (few unique values) and XGBoost (many values) can both be operationally difficult for different geometric reasons.

[Figure Placeholder: XGBoost Reachability Collapse and Cliff Recovery]

### 7.4 Broad occupancy allocators (HDDT and Bagged HDDT)

**Phenomenon.** HDDT-family methods often exhibit broader threshold-accessible support and smoother minority reachability than quantized/cliff archetypes.

**Plausible mechanism.** The evidence is consistent with Hellinger-style split behavior preserving minority separation under skew, while bagging stabilizes occupancy without collapsing support diversity into a few dominant mass points.

**Supporting evidence.** Broad allocator labels in synthesis/regime tables (`reports/paper_operational_synthesis.md`, `reports/allocation_regime_summary.md`) and occupancy summaries with less extreme concentration than CART.

**Deployment consequence.** Wider controllability bands under threshold relaxation, often with less catastrophic accessibility collapse.

[Figure Placeholder: HDDT vs Bagged HDDT Broad Occupancy Comparison]

### 7.5 Conservative occupancy (LightGBM) as preliminary archetype

In this benchmark setting, LightGBM is frequently tagged conservative in regime synthesis. However, this archetype is less stable than the CART/XGBoost/HDDT signatures and should be treated as preliminary. Evidence supports conservative tendencies under severe skew in several datasets, but cross-dataset variability warrants caution.

[Figure Placeholder: LightGBM Conservative Occupancy (Preliminary)]

### 7.6 Threshold reachability to operational trajectories

Reachability curves are not only descriptive; they induce threshold-parameterized deployment trajectories. Because recall at threshold is constrained by minority reachability, the shape of \(R(t)\) determines movement in PR space:
- staircase \(R(t)\) produces piecewise-constant PR movement,
- cliff \(R(t)\) produces abrupt trajectory bends,
- broad \(R(t)\) produces smoother policy paths.

This provides a direct bridge from posterior occupancy geometry to threshold controllability and operational fragility.

[Figure Placeholder: Reachability-to-PR Trajectory Mapping]

### 7.7 Calibration persistence: reliability vs accessibility geometry

Calibration interaction artifacts show that reliability can improve while operational geometry often persists. In integrated synthesis (`reports/paper_operational_synthesis.md`), calibration method changes ECE/Brier by model, but major regime signatures recur.

Interpretation: calibration can rescale probabilities and improve numerical correctness, but it does not necessarily redistribute minority mass into threshold-accessible regions. Reliability correction and accessibility correction are therefore distinct objectives.

[Figure Placeholder: Pre/Post Calibration Reachability Overlay]

---

## 8. Occupancy Regimes as Dataset-Conditional Archetypes

We treat regimes as recurring operational signatures rather than immutable properties of model families. Regime expression is dataset-sensitive and can vary with skew ratio, minority geometry, feature separability, sample size, boundary complexity, and regularization.

In this benchmark slice, boundary/cam/compustat often exhibit the most pathological accessibility behaviors; oil appears transitional in several summaries; satimage is comparatively healthier for several models. These differences reinforce that regimes are explanatory archetypes conditioned on data and operating context.

From `reports/allocation_regime_summary.md`, current synthesis remains useful as a compact map:
- quantized: CART (stable signature),
- cliff: XGBoost (frequent signature under severe skew),
- broad: HDDT/Bagged HDDT/Random Forest (recurring in current datasets),
- conservative: LightGBM (provisional in this draft).

[Figure Placeholder: Regime Scatterplot and Dataset Faceting]

---

## 9. Discussion

The main practical implication is that deployment readiness under severe imbalance is governed by accessibility geometry, not ranking alone. A high-ranking model may still allocate minority mass below feasible action thresholds, producing fragile operations even with strong aggregate metrics.

For operational systems with constrained review bandwidth, occupancy-aware evaluation changes model selection criteria: one should explicitly assess reachability shape, support concentration, and threshold controllability.

This reframing does not claim universal superiority for any family. It argues for a mechanistic deployment lens.

---

## 10. Limitations and Future Work

Limitations in this draft:
- binary classification only
- no neural tabular families
- finite dataset scope and no temporal drift modeling
- heuristic regime rules and evolving occupancy metrics
- no explicit queue-coupled policy simulation

Future directions:
1. occupancy derivatives and finer threshold-continuation analysis
2. multiclass and temporal imbalance settings
3. neural occupancy geometry comparisons
4. queue-aware threshold policy optimization over reachability surfaces
5. stress-testing regime persistence under domain shift

---

## 11. Conclusion

This revision centers a stronger thesis: posterior occupancy geometry explains when ranking does and does not become deployable threshold behavior under severe imbalance. Threshold reachability provides the accessibility bridge; elasticity and trajectories characterize controllability; calibration analysis shows that reliability improvement is not always accessibility improvement.

Under these benchmark conditions, recurring archetypes (quantized, cliff, broad, conservative) help explain deployment behavior, provided they are treated as dataset-conditional signatures rather than fixed universal identities.

---

## Draft Notes for Next Iteration

- tighten quantitative references with confidence intervals and split-level variability
- add panel-level citations once figure numbering stabilizes
- expand related work and citation grounding
- test alternative regime decision rules for sensitivity
- evaluate robustness of conservative archetype assignment for LightGBM
