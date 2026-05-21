# Ranking Is Not Deployment:
## Operational Allocation Geometry Under Severe Class Imbalance

Optional subtitle: *A Modern Reassessment of Probability Allocation Behavior in Imbalanced Classification*

---

## 1. Introduction

Severe class imbalance is a deployment problem before it is a ranking problem. In fraud detection, security operations, medical triage, and rare-event industrial monitoring, decision systems are governed by thresholds, review capacity, and false-positive tolerance. These systems do not consume AUROC directly; they consume a thresholded allocation of attention.

This paper starts from an empirical discrepancy repeatedly observed across synthetic and legacy datasets: models with strong ranking metrics can still produce near-failure minority recall at operational thresholds, while other models with weaker ranking can remain operationally usable. We refer to this discrepancy as the ranking-deployment gap.

Our working hypothesis is that classifier families induce distinct posterior occupancy geometries under severe imbalance. By posterior occupancy geometry, we mean how predicted positive-class probabilities are distributed over the test population and, critically, how minority mass is positioned relative to operational thresholds. This geometric structure governs threshold accessibility, recall recovery, and operational fragility.

The central claim of this draft is intentionally narrow: ranking quality and deployment behavior are related but non-equivalent properties under severe imbalance. The manuscript develops this claim using threshold dynamics, posterior occupancy figures, calibration interaction analysis, and regime synthesis.

**Contributions in this first draft context.**
1. A reproducible benchmark pipeline spanning synthetic skew scenarios and modernized legacy HDDT datasets.
2. A threshold-sweep protocol that exposes deployment dynamics across fixed thresholds (0.50, 0.25, 0.10, 0.05, 0.01).
3. Occupancy-oriented diagnostics (support discreteness, concentration, reachability) that characterize posterior allocation structure.
4. A calibration interaction analysis showing frequent persistence of operational geometry after Platt and isotonic correction.
5. A preliminary taxonomy of operational allocation regimes (quantized, cliff, broad, conservative).

[Figure Placeholder: Project Overview and Analysis Flow]

---

## 2. Related Work (Initial Draft)

Work on imbalanced learning includes class weighting, resampling, specialized split criteria (including Hellinger-based trees), and ensemble methods such as random forests and gradient boosting. Parallel literature evaluates ranking quality under imbalance using AUROC, average precision, and PR curves. Calibration literature focuses on reliability, Brier/ECE improvements, and better probability alignment.

What is less developed is a deployment-centered account of how posterior mass is allocated under severe skew and how that allocation controls threshold accessibility. Existing evaluations often stop at ranking and calibration quality, while operational behavior depends on survival of minority predictions above realistic thresholds.

This manuscript aims to bridge that gap by treating posterior allocation geometry as a first-class object and connecting it to threshold dynamics and deployment controllability.

---

## 3. Benchmark Infrastructure

The infrastructure combines synthetic imbalance generation with a curated legacy HDDT dataset suite. Synthetic scenarios support controlled skew and repeated splits; legacy datasets provide real-data stress tests (including boundary, cam, compustat, oil, satimage in the severe subset emphasized here).

The pipeline includes reproducible ingestion, profiling, benchmark runners, threshold sweeps, occupancy analyses, elasticity summaries, regime synthesis, calibration interaction artifacts, and paper-facing figure/report generation. Artifact snapshotting and structured reports are used to keep analysis traceable and iteration-friendly.

Threshold analyses use a fixed operational ladder (0.50, 0.25, 0.10, 0.05, 0.01) with repeated stratified splits (5x2-style protocol) to stabilize estimates while preserving interpretability.

---

## 4. Experimental Design

### 4.1 Model families

We focus on tree-centric families used in current infrastructure:
- CART
- HDDT
- Bagged HDDT
- Random Forest
- XGBoost
- LightGBM

The manuscript does not claim full model-zoo coverage. The purpose is to compare representative allocation behaviors under severe imbalance.

### 4.2 Metrics and operational objects

Ranking metrics include AUROC and average precision. Operational metrics include precision, recall, F1, balanced accuracy, and Brier score across fixed thresholds.

Allocation/occupancy summaries include entropy-like spread, support size, concentration, discrete support diagnostics, and threshold-occupancy persistence.

### 4.3 Reachability and elasticity formalism

We define minority threshold reachability:

\[
R(t) = P(\hat p(x) \ge t \mid y=1)
\]

which is the minority survival function over thresholds. This quantity directly tracks accessible recall potential at threshold \(t\).

We define threshold elasticity informally as:

\[
E(t) = \left|\frac{dR(t)}{dt}\right|
\]

and estimate it over threshold intervals from the fixed ladder. Large localized elasticity indicates operational cliffs; near-zero elasticity indicates threshold invariance.

### 4.4 Calibration interaction protocol

For each split, calibration is performed on held-out calibration subsets (no test leakage), comparing raw, Platt, and isotonic outputs. We then re-evaluate threshold behavior and occupancy-derived summaries to test whether calibration changes numerical reliability only, or materially changes operational geometry.

---

## 5. Ranking Performance Under Imbalance

Across synthetic and legacy settings, ranking metrics frequently remain strong even when default-threshold minority recall is weak. This reproduces the early study-journal observation: high AUROC/AP can coexist with operational collapse.

The key role of this section is setup: ranking quality alone is not sufficient to infer deployment accessibility. Subsequent sections explain this divergence through threshold dynamics and occupancy geometry.

[Figure Placeholder: Ranking Metrics vs Default-Threshold Recall]

---

## 6. Threshold Dynamics Under Imbalance

Threshold sweeps reveal distinct dynamics:
- CART often appears near-invariant across threshold relaxation.
- XGBoost frequently exhibits cliff-like recall recovery.
- HDDT-family and random-forest variants more often show broader, staged recovery.

From `reports/threshold_elasticity_summary.md`, XGBoost tends to exhibit the largest recall jumps and highest mean recall elasticity on difficult datasets (e.g., boundary), while CART remains near zero elasticity. These patterns motivate operational classes:
- **Threshold-invariant allocators** (minimal response across threshold ladder)
- **Cliff allocators** (abrupt recovery in narrow threshold regions)
- **Broad allocators** (distributed recovery with smoother accessibility gains)

Operational smoothness indices reinforce this divide: CART is maximal by construction of its flat response; XGBoost is frequently least smooth where cliffs dominate.

[Figure Placeholder: Recall vs Threshold and Elasticity Intervals]

---

## 7. Posterior Occupancy Geometry

This section is the conceptual center of the manuscript.

### 7.1 Posterior occupancy under imbalance

Posterior occupancy describes where prediction mass resides in \([0,1]\), with emphasis on minority-support accessibility above operational thresholds. Two models can rank similarly yet allocate minority mass very differently with respect to decision cutoffs. That allocation difference is operationally consequential.

We use three complementary views:
1. Class-conditional ECDFs (shape and compression)
2. Reachability curves \(R(t)\) (accessibility survival)
3. Unique support histograms/statistics (discreteness and concentration)

[Figure Placeholder: Occupancy Figure Family Overview]

### 7.2 Quantized posterior support

The strongest quantized behavior appears in CART. In `reports/posterior_occupancy_figures/posterior_occupancy_figure_summary.md`, CART has `unique_score_count_mean = 2.0`, `largest_mass_fraction_mean = 0.9429`, and `top_5_mass_fraction_mean = 1.0000`. This is an extreme finite-posterior-alphabet signature.

Mechanistically, finite leaves with empirical class-frequency outputs can induce coarse posterior alphabets under skew. Operationally, threshold movement becomes ineffective unless crossing one of a small number of mass points, producing staircase ECDFs and near-flat reachability segments. This explains threshold invariance without requiring superior ranking.

[Figure Placeholder: CART ECDF Staircase and Support Histogram]

### 7.3 Compressed minority occupancy

XGBoost illustrates a distinct mechanism: many unique scores may coexist with compressed minority accessibility. In other words, numerical granularity does not imply operational reachability. Minority mass can remain concentrated below practical thresholds, then recover abruptly when thresholds are aggressively relaxed.

This is consistent with cliff allocation behavior in elasticity summaries and with reachability collapse/recovery patterns. A useful interpretive rule emerges: **many unique posterior values is neither necessary nor sufficient for deployable minority accessibility**.

[Figure Placeholder: XGBoost Reachability Collapse]

### 7.4 Threshold reachability as accessibility bridge

Reachability curves operationalize the occupancy-to-recall bridge. Since \(R(t)\) equals minority survival above threshold, it maps directly to the achievable recall path as decision policy relaxes.

Steep local drops in \(R(t)\) imply fragile operating regions where small threshold shifts produce large recall changes. Flat segments imply stable but potentially unresponsive behavior. This framing unifies occupancy geometry and threshold dynamics in a single object.

[Figure Placeholder: Comparative Reachability Curves by Model Family]

### 7.5 Calibration persistence

Calibration improves probabilistic correctness in many settings (lower ECE/Brier for several models in synthesis tables), but often preserves broad operational regime identity. In `reports/paper_operational_synthesis.md`, best-calibration choices vary by model, yet regime assignments remain structurally coherent (e.g., CART quantized, XGBoost cliff, HDDT-family broad).

This supports a constrained claim: post-hoc calibration can correct reliability while leaving core occupancy geometry substantially intact. Therefore calibration and deployment accessibility are related but non-identical levers.

[Figure Placeholder: Pre/Post Calibration Reachability Overlay]

### 7.6 Occupancy regimes

Regime synthesis from `reports/allocation_regime_summary.md` and integrated synthesis indicates:
- **Quantized allocator:** CART
- **Cliff allocator:** XGBoost
- **Broad allocators:** HDDT, Bagged HDDT, Random Forest
- **Conservative allocator:** LightGBM (in current heuristic taxonomy)

These are descriptive, not ontological categories. Their value is explanatory: they compress diverse threshold phenomena into interpretable geometry classes that align with observed occupancy and elasticity patterns.

[Figure Placeholder: Regime Scatter (Support vs Recall Elasticity)]

---

## 8. Operational Trajectories

Threshold parameterization induces trajectories through precision-recall space. These trajectories represent deployment paths, not just evaluation curves. Under this view:
- Cliff allocators produce abrupt trajectory bends and narrow controllability bands.
- Quantized allocators produce piecewise-constant movement with large insensitive regions.
- Broad allocators produce more gradual traversal with wider controllability intervals.

Operationally, this matters for queue management and policy tuning: smooth trajectories allow incremental policy changes, while cliffs force brittle threshold jumps.

[Figure Placeholder: Threshold-Indexed PR Trajectory Panels]

---

## 9. Discussion

The manuscript’s practical claim is that deployment behavior under severe imbalance is governed by accessibility geometry, not ranking alone. This distinction is relevant to high-stakes systems where minority misses are costly and review capacity is finite.

In fraud/SOC/medical pipelines, a model can look excellent on AUROC while allocating minority mass below feasible thresholds, yielding operational fragility. Conversely, models with modest ranking but broader accessibility may offer safer controllability. The point is not that one family universally dominates; it is that allocation geometry determines how ranking translates into action.

This framing also clarifies why calibration alone may not resolve deployment failure: numerical reliability can improve while accessibility structure remains restrictive.

---

## 10. Limitations and Future Work

Current scope limitations:
- binary classification only
- no neural tabular baselines in this draft phase
- limited real-dataset suite (though diverse enough for repeated patterns)
- occupancy metrics and regime rules are still maturing
- no temporal or queue-coupled deployment simulation

Immediate future directions:
1. occupancy-derivative analysis beyond fixed threshold ladders
2. multiclass and structured-imbalance extensions
3. neural tabular occupancy geometry comparison
4. queue-aware decision policies coupled to reachability curves
5. robustness checks for regime persistence under broader data shifts

---

## 11. Conclusion

This first manuscript draft advances a focused thesis: **ranking is not deployment** under severe class imbalance. Classifier families induce distinct posterior occupancy geometries, these geometries shape threshold reachability, and threshold reachability governs operational accessibility.

CART’s quantized support, XGBoost’s cliff-like recovery, and the broader occupancy patterns of HDDT-family methods illustrate that deployment behavior is a structural property not captured by ranking metrics alone. Calibration helps, but often preserves core geometry. Therefore, operational model selection should explicitly evaluate occupancy and reachability, not only ranking quality.

---

## Draft Notes for Next Iteration

- tighten quantitative claims with direct table citations and confidence intervals
- insert finalized figure panel references and captions
- expand related work with targeted citations
- refine taxonomy language (descriptive vs causal)
- decide whether to split regime synthesis into methods or results subsections
