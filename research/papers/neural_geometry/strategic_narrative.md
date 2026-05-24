# Operational Probability Accessibility Under Severe Class Imbalance

## Toward a Theory of Allocation Geometry, Reachability Morphology, and Threshold Behavior

---

# Working Thesis

Modern classifier evaluation under severe class imbalance is overly dominated by:

* ranking metrics,
* aggregate discrimination quality,
* and calibration measures.

However, these properties do not fully characterize:

* operational accessibility,
* threshold behavior,
* deployment controllability,
* or posterior occupancy structure.

This work proposes that classifiers induce distinct:

> operational probability accessibility geometries

under severe imbalance.

These geometries shape:

* threshold behavior,
* recall collapse,
* accessibility persistence,
* operational smoothness,
* and deployment robustness,

in ways not captured by AUROC or Average Precision alone.

---

# Central Conceptual Claim

The core emerging hypothesis is:

> Ranking quality, calibration quality, and operational accessibility morphology are related but fundamentally non-equivalent properties.

This distinction appears especially important under:

* extreme skew,
* resource-constrained review systems,
* and threshold-mediated deployment environments.

---

# Candidate Core Objects

## 1. Reachability

Define minority reachability:

[
R(t) = P(\hat{p}(x) \ge t \mid y = 1)
]

Interpretation:

* probability that minority examples remain operationally accessible at threshold (t).

This object unifies:

* recall,
* threshold accessibility,
* operational persistence,
* occupancy survival,
* and deployment sensitivity.

---

## 2. Accessibility Morphology

Not merely:

* whether minority posterior support exists,

but:

* how accessibility decays as thresholds increase.

Candidate morphology properties:

* cliff-like decay,
* smooth decay,
* persistence,
* elasticity concentration,
* calibration sensitivity,
* occupancy continuity.

---

## 3. Operational Smoothness

Classifiers differ in:

* how abruptly operational behavior changes under threshold movement.

Define:

* operationally smooth allocators,
* cliff allocators,
* quantized allocators,
* broad allocators.

This morphology appears classifier-family-dependent but also optimization-sensitive.

---

# Narrative Arc of the Research Program

---

# Phase I — HDDT Reproduction

Initial question:

> Does HDDT still outperform modern imbalance methods?

Early experiments:

* reproduced historical HDDT behavior,
* compared against:

  * XGBoost,
  * RandomForest,
  * LightGBM,
  * CART.

Unexpected finding:

* strong ranking models often collapsed operationally at realistic thresholds.

This produced the first important distinction:

| Property           | Observation          |
| ------------------ | -------------------- |
| AUROC              | strong               |
| Recall@0.50        | catastrophic         |
| Threshold recovery | nonlinear/cliff-like |

This initiated the conceptual transition away from:

* leaderboard benchmarking.

---

# Phase II — Allocation Geometry

The project evolved toward:

> how classifiers allocate posterior probability mass under imbalance.

Observed distinctions:

| Regime                  | Characteristics               |
| ----------------------- | ----------------------------- |
| cliff allocators        | abrupt recall recovery        |
| quantized allocators    | threshold-invariant behavior  |
| broad allocators        | smoother accessibility        |
| conservative allocators | compressed minority occupancy |

Key insight:

* ranking quality alone obscures operational accessibility structure.

---

# Phase III — Occupancy and Reachability

The framework expanded to include:

* occupancy entropy,
* support size,
* posterior concentration,
* threshold elasticity,
* occupancy persistence,
* ECDF behavior,
* reachability curves.

This enabled:

* operational morphology analysis.

The project increasingly shifted from:

* “which classifier wins”
  toward:
* “how operational accessibility evolves under threshold movement.”

---

# Phase IV — Neural Extension

A constrained MLP baseline integrated cleanly into the framework.

Important result:

* neural learners are analyzable through the same operational geometry lens.

Unexpected finding:

* baseline BCE-trained MLPs behaved as:

> cliff allocators.

This suggested:

* cliff morphology is not tree-specific.

---

# Phase V — Objective Perturbation Transition

The first major theoretical breakthrough emerged from:

* fixed-architecture MLP perturbation experiments.

Under:

* oversampling,
* weighting,
* and altered imbalance pressure,

MLP transitioned from:

> cliff_allocator
> to:
> smooth_allocator

without architecture changes.

This appears extremely important.

Emerging implication:

> operational geometry is not architecture-only.

Instead:

* optimization pressure,
* imbalance structure,
* and accessibility incentives

appear to shape morphology.

---

# Phase VI — Geometry Transition Analysis

Transition analysis revealed:

* support broadening helps but is not sufficient,
* smoothness can improve without large support expansion,
* threshold morphology changes dominate the transition,
* and calibration may improve reliability while worsening accessibility smoothness.

This produced another major distinction:

| Property        | Not Equivalent To         |
| --------------- | ------------------------- |
| support breadth | operational smoothness    |
| calibration     | accessibility persistence |
| ranking         | deployability             |
| reliability     | morphology                |

This appears central.

---

# Emerging Theoretical Structure

The project increasingly suggests classifiers possess multiple partially independent operational dimensions:

| Dimension                | Meaning                   |
| ------------------------ | ------------------------- |
| ranking quality          | relative ordering         |
| calibration              | probabilistic correctness |
| support breadth          | occupancy extent          |
| accessibility morphology | threshold decay structure |
| smoothness               | threshold controllability |
| persistence              | accessibility survival    |

These dimensions appear:

* coupled,
  but:
* fundamentally non-equivalent.

---

# Important Emerging Hypotheses

---

# H1 — Operational Geometry Is Optimization-Sensitive

Allocation morphology changes under:

* weighting,
* oversampling,
* and imbalance pressure.

Therefore:

> operational accessibility is not purely architectural.

---

# H2 — Calibration and Operational Smoothness May Conflict

Observed:

* ECE/Brier improve under calibration,
  while:
* accessibility morphology sometimes re-steepens.

Possible implication:

> probabilistic correctness may trade off against operational smoothness.

This appears potentially very important.

---

# H3 — Reachability Morphology May Be a Primary Classifier Signature

Different classifier families appear to induce:

* distinct reachability shapes,
* elasticity concentration patterns,
* and occupancy persistence behavior.

Possibly:

> the shape of (R(t)) itself may become a classifier-family signature.

---

# H4 — Bagged HDDT Remains Structurally Distinct

Bagged HDDT continues exhibiting:

* broader occupancy,
* smoother accessibility,
* and less catastrophic threshold collapse.

Current evidence suggests this is not fully explained by:

* underfitting,
* weak ranking,
* or low-capacity behavior.

This remains one of the most important unresolved mechanisms.

---

# Proposed Conceptual Vocabulary

| Term                     | Meaning                                  |
| ------------------------ | ---------------------------------------- |
| allocation geometry      | global posterior structure               |
| accessibility morphology | threshold accessibility behavior         |
| reachability persistence | survival under threshold increase        |
| cliff allocation         | concentrated elasticity transitions      |
| smooth allocation        | distributed threshold response           |
| occupancy compression    | minority posterior collapse              |
| operational smoothness   | controllability under threshold movement |
| threshold morphology     | shape of accessibility decay             |

---

# Candidate Manuscript Structure

## 1. Introduction

* severe imbalance and operational deployment failure
* limits of ranking metrics
* motivating operational accessibility

## 2. Related Work

* imbalance learning
* calibration
* threshold optimization
* uncertainty estimation
* operational ML

## 3. Allocation Geometry Framework

* occupancy
* accessibility
* reachability
* elasticity
* operational smoothness

## 4. Experimental Framework

* synthetic + legacy datasets
* repeated evaluation
* threshold sweeps
* occupancy metrics

## 5. Allocation Regimes

* cliff
* quantized
* broad
* conservative

## 6. Neural Extension

* MLP integration
* neural cliff behavior

## 7. Objective Perturbation Transitions

* weighting
* oversampling
* geometry transitions

## 8. Calibration vs Accessibility

* calibration-induced re-steepening
* reliability vs morphology tension

## 9. Mechanistic Interpretation

* optimization pressure
* occupancy persistence
* threshold morphology
* unresolved Bagged HDDT mechanism

## 10. Implications

* fraud systems
* medical triage
* operational queues
* deployment-sensitive ML

## 11. Limitations

* current model scope
* no large neural architectures yet
* no custom occupancy-aware objectives yet

## 12. Future Work

* focal loss
* reachability-aware optimization
* occupancy-aware objectives
* representation-space geometry
* operational morphology regularization

---

# Most Important Conceptual Transition

The project no longer appears primarily concerned with:

> “Which classifier performs best?”

Instead, it increasingly concerns:

> how classifiers allocate operationally accessible probability mass under severe class imbalance.

That now appears to be the true center of gravity of the research program.
