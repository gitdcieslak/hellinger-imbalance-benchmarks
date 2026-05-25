# Operational Morphology Framework

## Working Research Framework for Operational Probability Accessibility Under Severe Class Imbalance

---

# Status of This Document

This document is:

* a working conceptual framework,
* an evolving synthesis artifact,
* and a research coordination document.

It is NOT:

* a finalized theory,
* a formal publication draft,
* or a complete mathematical treatment.

The purpose of this document is to:

* stabilize terminology,
* organize emerging concepts,
* guide experiment design,
* and consolidate mechanistic hypotheses.

This framework should evolve as:

* experiments accumulate,
* concepts stabilize,
* and mechanisms become clearer.

---

# Core Motivation

Most classifier evaluation under severe class imbalance emphasizes:

* ranking metrics,
* classification accuracy,
* and calibration quality.

However, these metrics do not fully characterize:

* operational accessibility,
* threshold-mediated deployment behavior,
* queue sensitivity,
* operational controllability,
* or accessibility persistence.

This framework emerges from the observation that:

> classifiers may achieve strong ranking quality while exhibiting catastrophic operational accessibility collapse at realistic deployment thresholds.

This distinction increasingly appears fundamental.

---

# Central Hypothesis

The framework currently centers on the following hypothesis:

> Ranking quality, calibration quality, and operational accessibility morphology are related but fundamentally non-equivalent properties.

This appears especially important under:

* extreme skew,
* threshold-mediated deployment,
* resource-constrained review systems,
* fraud detection,
* medical triage,
* and operational queue systems.

---

# Guiding Perspective

The framework does NOT primarily view classifiers as:

* static ranking systems.

Instead, classifiers are viewed as:

> operational accessibility allocation systems.

The central question becomes:

> how do classifiers allocate operationally accessible posterior probability mass under severe imbalance?

---

# Core Conceptual Objects

---

# 1. Reachability

Primary object:

[
R(t) = P(\hat{p}(x) \ge t \mid y = 1)
]

Interpretation:

* probability that minority examples remain operationally accessible at threshold (t).

Reachability unifies:

* recall,
* threshold accessibility,
* occupancy survival,
* operational persistence,
* and deployment sensitivity.

---

## Candidate Reachability Features

| Feature                  | Meaning                           |
| ------------------------ | --------------------------------- |
| area under reachability  | global accessibility              |
| reachability persistence | survival under threshold increase |
| decay rate               | accessibility steepness           |
| half-life threshold      | accessibility survival point      |
| derivative localization  | where collapse occurs             |
| curvature                | smooth vs phase-like decay        |
| integrated accessibility | cumulative operational access     |

---

# 2. Accessibility Morphology

Accessibility morphology describes:

> the shape of operational accessibility behavior under threshold evolution.

This differs from:

* ranking quality,
* calibration,
* or posterior support breadth alone.

Candidate morphology forms:

* cliff-like,
* smooth,
* quantized,
* broad,
* phase-transition-like,
* calibration-sensitive.

---

# 3. Operational Smoothness

Operational smoothness describes:

* how gradually accessibility changes as thresholds evolve.

Smooth allocators:

* distribute operational change across thresholds.

Cliff allocators:

* concentrate operational transition into narrow threshold regions.

Smoothness appears operationally important because:

* threshold tuning,
* review queues,
* and deployment systems
  often require controllable accessibility behavior.

---

# 4. Occupancy

Occupancy describes:

* how posterior probability mass is distributed.

Important occupancy concepts include:

* support breadth,
* entropy,
* sparsity,
* concentration,
* compression,
* and accessibility persistence.

Current evidence suggests:

> occupancy breadth alone is insufficient to explain operational smoothness.

This distinction appears extremely important.

---

# 5. Elasticity

Elasticity describes:

* sensitivity of operational behavior to threshold movement.

Current evidence suggests:

* cliff allocators exhibit concentrated elasticity,
* smooth allocators redistribute elasticity across thresholds.

Elasticity appears central to:

* deployment robustness,
* operational controllability,
* and threshold stability.

---

# Important Emerging Distinctions

Current evidence suggests the following properties are:

* related,
  but:
* fundamentally non-equivalent.

| Property                  | Meaning                           |
| ------------------------- | --------------------------------- |
| ranking quality           | relative ordering ability         |
| calibration               | probabilistic correctness         |
| support breadth           | occupancy extent                  |
| accessibility persistence | survival under threshold increase |
| operational smoothness    | controllability                   |
| threshold morphology      | accessibility decay structure     |
| compression               | minority occupancy collapse       |

This non-equivalence currently appears central to the research program.

---

# Current Regime Taxonomy

The current taxonomy is:

* provisional,
* empirical,
* and revisable.

---

## Cliff Allocators

Characteristics:

* concentrated elasticity,
* abrupt accessibility collapse,
* large recall jumps,
* operational fragility.

Examples observed:

* XGBoost
* BCE-trained MLP

---

## Smooth Allocators

Characteristics:

* distributed elasticity,
* smoother threshold evolution,
* improved operational controllability.

Examples observed:

* weighted MLP
* oversampled MLP

---

## Broad Allocators

Characteristics:

* broader occupancy persistence,
* smoother accessibility survival,
* less catastrophic collapse,
* operational accessibility stability.

Primary observed example:

* Bagged HDDT

Current mechanisms remain unresolved.

---

## Quantized Allocators

Characteristics:

* discrete posterior support,
* threshold-invariant behavior,
* staircase accessibility transitions.

Primary observed example:

* CART

---

## Conservative Allocators

Characteristics:

* compressed minority occupancy,
* low operational accessibility at realistic thresholds,
* strong ranking with weak operational reachability.

Observed frequently in:

* boosted ensembles.

---

# Mechanism Hypotheses

This section captures active candidate explanations.

These hypotheses remain provisional.

---

# H1 — Optimization Pressure Shapes Morphology

Current evidence strongly suggests:

* weighting,
* oversampling,
* and imbalance pressure

can materially alter operational morphology under fixed architecture.

This appears to weaken:

> architecture-only explanations.

---

# H2 — BCE Encourages Accessibility Compression

Possible mechanism:

* BCE under severe imbalance rewards majority confidence more strongly than broad minority accessibility.

This may induce:

* conservative occupancy,
* compressed reachability,
* and cliff-like threshold behavior.

---

# H3 — Calibration May Trade Off Against Smoothness

Observed:

* calibration improves ECE/Brier,
  while:
* operational accessibility sometimes re-steepens.

Possible implication:

> probabilistic correctness and operational smoothness may conflict.

This currently appears one of the most important emerging findings.

---

# H4 — Bagged HDDT Preserves Accessibility Structure

Bagged HDDT consistently exhibits:

* broader accessibility,
* smoother persistence,
* and reduced collapse severity.

Potential explanations include:

* Hellinger splitting,
* bagging stabilization,
* occupancy smoothing,
* reduced accessibility compression,
* or ensemble persistence effects.

Current mechanism remains unresolved.

---

# H5 — Support Breadth Is Not Sufficient

Current evidence suggests:

* smoothness improvements can occur without large support expansion.

This appears extremely important.

Possible implication:

> accessibility morphology depends not merely on occupancy quantity, but on occupancy accessibility topology.

---

# Candidate Core Morphology Dimensions

Current candidate basis dimensions:

| Dimension               | Candidate Feature                  |
| ----------------------- | ---------------------------------- |
| accessibility           | area under reachability            |
| smoothness              | max elasticity                     |
| persistence             | threshold survival                 |
| concentration           | occupancy entropy                  |
| compression             | minority support width             |
| calibration sensitivity | morphology delta after calibration |

These dimensions remain provisional.

---

# Calibration-Morphology Interaction

One of the strongest emerging themes:

Calibration:

* improves reliability metrics,
  but:
* may worsen operational accessibility morphology.

Observed effects:

* increased accessibility steepness,
* reduced smoothness,
* regime re-steepening,
* and accessibility redistribution.

This appears scientifically important because:

* calibration literature often implicitly assumes operational improvement follows probabilistic improvement.

Current evidence suggests:

> this may not universally hold under severe imbalance.

---

# Geometry Transition Perspective

The framework increasingly treats:

* morphology itself,
  as:
* a dynamic object.

Current evidence suggests:

* optimization perturbations,
* calibration transforms,
* and imbalance pressure

can induce:

* regime transitions,
* morphology transitions,
* and accessibility redistribution.

This transition perspective now appears central.

---

# Operational Interpretation Layer

The framework remains grounded in:

* deployment behavior,
  not:
* abstract geometry alone.

Potential operational implications include:

| Operational Domain  | Morphology-Relevant Concern  |
| ------------------- | ---------------------------- |
| fraud detection     | review queue controllability |
| medical triage      | accessibility persistence    |
| intrusion detection | threshold robustness         |
| resource allocation | smooth workload growth       |
| monitoring systems  | operational stability        |

This operational grounding is currently one of the framework’s most important features.

---

# Current Research Priorities

---

# Priority 1 — Mechanism Isolation

Highest-priority open question:

> what structural mechanisms produce broad accessibility morphology?

Especially:

* why Bagged HDDT remains structurally distinct.

---

# Priority 2 — Reachability Theory

Need deeper understanding of:

* reachability shape,
* elasticity localization,
* persistence,
* calibration sensitivity,
* and morphology invariants.

---

# Priority 3 — Morphology Transition Analysis

Need stronger characterization of:

* cliff → smooth transitions,
* calibration-induced re-steepening,
* and optimization-induced geometry shifts.

---

# Priority 4 — Controlled Neural Perturbation

Future controlled perturbations may include:

* focal loss,
* balanced minibatches,
* label smoothing,
* occupancy-aware objectives,
* and threshold-aware optimization.

These should remain:

* tightly constrained,
* mechanism-focused,
* and interpreted through morphology.

---

# Priority 5 — Operational Queue Simulation

Longer-term direction:

* connect morphology directly to:

  * queue systems,
  * review capacity,
  * threshold governance,
  * and deployment robustness.

---

# Important Constraints

The framework should avoid:

* benchmark-zoo drift,
* architecture proliferation,
* premature formalism,
* and abstract geometry disconnected from deployment behavior.

The framework should remain:

> operationally grounded.

---

# Current Position of the Research Program

The project no longer appears primarily concerned with:

> “Which classifier performs best?”

Instead, it increasingly concerns:

> how classifiers allocate operationally accessible posterior probability mass under severe class imbalance.

This now appears to be the central organizing principle of the research direction.
