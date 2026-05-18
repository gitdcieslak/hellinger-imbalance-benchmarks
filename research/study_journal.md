# HDDT Imbalance Study Journal

---

# 2026-05-15

## Initial Goal

Initial objective:
- reimplement HDDT,
- benchmark against modern methods,
- determine whether HDDT still provides meaningful imbalance advantages.

At this point the project framing is mostly:
- “modern reproduction study.”

---

# 2026-05-16

## Synthetic Threshold Sweep Surprise

Unexpected observation:
- XGBoost and RandomForest achieve good AUROC/AP,
- but default-threshold recall collapses under severe imbalance.

Example:
- AUROC > 0.90,
- recall near 0.0 at threshold 0.50.

Threshold sweep recovery:
- lowering threshold to 0.01 recovers recall dramatically.

Initial interpretation:
- ranking quality != operational deployment behavior.

Need to investigate whether:
- this is synthetic artifact,
- calibration issue,
- or general posterior allocation phenomenon.

---

# 2026-05-16

## CART Behaves Very Differently

Threshold sweeps barely affect CART.

Possible explanation:
- leaf probabilities effectively discrete,
- thresholding has little impact,
- geometry fundamentally different from boosting ensembles.

Potential conceptual framing:
- “allocation regimes.”

Need:
- score distribution plots,
- leaf probability inspection.

---

# 2026-05-17

## HDDT Appears Operationally Distinct

HDDT often:
- lower AUROC,
- materially better recall/F1 at threshold 0.50.

Especially visible on:
- oil,
- compustat,
- boundary.

Possible explanation:
- less conservative minority posterior allocation.

Question:
- is this due to Hellinger splitting itself,
- or simply tree discreteness?

Need:
- compare against weighted CART,
- compare against bagged HDDT,
- inspect score distributions.

---

# 2026-05-17

## Real Dataset Validation

Legacy HDDT datasets reproduce the synthetic observations surprisingly well.

This is important because:
- synthetic artifact concern is reduced significantly.

Especially strong:
- boundary,
- cam,
- compustat.

Emerging belief:
- there may be a real distinction between:
  - ranking quality,
  - posterior allocation behavior.

---

# 2026-05-17

## Emerging Conceptual Shift

The study no longer feels like:
- “Does HDDT beat XGBoost?”

Instead it increasingly feels like:
- “How do classifiers allocate posterior mass under severe imbalance?”

This is probably the most important conceptual transition so far.

Need to avoid:
- benchmark-zoo drift,
- excessive model expansion.

Core conceptual thread feels more important now than adding more algorithms.

---

# Open Questions

## Calibration
Are these models actually miscalibrated?
Or merely conservative?

Need:
- reliability curves,
- ECE/Brier analysis.

---

## Probability Geometry
Would full prediction-space landscapes reveal:
- compressed minority regions,
- smoother boosting manifolds,
- discrete tree partitions?

Possibly valuable for figures later.

---

## Weighting Equivalence
Need to determine:
- whether LightGBM weighting approaches are operationally equivalent,
- whether weighting changes geometry or only threshold scale.

---

## Bagged HDDT
Bagged HDDT is performing better than expected.

Question:
- does bagging stabilize HDDT posterior allocation?
- does it reduce fragmentation?
- is this the real modern form of HDDT?

---

# Risks

## Benchmark Scope Explosion

Potential danger:
- project expands into giant benchmarking framework.

Need to keep:
- conceptual coherence,
- imbalance operational behavior focus.

---

## Hyperparameter Explosion

Avoid:
- Optuna sweeps,
- architecture wars,
- endless tuning.

The interesting signal currently survives under relatively modest/default configurations.


# 2026-05-18

## Threshold-Response Visualization Results

Generated threshold-response visualizations for legacy HDDT benchmark datasets using:
- repeated 5x2 stratified evaluation,
- fixed operational thresholds:
  - 0.50
  - 0.25
  - 0.10
  - 0.05
  - 0.01

Datasets examined:
- boundary
- cam
- compustat
- oil
- satimage

Models:
- CART
- HDDT
- Bagged HDDT
- Random Forest
- XGBoost
- LightGBM

Primary plots generated:
- recall vs threshold
- F1 vs threshold
- precision vs threshold
- balanced accuracy vs threshold

The recall-vs-threshold plots immediately revealed several strong structural behaviors.

---

# Observation: CART Appears Threshold-Invariant

CART behaves very differently from all ensemble methods.

Across nearly all datasets:
- recall remains nearly flat across thresholds,
- threshold relaxation has little effect,
- curves appear almost piecewise constant.

Interpretation:
- CART may allocate only a small discrete set of posterior probabilities,
- threshold movement therefore has limited operational effect,
- leaf probability structure may dominate operational behavior.

This now appears visually obvious rather than speculative.

Potential framing:
- “discrete allocation regime.”

Need future work:
- inspect leaf posterior distributions directly,
- quantify score discreteness,
- compare against HDDT leaf structure.

---

# Observation: XGBoost Exhibits Severe Threshold Collapse

XGBoost demonstrates the strongest threshold-collapse behavior.

Especially visible on:
- boundary
- cam
- compustat

Observed behavior:
- near-zero recall at threshold 0.50,
- minimal recovery at 0.25,
- dramatic nonlinear recall recovery at:
  - 0.10
  - 0.05
  - 0.01

Interpretation:
- XGBoost appears to preserve ranking quality,
- while allocating highly conservative posterior probabilities under imbalance.

This reinforces the emerging distinction between:
- ranking capability,
- and operational deployment behavior.

Important implication:
- AUROC alone substantially hides this phenomenon.

---

# Observation: Bagged HDDT Appears Operationally Stable

Bagged HDDT is emerging as one of the most operationally interesting models.

Observed behavior:
- materially higher recall at threshold 0.50,
- smoother threshold-response curves,
- less catastrophic collapse,
- less dependence on aggressive threshold relaxation.

Especially visible on:
- boundary
- cam
- compustat

Possible interpretations:
- HDDT may allocate minority posterior mass less conservatively,
- bagging may stabilize minority posterior allocation,
- HDDT ensembles may reduce posterior compression compared to boosting ensembles.

This may become one of the central empirical findings of the study.

---

# Observation: satimage Behaves Differently

satimage behaves substantially differently from:
- boundary
- cam
- compustat

Observed behavior:
- all models achieve relatively high recall even at threshold 0.50,
- threshold-response curves are smoother,
- operational collapse is much weaker.

Interpretation:
- threshold-collapse behavior is not universal,
- the phenomenon likely intensifies under more severe imbalance and/or more difficult minority geometry.

satimage may function as an important “healthy imbalance” comparison dataset.

This is scientifically important because it weakens the possibility that:
- threshold collapse is merely a universal artifact of all imbalanced learning.

---

# Observation: oil Appears to Be a Transitional Dataset

oil may represent an intermediate regime between:
- pathological threshold-collapse datasets,
and:
- healthier operational regimes like satimage.

Observed behavior:
- all models respond meaningfully to threshold relaxation,
- but collapse is less extreme than boundary/cam/compustat,
- family differences remain visible.

oil may become useful as:
- an explanatory bridge dataset,
- a visualization dataset,
- or a running example in the paper narrative.

---

# Emerging Conceptual Direction

The project increasingly appears to concern:

> how classifiers allocate posterior probability mass under severe imbalance.

Rather than:
- simple ranking quality,
- or benchmark leaderboard performance.

A potentially important emerging distinction:

| Property | Meaning |
|---|---|
| AUROC | ranking quality |
| Average Precision | ranking under imbalance |
| Threshold behavior | operational deployability |
| Probability allocation geometry | posterior mass structure |
| Calibration | probabilistic correctness |

These properties increasingly appear:
- related,
- but fundamentally non-equivalent.

---

# Emerging Allocation Regimes

Current tentative conceptual framing:

| Allocation Regime | Characteristics | Candidate Models |
|---|---|---|
| Conservative allocators | strong ranking, severe threshold collapse | XGBoost, RF, LightGBM |
| Moderate allocators | smoother operational behavior | HDDT, Bagged HDDT |
| Discrete allocators | threshold-invariant behavior | CART |

This framing remains preliminary but now appears visually supported.

---

# Important Future Work

## Precision-vs-threshold analysis

Current recall plots show operational recovery.

Need to determine:
- what precision cost is paid for recovery,
- whether threshold relaxation remains operationally viable.

Potentially critical next figure.

---

## Score Geometry

Need deeper analysis of:
- posterior compression,
- minority score occupancy,
- score quantiles,
- ECDF behavior,
- score manifold structure.

Potentially important for:
- “allocation geometry” framing.

---

## Calibration

Need to determine:
- whether collapse is purely calibration-related,
- or reflects deeper posterior allocation structure.

Possible future work:
- reliability diagrams,
- ECE,
- isotonic calibration,
- Platt scaling.

---

# Meta Observation

The study is increasingly diverging from:
- “Does HDDT outperform modern methods?”

and evolving toward:

> “How do different classifier families express uncertainty and allocate posterior probability mass under severe class imbalance?”

This appears substantially more scientifically interesting.

# 2026-05-18

## Allocation Geometry Is Becoming the Central Theme

The project now appears to be converging on:

> operational allocation geometry under severe imbalance

rather than:
- classifier leaderboard performance,
- or generic imbalance benchmarking.

This feels like the most important conceptual clarification so far.

---

# Allocation Concentration Metrics

Implemented:
- entropy,
- effective support size,
- Gini concentration,
- score-mass occupancy metrics.

Initial observations:

## CART
- extremely low entropy,
- extremely low support size,
- effectively quantized posterior allocation.

Operational interpretation:
- threshold movement has almost no effect,
- posterior space appears discretized.

This now strongly supports the:
> “quantized allocator”
framing.

---

## XGBoost
- moderate entropy,
- but extremely high recall elasticity,
- strongest cliff behavior observed so far.

Operational interpretation:
- XGBoost appears highly operationally unstable under severe imbalance,
- recall recovery often occurs abruptly only after aggressive threshold relaxation.

This appears to support:
> “cliff allocator”
behavior.

---

## HDDT and Bagged HDDT
Observed:
- broader support,
- smoother recall recovery,
- less catastrophic threshold collapse.

Bagged HDDT in particular appears:
- operationally stable,
- relatively controllable,
- surprisingly competitive.

Possible interpretation:
- bagging stabilizes minority posterior allocation geometry.

This is becoming scientifically interesting.

---

# Threshold Elasticity Appears Extremely Important

Threshold elasticity analysis appears more informative than expected.

The most important distinction now may not be:
- ranking quality,

but:
- how operational behavior changes as thresholds relax.

This seems especially relevant in:
- human-review systems,
- queue-based fraud systems,
- medical triage pipelines.

Potential key idea:
> some classifiers are operationally fragile.

---

# Precision–Recall Trajectories

The new trajectory plots appear extremely valuable.

Observed:
- boosted ensembles often exhibit long low-recall regions followed by abrupt transitions,
- CART trajectories remain nearly stationary,
- HDDT-family models appear smoother and broader.

The trajectory framing feels substantially stronger than:
- static PR curves,
- or isolated threshold plots.

Potential interpretation:
- classifiers trace fundamentally different operational paths through deployment space.

---

# Emerging Big Picture

The project now appears to concern:

| Property | Meaning |
|---|---|
| Ranking quality | relative ordering ability |
| Allocation geometry | posterior mass structure |
| Threshold elasticity | operational sensitivity |
| Smoothness | deployment controllability |
| Calibration | probabilistic correctness |

These properties increasingly appear:
- related,
- but non-equivalent.

This may ultimately become the central scientific contribution.

---

# Important Open Questions

## Calibration Interaction
Does calibration:
- fundamentally alter allocation geometry,
or:
- merely reparameterize thresholds?

This now feels like one of the most important unresolved questions.

---

## Prediction-Space Occupancy
Need to investigate:
- minority occupancy regions,
- posterior sparsity,
- score manifold structure,
- score quantization.

This could become:
- a major figure family,
- or possibly a follow-on paper.

---

## Neural Tabular Models
Unknown:
- whether neural tabular models behave more like:
  - conservative allocators,
  - cliff allocators,
  - or broad allocators.

Potentially important future comparison.

---

# Meta Observation

The project increasingly feels less like:
- an HDDT reproduction study,

and more like:

> a study of operational probability allocation behavior under severe class imbalance.

That conceptual transition now feels complete.