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

# 2026-05-19

## Posterior Occupancy Geometry Begins to Stabilize

The project appears to have crossed an important conceptual threshold.

Initial work focused primarily on:
- benchmark modernization,
- HDDT reproduction,
- threshold-sweep behavior,
- and operational instability observations.

However, the introduction of:
- posterior occupancy metrics,
- threshold reachability curves,
- ECDF occupancy analysis,
- and posterior support histograms

has substantially clarified the emerging theory direction.

The work increasingly appears to support the hypothesis that classifier families induce structurally distinct posterior occupancy geometries under severe imbalance.

Several particularly important observations emerged:

### CART
CART now appears strongly explained by:
- finite discrete posterior support,
- extremely small posterior alphabets,
- and highly concentrated posterior mass.

Operational threshold invariance increasingly appears to emerge directly from posterior support collapse.

Observed:
- unique_score_count ≈ 2
- top_5_mass_fraction ≈ 1.0

The ECDF and reachability curves visually reinforce this interpretation through staircase occupancy transitions and piecewise-constant threshold behavior.

### XGBoost
XGBoost appears to exhibit a fundamentally different failure mode.

Unlike CART:
- XGBoost produces many unique posterior values,
- but still exhibits severe operational fragility and threshold collapse.

This suggests:
- the problem is not posterior discreteness,
but:
- compressed minority occupancy accessibility.

Minority posterior mass appears concentrated into low-probability regions that remain operationally inaccessible at realistic thresholds.

This may become one of the paper’s most important findings.

### HDDT / Bagged HDDT
HDDT-family methods increasingly appear to produce:
- broader posterior occupancy,
- smoother threshold accessibility,
- and less catastrophic operational collapse.

Bagged HDDT especially appears to stabilize occupancy support while preserving broad operational accessibility.

### Reachability Curves
Threshold reachability curves now appear to be one of the paper’s most important conceptual objects:

R(t) = P( p_hat(x) >= t | y = 1 )

This framing unifies:
- recall,
- occupancy survival,
- operational accessibility,
- threshold elasticity,
- and deployment behavior.

The paper increasingly appears less concerned with:
- “which classifier wins,”

and more concerned with:
- “how classifier families allocate operationally accessible posterior probability mass under severe imbalance.”

This feels like a major conceptual transition for the project.

# 2026-05-22

## Neural MLP Extension Successfully Integrates into Allocation Geometry Framework

Completed the first constrained neural extension experiment using a lightweight MLP baseline integrated directly into the existing operational allocation geometry framework.

Important outcome:

* the framework generalized cleanly to neural learners,
* without requiring substantial architectural modification,
* special-case neural logic,
* or new metric families.

This appears conceptually important.

The existing operational allocation metrics:

* threshold elasticity,
* occupancy geometry,
* posterior concentration,
* reachability behavior,
* and regime synthesis

all remained meaningful for neural outputs.

This substantially strengthens the emerging belief that:

> operational allocation geometry may represent a model-family-independent analysis framework rather than a tree-specific phenomenon.

---

# MLP Does NOT Behave Like a Broad Allocator

One of the most important findings from the initial neural extension:

MLP did not behave like:

* HDDT,
* or Bagged HDDT.

Instead, MLP primarily expressed:

> cliff allocator behavior.

This is scientifically important.

Observed:

* strong threshold-collapse behavior,
* operational fragility under severe imbalance,
* poor default-threshold accessibility,
* and large nonlinear recall recovery after threshold relaxation.

Especially notable:

* full default-threshold collapse on:

  * boundary,
  * compustat,
* near-collapse on:

  * cam.

At threshold 0.01:

* substantial recall recovery occurred.

This strongly resembles:

* the earlier XGBoost observations,
  rather than:
* the broader occupancy behavior observed in Bagged HDDT.

This is an important conceptual reinforcement.

---

# Neural Learners Appear Operationally Analyzable

The most important meta-result may not be MLP performance itself.

The more important result may be:

> neural learners appear analyzable as operational allocation systems.

This is significant because:

* the project initially emerged from tree-based imbalance behavior,
* but the operational geometry framing now appears portable across model families.

The allocation-regime framework:

* conservative,
* cliff,
* broad,
* quantized

appears capable of describing neural behavior as well.

That substantially increases confidence that:

* the project may be uncovering deeper properties of classifier uncertainty allocation under imbalance.

---

# Calibration Improves Error Metrics But Preserves Regime Identity

One of the strongest conceptual findings so far:

Calibration materially improved:

* ECE,
* and Brier score,

but:

* did NOT fundamentally alter the inferred operational regime.

MLP remained:

> cliff-like

under:

* raw,
* Platt,
* and isotonic calibration.

This appears extremely important.

Possible implication:

> calibration may reparameterize threshold accessibility without fundamentally altering posterior occupancy geometry.

If this continues to hold across:

* additional neural learners,
* and additional classifier families,

then:

* allocation geometry may represent a deeper classifier-family property than calibration quality alone.

This now feels like one of the most important emerging theoretical directions in the project.

---

# Bagged HDDT Now Appears More Structurally Distinct

The neural results indirectly strengthened the apparent uniqueness of Bagged HDDT.

The original possibility was:

* HDDT-family behavior might simply emerge from:

  * lower-capacity learners,
  * or weaker ranking quality.

The MLP results weaken this interpretation.

Despite:

* smooth function approximation,
* dense learned representations,
* and continuous outputs,

MLP still exhibited:

* occupancy compression,
* operational cliffs,
* and threshold fragility.

This increases the plausibility that:

* Bagged HDDT may possess structurally distinct allocation properties,
  rather than:
* merely underfit behavior.

This now feels increasingly important.

---

# Emerging Hypothesis: Representation Learning May Encourage Occupancy Compression

A potentially important speculative direction emerged:

Modern neural learners may optimize:

* ranking,
* separability,
* and likelihood,

while simultaneously compressing minority posterior occupancy into operationally inaccessible probability regions.

Possible interpretation:

> representation learning itself may encourage conservative posterior allocation under severe imbalance.

This remains highly speculative but now appears plausible enough to investigate further.

---

# Important Conceptual Transition

The project now appears substantially removed from:

* “Does HDDT outperform modern models?”

and increasingly centered on:

> how classifier families allocate operationally accessible posterior probability mass under severe imbalance.

The neural extension appears to reinforce rather than weaken this conceptual transition.

That feels significant.

---

# Important Constraints Going Forward

Need to continue resisting:

* architecture-zoo expansion,
* hyperparameter wars,
* benchmark drift,
* and leaderboard framing.

The strongest signal currently remains:

> operational allocation geometry itself.

Neural learners should continue to be treated primarily as:

* probes into allocation behavior,
  rather than:
* competitive benchmark entrants.

---

# Immediate Next Questions

## Second Neural Model

Need to determine whether:

* FT-Transformer,
* TabNet,
* or another structured neural learner

exhibits:

* similar cliff behavior,
* broader occupancy,
* or an entirely new operational regime.

Only one additional neural architecture should likely be added initially.

---

## Occupancy-Preserving Objectives

Need to investigate whether neural objectives could explicitly encourage:

* broader minority occupancy,
* smoother threshold reachability,
* or operationally stable posterior support.

Potential directions:

* occupancy regularization,
* threshold-aware losses,
* reachability-aware objectives,
* or elasticity penalties.

---

## Reachability as a Primary Object

Threshold reachability curves increasingly appear central:

[
R(t) = P(\hat{p}(x) \ge t \mid y = 1)
]

This object now appears to unify:

* recall,
* occupancy accessibility,
* threshold elasticity,
* and deployment behavior.

Possibly one of the most important conceptual objects in the entire project.
