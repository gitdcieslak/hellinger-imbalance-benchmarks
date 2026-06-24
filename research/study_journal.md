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

# 2026-05-23

## Objective Perturbation Significantly Alters Neural Allocation Geometry

Completed the first controlled neural objective perturbation study using a fixed MLP architecture under varying imbalance-pressure mechanisms.

This appears to be one of the most important results in the project so far.

The key finding:

> operational allocation geometry is not determined by architecture alone.

This now appears strongly supported.

---

# Controlled MLP Perturbation Study

The experiment held:

* architecture,
* hidden layers,
* optimization family,
* and overall model class

effectively constant while varying:

* imbalance pressure,
* sampling behavior,
* and weighting strategy.

Variants examined:

* `mlp_bce`
* `mlp_oversampled`
* `mlp_weighted`

Importantly:

* no architecture-zoo expansion occurred,
* no GPU-heavy infrastructure was required,
* no custom occupancy-aware loss was introduced,
* and the existing operational allocation framework remained fully reusable.

This substantially strengthens confidence in the framework itself.

---

# MLP BCE Remains Cliff-Like

The baseline BCE-trained MLP reproduced the previously observed operational cliff behavior.

Observed:

* poor default-threshold accessibility,
* large recall recovery cliffs,
* low operational smoothness,
* and severe threshold sensitivity.

Regime synthesis continued classifying:

> `mlp_bce` as `cliff_allocator`.

This strongly reinforces the earlier observation that:

* standard imbalance training pressure may encourage conservative posterior accessibility.

---

# Oversampling and Weighting Materially Change Geometry

The most important result:

Both:

* `mlp_oversampled`
* and:
* `mlp_weighted`

transitioned from:

> `cliff_allocator`

to:

> `smooth_allocator`

under the existing regime synthesis framework.

This is extremely important.

Observed improvements:

* substantially higher default-threshold recall,
* reduced threshold-collapse severity,
* improved operational smoothness,
* reduced maximum recall jumps,
* and broader effective posterior support.

The most striking implication:

> operational geometry can move substantially under altered imbalance pressure even when architecture remains fixed.

This now appears strongly supported empirically.

---

# Geometry Is Emerging as an Optimization Property

The project now appears to be converging toward a deeper interpretation:

> operational probability accessibility geometry may emerge from the interaction between:
>
> * optimization pressure,
> * imbalance structure,
> * and representation behavior.

This appears substantially richer than:

* simple architecture comparisons,
  or:
* leaderboard benchmarking.

The current evidence increasingly suggests:

* optimization objectives shape operational accessibility structure,
* not merely ranking quality.

This feels like a major conceptual transition.

---

# Important Observation: Ranking and Geometry Continue to Decouple

One of the strongest recurring patterns:

Changes that improved:

* operational smoothness,
* accessibility,
* and threshold behavior

did not always align perfectly with:

* ranking metrics.

This continues reinforcing the distinction between:

* ranking quality,
* and operational deployability.

That distinction now appears central to the project.

---

# Bagged HDDT Remains Structurally Interesting

The perturbation study unexpectedly strengthened the scientific importance of Bagged HDDT.

Initially, one possible interpretation was:

* broad allocation behavior might simply reflect:

  * weaker learners,
  * or underfit behavior.

The new MLP perturbation results weaken this interpretation.

Even with:

* continuous outputs,
* learned representations,
* and altered imbalance pressure,

MLP variants moved only partially toward:

> broad allocation behavior.

They became:

> smoother,

but did not fully reproduce:

* Bagged-HDDT-like occupancy breadth.

This now increases the plausibility that:

* Bagged HDDT may possess structurally distinct accessibility properties,
  rather than:
* merely reflecting optimization weakness.

This feels increasingly important.

---

# Calibration Results Became More Nuanced

The earlier neural study suggested:

* calibration improved ECE/Brier,
* while preserving regime identity.

The perturbation study complicated this interpretation.

Observed:

* `mlp_bce` remained regime-stable (`cliff_allocator`) after calibration,
  but:
* `mlp_oversampled`
* and:
* `mlp_weighted`

sometimes transitioned back toward:

> cliff-like interpretations after calibration transforms.

This is scientifically important.

Possible implication:

> calibration may interact nontrivially with occupancy geometry and threshold accessibility.

This weakens the earlier possibility that:

* calibration simply reparameterizes thresholds while preserving geometry completely.

The relationship between:

* calibration,
* accessibility,
* and operational morphology

now appears more subtle than initially believed.

---

# Reachability and Geometry Transition Analysis Now Appear Central

The perturbation study suggests that:

* geometry itself is evolving under optimization pressure.

This now motivates a deeper transition analysis focused on:

* reachability curve evolution,
* occupancy persistence,
* threshold elasticity trajectories,
* posterior support expansion/compression,
* and operational morphology shifts.

The project increasingly appears less concerned with:

* “Which classifier wins?”

and more concerned with:

> “How do optimization pressures shape operational probability accessibility geometry under severe imbalance?”

That now feels like the central scientific question.

---

# Important Strategic Constraint

Need to continue resisting:

* architecture-zoo drift,
* GPU-scale benchmark escalation,
* and leaderboard framing.

The strongest signal currently remains:

> operational geometry transition itself.

The simplicity of the perturbation experiments is part of what makes the current findings convincing.

---

# Immediate Next Questions

## Geometry Transition Analysis

Need to explicitly characterize:

* what geometric transformations occur when models move from:

  * cliff_allocator
    to:
  * smooth_allocator.

Especially:

* reachability curve shape,
* occupancy persistence,
* entropy/support changes,
* compression reduction,
* and threshold elasticity evolution.

---

## Focal Loss as a Controlled Follow-Up

Focal loss now appears scientifically justified as:

* a targeted imbalance-pressure perturbation,
  rather than:
* a generic neural improvement experiment.

However:

* it should remain tightly constrained,
* minimally implemented,
* and interpreted primarily through geometry transition analysis.

---

## Mechanism of Broad Allocation

The most important unresolved question may now be:

> what structural properties produce genuinely broad allocation behavior?

Bagged HDDT remains the strongest unresolved anomaly in the current framework.

That now feels increasingly central.



# Research Log — Allocation Morphology Emergence

## Date

2026-06-02

---

# Context

This line of work began as an attempt to understand why models with similar ranking and calibration performance exhibited materially different accessibility behavior under threshold policy changes in severely imbalanced settings.

The original working hypothesis for Paper 2 was that accessibility persistence and cliff behavior might emerge from latent representation topology:

```text
objective
    ↓
representation topology
    ↓
accessibility geometry
```

Specifically, we investigated whether minority-support connectedness, fragmentation, or voxel-like occupancy structure in latent space explained accessibility persistence and cliffiness.

---

# Topology Investigation

We implemented multiple topology diagnostics over minority support:

* kNN positive-support graphs
* radius-based positive-support graphs
* connected-component statistics
* giant-component fraction
* isolated-positive fraction
* component entropy
* mean and median component size

The expectation was that:

* connected minority support would produce smooth accessibility decay,
* fragmented support would produce cliff-like accessibility loss.

However, across the current synthetic MLP pilot:

* topology metrics varied,
* but topology consistently exhibited weak explanatory power for both survival level and survival shape.

In mediation-style regressions:

* topology-only models performed poorly,
* and topology added little or negative incremental value after allocation metrics were included.

Current evidence suggests topology is not the dominant first-order explanation in this setting.

This does *not* rule out topology entirely. It may:

* require harder datasets,
* require richer representation spaces,
* emerge more strongly under deployment shift,
* or become relevant only for specific model families.

But topology is currently a secondary hypothesis rather than the leading explanation.

---

# Emergence of Allocation Morphology

The research direction shifted after introducing explicit score-allocation diagnostics.

Initially, allocation was treated as a mostly one-dimensional concept related to concentration or quantization.

However, the current experiments strongly suggest allocation is multi-dimensional.

Two major axes emerged:

## 1. Allocation Breadth

Operational interpretation:

> How widely positive examples occupy score space.

Primary metrics:

* positive_histogram_entropy
* positive_effective_score_bins

Observed relationships:

* higher breadth strongly associated with reduced accessibility cliffiness,
* broader allocation produces smoother accessibility decay trajectories.

Key pilot result:

```text
positive_histogram_entropy
↔ minority_survival_cliffiness

r ≈ -0.80
```

This is currently one of the strongest observed relationships in the entire program.

---

## 2. Allocation Elevation

Operational interpretation:

> How much positive probability mass reaches high-confidence score regions.

Primary metrics:

* positive_top_bin_mass
* positive_max_bin_mass

Observed relationships:

* higher elevation strongly associated with higher survival level.

Key pilot result:

```text
positive_top_bin_mass
↔ minority_survival_auc

r ≈ 0.88
```

This is the single strongest correlation observed in the current pilot.

---

# Emerging Interpretation

Accessibility geometry now appears decomposable into at least two partially independent properties:

| Accessibility Property      | Allocation Driver |
| --------------------------- | ----------------- |
| Survival Level              | Elevation         |
| Survival Shape / Cliffiness | Breadth           |

This is a substantially richer interpretation than the earlier “single persistence object” framing.

The current working interpretation is:

```text
objective
    ↓
allocation morphology
    ↓
accessibility geometry
```

rather than:

```text
objective
    ↓
topology
    ↓
accessibility geometry
```

---

# Dropout Findings

Feature-dropout training variants were introduced as an approximation to feature-subspace sampling analogous to Random Forest feature sampling.

Expectation:

```text
dropout
    ↓
broader allocation
    ↓
better accessibility
```

Observed result was more nuanced.

Dropout generally:

* increased breadth,
* reduced cliffiness,
* but often reduced elevation and survival level.

This suggests a tradeoff:

| Effect         | Dropout Tendency |
| -------------- | ---------------- |
| Breadth        | Increase         |
| Cliffiness     | Decrease         |
| Elevation      | Often decrease   |
| Survival Level | Often decrease   |

This was especially visible relative to weighted BCE.

The result suggests:

> Broad allocation alone is insufficient.

Allocation breadth and allocation elevation appear separable.

---

# Allocation Morphology Space

The current pilot suggests allocation may naturally organize into a two-axis morphology space:

| Breadth | Elevation | Interpretation                    |
| ------- | --------- | --------------------------------- |
| High    | High      | Broad + confident allocation      |
| High    | Low       | Diffuse but weak allocation       |
| Low     | High      | Concentrated confident allocation |
| Low     | Low       | Weak concentrated allocation      |

This framing resembles a portfolio-style tradeoff between diversification and concentration.

The allocator taxonomy may ultimately become a taxonomy over regions of this morphology space.

---

# Current Working Thesis For Paper 2

Current evidence supports the following thesis:

> Under severe class imbalance, learning objectives induce distinct allocation morphologies. Allocation morphology appears to possess at least two axes—breadth and elevation—which separately govern accessibility shape and accessibility level.

---

# Current Confidence Levels

## High Confidence

* Accessibility geometry is not fully characterized by ranking metrics.
* Allocation metrics explain accessibility behavior better than topology metrics in the current pilot.
* Breadth and elevation behave differently and appear partially separable.
* Survival level tracks elevation strongly.
* Cliffiness tracks breadth strongly.

## Moderate Confidence

* Dropout behaves like a breadth-expanding mechanism.
* Weighted BCE behaves like an elevation-enhancing mechanism.
* Allocation morphology may provide a more explanatory framework than topology for accessibility geometry.

## Low Confidence / Open Questions

* Whether topology becomes important on harder or real datasets.
* Whether these findings generalize beyond sklearn MLPs.
* Whether the same morphology structure appears in boosted trees or transformers.
* Whether calibration manipulations directly alter morphology.
* Whether allocation morphology predicts deployment robustness.

---

# Immediate Next Experiment

Most promising next direction:

## Allocation Morphology Trajectories

Checkpoint training over epochs and measure:

* breadth
* elevation
* survival_auc
* cliffiness

for:

* BCE
* weighted BCE
* oversampling
* dropout

Goal:

> Understand how different learning objectives move through allocation morphology space during training.

This may become the central mechanistic result of Paper 2.


Research Log Entry

Project: Accessibility Geometry / Allocation Morphology
Date: 2026-06-03
Experiment: Dense Weighted BCE Feature-Dropout Sweep Across Skew Regimes

Objective

Evaluate whether feature-dropout acts as a breadth-expanding regularizer and determine how dropout affects allocation morphology and accessibility under extreme class imbalance.

This experiment was motivated by an analogy to Random Forest feature subspace sampling. The working hypothesis was:

Feature Dropout
    →
More Diverse Internal Representations
    →
Greater Allocation Breadth
    →
Improved Accessibility

The broader goal was to understand how learning objectives move through allocation morphology space and whether accessibility can be manipulated independently of conventional ranking metrics.

Experimental Design
Model Family

Weighted BCE classifier with feature-dropout augmentation.

Implementation note:

Dropout was approximated through Bernoulli feature masking during training.
Dense sweep used an sklearn MLP configuration with no hidden layers to keep the full grid computationally feasible.
This experiment should therefore be interpreted primarily as a feature-subspace perturbation study rather than a deep-network dropout study.
Sweep Parameters
Parameter	Values
Dropout Rate	0.00 – 0.50 (step 0.05)
Skew Ratios	25, 100, 500, 1000
Minority Count	100
Seeds	20
Total Fits	880
Recorded Metrics

Allocation Morphology

Breadth
Elevation

Accessibility

Minority Survival AUC
Minority Survival Cliffiness

Traditional Metrics

AUROC
Average Precision
Results
Observation 1: Accessibility Improves Consistently With Dropout

Across all four skew regimes:

Higher Dropout
    →
Higher Survival AUC

Best Survival AUC occurred at:

dropout = 0.50

for every skew ratio tested.

Examples:

Skew	Survival AUC @ 0.00	Survival AUC @ 0.50
25	0.960	0.967
100	0.930	0.966
500	0.905	0.962
1000	0.914	0.960

The effect becomes larger as skew increases.

Observation 2: Breadth Decreases Under Dropout

This directly contradicts the original hypothesis.

Across every skew regime:

Higher Dropout
    →
Lower Breadth

Examples:

Skew	Breadth @ 0.00	Breadth @ 0.50
100	0.560	0.381
500	0.675	0.406
1000	0.624	0.431

Instead of expanding allocation support, dropout appears to compress it.

Observation 3: Elevation Increases Under Dropout

Across every skew regime:

Higher Dropout
    →
Higher Elevation

Examples:

Skew	Elevation @ 0.00	Elevation @ 0.50
100	0.833	0.871
500	0.795	0.858
1000	0.822	0.856

This increase is highly consistent.

Observation 4: Cliffiness Increases Under Dropout

Unexpectedly:

Higher Dropout
    →
Higher Cliffiness

Examples:

Skew	Cliffiness @ 0.00	Cliffiness @ 0.50
100	0.654	0.977
500	0.642	0.978
1000	0.606	0.949

Accessibility level improves while accessibility shape becomes more concentrated.

Correlation Structure
Elevation vs Survival AUC
Skew	Correlation
25	0.866
100	0.840
500	0.883
1000	0.787

This is one of the strongest relationships observed in the entire accessibility program.

Interpretation:

Elevation
appears closely tied to
Accessibility Level

where level is represented by Survival AUC.

Breadth vs Cliffiness
Skew	Correlation
25	-0.199
100	-0.526
500	-0.709
1000	-0.670

The relationship strengthens substantially under severe imbalance.

Interpretation:

Breadth
appears closely tied to
Accessibility Shape

where shape is represented by cliffiness.

Emerging Interpretation

The original expectation was:

Dropout
    →
More Breadth
    →
Better Accessibility

The observed behavior is instead:

Dropout
    →
More Elevation
    →
Higher Survival

while simultaneously

Dropout
    →
Less Breadth
    →
More Cliffiness

This suggests that accessibility cannot be represented by a single scalar quantity.

Instead, the evidence increasingly supports:

Accessibility
    =
(Level, Shape)

where

Level
    ≈ Survival AUC

Shape
    ≈ Cliffiness

and these dimensions can move independently.

Implications For Paper 1

This experiment does not change the central Paper 1 claim:

Ranking quality is not sufficient to characterize accessibility.

However, it strengthens a future-work direction:

Accessibility appears to possess internal structure.

A possible framing:

Accessibility Level
    (how much minority support survives)

Accessibility Shape
    (how that support disappears)

The dropout sweep provides evidence that interventions can improve one dimension while worsening another.

This supports treating accessibility as a structured property of learning systems rather than a single quantity.

Open Questions
1. Why does dropout increase elevation?

Possible explanations:

Increased robustness to majority noise.
Implicit ensemble averaging.
More stable minority decision boundaries.
Reduced memorization of majority-specific features.
2. Why does breadth collapse?

Possible explanations:

Dropout concentrates minority mass into fewer high-confidence regions.
Feature masking removes weak minority-support pathways.
Accessibility is becoming taller rather than wider.
3. Is this fundamentally an information-density phenomenon?

Hypothesis:

Rare-event failures
    may occur because
    the model has insufficient information
    in portions of minority space.

Morphology may be measuring how learning algorithms allocate finite informational resources across minority regions.

4. Does oversampling interact with dropout?

The next major experiment should be:

Weighted BCE × Oversampling Grid

to determine whether:

weighting primarily drives elevation,
oversampling primarily drives breadth,
dropout amplifies one or both effects.

This experiment may provide the first clear mechanism map for allocation morphology.

Current Belief

Confidence is increasing that allocation morphology is measuring something real and structurally meaningful.

The specific mechanisms continue to surprise us.

The morphology variables themselves continue to reappear across objectives, skew regimes, and interventions.

The strongest emerging picture is:

Breadth
    ↔ Accessibility Shape

Elevation
    ↔ Accessibility Level

and learning objectives appear to navigate this morphology space along different trajectories rather than simply producing better or worse ranking models.




# 2026-06-04

## Density appears first-order, but fragmentation remains unresolved.

Current evidence supports density as the dominant accessibility axis, but the synthetic construction still allows density, support volume, and fragmentation to be partially entangled. The next test should hold local density/separability approximately fixed while varying the number of minority islands.




# Research Log Entry: Density Thresholds and Morphology Coordinates

## Date

2026-06-10

## Context

After the initial accessibility manuscript stabilized, Paper 2 exploration shifted toward understanding the mechanisms that produce accessibility morphology under severe class imbalance.

Prior experiments suggested that accessibility is not explained by ranking metrics alone and that allocation morphology may be governed by internal structural variables such as breadth, elevation, density, fragmentation, and local ambiguity.

This entry records the first strong evidence that accessibility may undergo a density-mediated transition in morphology space.

---

## Experiment

Dense density × separability threshold sweep.

### Design

The experiment varied:

* minority covariance as a proxy for local minority information density,
* centroid distance as a proxy for separability,
* model objective/perturbation:

  * `mlp_weighted_bce`
  * `mlp_weighted_bce_dropout_0_1`

Global imbalance was held fixed.

### Density Grid

```text
0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.60
```

### Separability Grid

```text
0.5, 1.0, 2.0, 4.0
```

### Core Metrics

Accessibility:

* minority survival AUC
* minority survival cliffiness

Morphology:

* breadth
* elevation

Conventional:

* AUROC
* average precision

---

## Main Observations

### 1. Density is the dominant accessibility axis

As minority covariance increased, accessibility degraded substantially.

Empirically:

* breadth increased strongly with minority covariance,
* elevation decreased strongly with minority covariance,
* minority survival AUC decreased strongly with minority covariance,
* cliffiness increased and then saturated.

This supports the interpretation that minority density is a first-order control variable for accessibility morphology.

---

### 2. Separability is secondary but real

Increasing centroid distance improved survival and elevation, and reduced breadth/cliffiness.

However, separability appeared weaker than density.

Interpretation:

* density controls whether minority support remains compact and accessible,
* separability helps retain elevation once density begins to degrade.

---

### 3. Accessibility collapse appears threshold-like

The density curves do not look purely linear.

There appears to be a transition region where morphology changes rapidly:

```text
roughly between minority_cov ≈ 0.20 and 0.80
```

Below this region:

```text
survival AUC ≈ 1
elevation ≈ 1
breadth ≈ 0
cliffiness ≈ 0
```

Within the transition region:

```text
survival AUC declines rapidly
elevation declines
breadth expands
cliffiness rises sharply
```

Beyond the transition region:

```text
survival AUC enters a degraded regime
elevation remains low
breadth remains high
cliffiness remains elevated
```

This suggests accessibility may exhibit a density-mediated phase transition.

---

## Morphology Interpretation

The experiment suggests that accessibility can be represented as movement through a morphology space defined by:

```text
breadth = spread of positive score allocation
elevation = concentration of positive mass at high accessibility
```

The observed trajectory is approximately:

```text
high density:
    low breadth
    high elevation
    high survival
    low cliffiness

transition density:
    increasing breadth
    falling elevation
    falling survival
    rising cliffiness

low density:
    high breadth
    low elevation
    degraded survival
    elevated cliffiness
```

This is one of the clearest empirical demonstrations so far that breadth and elevation may be fundamental accessibility coordinates.

---

## Relationship to Fragmentation Sweep

A separate fragmentation sweep suggested that fragmentation has a smaller effect on survival AUC and elevation than density, but can increase cliffiness and breadth.

This implies a possible hierarchy:

```text
density controls accessibility level
fragmentation controls accessibility instability
separability moderates elevation retention
```

Stated differently:

```text
density determines whether minority support is accessible at all,
fragmentation affects how stable that accessibility is,
separability affects how much elevation can be retained.
```

---

## Dropout Observation

Feature dropout did not behave as a simple accessibility-improving regularizer in this setting.

Instead, dropout tended to:

* increase breadth,
* decrease elevation,
* reduce survival AUC in many low-density regimes,
* sometimes reduce cliffiness.

This suggests dropout acts as a morphology perturbation rather than a uniformly beneficial intervention.

In this experiment, dropout often looked like a fragmentation or diffusion operator in morphology space.

---

## Emerging Hypothesis

Accessibility may be governed by a low-dimensional morphology state.

A candidate state representation is:

```text
M = (breadth, elevation)
```

where:

```text
elevation predicts accessibility level
breadth predicts accessibility instability / cliffiness
```

The next critical test is whether survival AUC and cliffiness can be predicted from breadth and elevation across multiple experiment families.

If so, breadth and elevation may be treated as operational state variables of accessibility.

---

## Working Claim

A tentative paper-facing statement:

> Accessibility failure under severe imbalance appears to undergo a density-mediated transition. As minority support becomes diffuse, positive score allocation moves from a compact high-elevation regime into a high-breadth, low-elevation regime, accompanied by survival degradation and increased cliffiness. Separability moderates this transition, while fragmentation appears to primarily affect instability rather than accessibility level.

---

## Open Questions

1. Is the density transition robust across model families beyond weighted BCE MLP variants?
2. Can breadth and elevation predict survival AUC across density, fragmentation, dropout, and objective perturbation experiments?
3. Does the transition persist under real enterprise rare-event datasets?
4. Is breadth/elevation morphology a sufficient state representation, or are additional axes needed?
5. Can this framework be connected to information density, support estimation, or selective prediction theory?

---

## Next Experiment

Run a ridge extraction experiment:

```text
Can survival AUC and cliffiness be predicted from breadth and elevation alone?
```

The analysis should pool rows from:

* density threshold sweep,
* fragmentation sweep,
* weighted dropout sweep,
* objective/topology allocation experiments if available.

If breadth/elevation explain accessibility outcomes across experiment families, then allocation morphology becomes a unifying framework rather than a dataset-specific descriptive artifact.

# Morphology Ridge Extraction: Accessibility Appears One-Dimensional, Cliffiness Requires Additional State

## Date

June 2026

## Context

A recurring hypothesis throughout the accessibility program has been that many apparently different interventions—density changes, separability changes, objective modifications, dropout, topology perturbations, and training trajectory effects—may ultimately be acting through a smaller set of latent accessibility state variables.

To investigate this possibility, a ridge extraction analysis was performed using the accumulated experimental corpus:

* Density Threshold Sweep
* Density × Separability Factorial
* Fragmentation Sweep
* Weighted Dropout Sweep
* MLP Objectives + Topology
* Allocation Trajectory Experiments

The goal was to determine whether the proposed morphology coordinates (breadth, elevation) could explain accessibility outcomes across experiment families.

---

## Result 1: Accessibility Collapses onto Morphology

A morphology-only model using breadth and elevation predicted minority survival AUC with surprisingly high accuracy:

* Morphology-only CV R² = 0.858
* Morphology + conventional metrics CV R² = 0.887

This performance generalized across experiment families despite substantial differences in how the underlying datasets were generated.

The implication is that many seemingly distinct perturbations may primarily affect accessibility through their movement in morphology space.

Conceptually:

Density
Separability
Dropout
Objectives
Fragmentation
Training Trajectory

→ Morphology State

→ Accessibility

rather than each perturbation directly determining accessibility.

---

## Result 2: Morphology Appears Nearly One-Dimensional

Several diagnostics suggest breadth and elevation lie on a common latent manifold.

Observed:

* Breadth/Elevation Correlation = -0.860
* PC1 Explained Variance = 0.930

Thus approximately 93% of morphology variation can be described by a single principal direction.

This suggests accessibility may largely evolve along a dominant accessibility ridge rather than occupying a genuinely two-dimensional state space.

Operationally:

Low Breadth / High Elevation
→ concentrated accessibility

High Breadth / Low Elevation
→ distributed accessibility

with most observed models moving along this tradeoff curve.

---

## Result 3: Cliffiness Is Not Morphology

The strongest negative result is also the most informative.

Morphology failed to predict minority survival cliffiness:

* Morphology-only CV R² = -0.184

while successfully predicting accessibility level.

This implies:

Accessibility Level
≈ Morphology

but

Accessibility Dynamics
≠ Morphology

and requires additional state.

The morphology coordinates capture where accessibility exists.

They do not capture how accessibility disappears as thresholds tighten.

---

## Result 4: Fragmentation Is the Largest Residual Family

The largest cliffiness residuals were associated with fragmentation experiments.

This is notable because fragmentation directly manipulates minority topology while approximately preserving overall density.

This observation suggests that topology contributes information not represented by breadth or elevation.

A provisional decomposition is:

Accessibility Level
= f(Morphology)

Accessibility Dynamics
= f(Morphology, Topology)

where topology includes quantities such as:

* Number of minority islands
* Largest connected component fraction
* Island entropy
* Positive isolation rate
* Component-size inequality

---

## Interpretation

The ridge extraction supports a new working hypothesis:

Morphology is the state space of accessibility.

Topology is the state space of accessibility dynamics.

If true, accessibility theory may naturally separate into:

1. Morphology

   * Breadth
   * Elevation

2. Topology

   * Connectivity
   * Fragmentation
   * Reachability structure

This provides a coherent explanation for why density, dropout, objectives, fragmentation, and training progression often produce similar accessibility outcomes despite very different mechanisms.

They may simply move models through a shared morphology manifold.

---

## Next Experiment

Topology-Augmented Morphology

Test whether adding topology coordinates can explain cliffiness residuals.

Candidate topology variables:

* n_islands
* largest_island_fraction
* island_entropy
* component_gini
* isolated_positive_fraction

Evaluate:

survival_auc ~ morphology

cliffiness ~ morphology + topology

A substantial improvement in cliffiness prediction would support the hypothesis that topology constitutes the missing accessibility state variable.


## Date 6/12/2026

Ridge/edge extraction on generating-parameter surfaces failed to explain accessibility dynamics. However, diagnostics suggest the extracted features were largely determined by grid construction and finite-difference artifacts. This experiment should not be interpreted as evidence against local-geometry explanations of cliffiness.

The failure of ridge extraction and topology augmentation appears to have been caused by model misspecification rather than lack of structure. Kernel smoothing reveals that cliffiness is highly predictable from morphology coordinates alone (CV R² ≈ 0.91), suggesting that accessibility dynamics are encoded as nonlinear geometry on the morphology manifold rather than requiring an additional state variable. Different experimental perturbations appear to trace distinct arcs across a common manifold.

The accessibility manifold is not a universal scalar coordinate. 
It is scalar-like for accessibility level, but two-dimensional and nonlinear for accessibility dynamics.


## Date 6/18/2026

Research Journal Entry
Accessibility Research Journal
Morphology Regimes Emerge as the Primary Explanatory Structure

Since the previous journal entry, the research program shifted from establishing the existence of accessibility morphology to identifying the underlying structure governing cliffiness behavior.

The initial objective was to determine whether cliffiness could be explained by a compact global equation derived from allocator morphology. Several competing hypotheses were explored.

Kernel Robustness

A kernel-based morphology model was subjected to extensive bandwidth sensitivity analysis.

Results showed:

Nonlinear morphology models consistently outperformed linear alternatives.
Performance remained stable across a wide range of RBF bandwidths.
Improvement was observed under grouped validation schemes rather than only random cross-validation.
Source-family holdout remained challenging, suggesting that extrapolation across allocator families is harder than interpolation within known families.

This established that the morphology signal is genuine and not an artifact of a particular kernel setting.

Conclusion

Cliffiness is robustly nonlinear with respect to morphology.

Morphology Equation Discovery

The next objective was to discover an explicit equation linking morphology coordinates to cliffiness.

Models examined:

linear terms
polynomial terms
log transforms
interaction terms
kernel regressors
random forests

The strongest compact equation obtained was:

log(breadth) + breadth + breadth² × elevation

with grouped CV performance around:

R² ≈ 0.60

Nonlinear machine learning models achieved higher performance:

R² ≈ 0.68–0.74

suggesting that a simple closed-form equation captures substantial but not complete structure.

Conclusion

Cliffiness is partially expressible through a compact nonlinear morphology equation, but significant residual structure remains.

Frontier Geometry Investigation

A competing theory proposed that cliffiness might primarily reflect distance from the feasible morphology frontier.

Several frontier-derived quantities were constructed:

frontier distance
normalized frontier position
frontier curvature
frontier width
vertical slack

Results were largely negative.

Frontier-only models produced negative grouped CV R² values.

Adding frontier quantities to morphology yielded only modest improvements.

Conclusion

Cliffiness is not primarily explained by distance to the morphology frontier.

Frontier geometry may contribute secondary signal but does not appear to be the organizing principle.

Morphology Atlas Construction

Attention then shifted from continuous equations toward discrete structure.

A morphology atlas was constructed using unsupervised clustering.

Key findings:

Clusters were reasonably stable across algorithms.
Between-cluster cliffiness variance greatly exceeded within-cluster variance.
Cluster identity strongly predicted cliffiness buckets.
Atlas clusters corresponded to recognizable regions of morphology space.

The resulting evidence suggested that allocator morphology may exhibit phase-like behavior rather than forming a smooth continuum.

Conclusion

Discrete morphology regimes appear to exist.

Regime Consolidation

The 14-cluster atlas was compressed into higher-level macro-regimes.

A four-regime solution preserved most predictive power while dramatically simplifying interpretation.

The resulting regimes were approximately:

Quantized Floor
low breadth
low elevation
low cliffiness

Associated with strongly quantized allocators.

Elevated Broad Plateau
high breadth
elevated survival
moderate cliffiness

Associated with broad, resilient allocators.

Broad-Flat High-Cliff Basin
high breadth
low elevation
high cliffiness

Associated with allocators exhibiting strong accessibility collapse.

Transitional / Mixed Region
boundary region between major phases
includes several near-zero-cliffiness substructures

Most importantly:

Global equations performed substantially worse than equations fitted within regimes.

This indicates that cliffiness is easier to explain locally than globally.

Conclusion

The strongest current theory is:

global cliffiness
    =
morphology regime selection
    +
simple local regime equation

This explanation outperformed:

linear morphology theory
global equation theory
frontier-distance theory

and remained consistent across multiple independent analyses.

Current Scientific Position

The research program has now moved from discovery into explanation.

The central question is no longer:

What causes cliffiness?

Instead it has become:

Why do these morphology regimes exist?

This represents a substantial maturation of the theory.

Current evidence suggests that accessibility behavior is governed by a small number of morphology phases, with cliffiness emerging from local dynamics within those phases rather than from a single global law.

# Research Journal — Accessibility Dynamics Emerges

## Date 6/23/2026


Today felt like a genuine turning point in the accessibility research program.

The early phase of this work was dominated by a simple question:

> Is accessibility real?

Paper 1 largely answered that question.

Across multiple model families, accessibility behaved as a distinct property not captured by ranking metrics, calibration metrics, or conventional performance measures. Models with similar AUROC, AP, and calibration could exhibit dramatically different threshold-mediated minority reachability.

That alone was an interesting observation.

The challenge for Paper 2 became understanding why.

Over the past several months, we explored a long sequence of hypotheses:

* Accessibility topology
* Accessibility capacity
* Quantization
* Support granularity
* Support construction
* Accessibility regimes
* Accessibility coordinates
* Occupancy and family structure

Many of these hypotheses failed in their strongest form.

At first this felt disappointing.

In retrospect, these failures were productive.

Support granularity collapsed into capacity.

Support construction largely compressed geometry rather than replacing it.

Simple quantized vs continuous regimes explained very little.

Accessibility coordinates remained strongly family structured.

Every failed hypothesis removed a possible explanation and narrowed the search space.

What remained was a surprisingly consistent picture:

Accessibility behaves less like a discrete property and more like a geometry.

Coordinates emerged as the most useful abstraction.

Models occupied coherent regions.

Nearby points exhibited similar morphology.

Distances in coordinate space correlated with morphology differences.

Accessibility increasingly looked like a state space.

The most surprising development came from the intervention work.

Initially the intervention vector hypothesis was blocked because no paired before/after intervention records existed.

After implementing paired intervention capture, we were able to recover 625 real intervention vectors corresponding to the CART → Random Forest transition.

The results were far stronger than expected.

The intervention vectors displayed:

* Mean cosine similarity ≈ 0.93
* Angular dispersion ≈ 0.07
* Strong cross-dataset stability
* Δcoordinate → Δcliffiness prediction R² ≈ 0.91

This is qualitatively different from previous results.

Previous results established structure.

This result suggests dynamics.

The coordinate system is no longer acting solely as a descriptive embedding.

It appears to capture predictable model movement.

The current evidence suggests:

Accessibility is a manifold.

Model families occupy regions of that manifold.

Interventions induce movement through that manifold.

Accessibility outcomes change as a consequence of that movement.

If these findings continue to hold, the central claim of Paper 2 may become:

> Accessibility coordinates do not merely summarize accessibility morphology. They define a state space in which model interventions induce predictable accessibility dynamics.

The next question is no longer whether accessibility has structure.

The next question is:

> What is the geometry of intervention flow?

Is bagging represented by a single global displacement vector?

Or does the direction and magnitude of bagging depend on where a model begins in accessibility space?

If intervention effects vary by location, accessibility may be better understood as a vector field than a fixed transformation.

That possibility feels particularly exciting because it would move the research program from describing accessibility to modeling accessibility dynamics.

For the first time, the research agenda feels less like a collection of disconnected empirical findings and more like a coherent theory beginning to emerge.
