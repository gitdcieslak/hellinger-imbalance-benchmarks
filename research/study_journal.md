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