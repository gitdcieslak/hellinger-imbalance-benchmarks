# Research Backlog

This document captures larger research directions, partially formed hypotheses, and future experimental ideas.

These are intentionally broader than the current HDDT imbalance study and may eventually become:
- standalone studies,
- papers,
- benchmark extensions,
- exploratory prototypes,
- or dead ends.

The purpose is preservation of potentially valuable research intuition before it is forgotten.

---

# Active Themes

Current recurring themes across multiple ideas:

- rare-event prediction,
- operational probability behavior,
- uncertainty under distribution shift,
- score allocation geometry,
- calibration vs ranking,
- detection of weak/rare signals,
- conservative posterior allocation,
- out-of-distribution behavior.

---

# Idea: Neural Classifiers for Extremely Rare Tabular Events

## Question

Are neural classifiers actually good at predicting extremely rare tabular events (e.g. 100:1 or worse imbalance)?

## Motivation

Modern neural architectures:
- often dominate large benchmark suites,
- but may struggle operationally under severe imbalance.

The current HDDT study suggests:
- ranking quality may diverge significantly from operational deployment behavior.

This raises the possibility that:
- neural tabular models may achieve strong AUROC,
- while still allocating posterior mass conservatively,
- leading to threshold collapse similar to boosting ensembles.

## Questions

- Do tabular neural models exhibit the same threshold-collapse behavior as XGBoost/LightGBM?
- Are neural posteriors even more conservative?
- Does focal loss meaningfully change allocation geometry?
- Do transformers differ from MLPs operationally?
- Does representation learning help rare-event allocation?
- How stable are neural probability estimates under extreme imbalance?

## Potential Models

- TabNet
- FT-Transformer
- SAINT
- MLP baselines
- NODE
- TabTransformer

## Potential Metrics

- threshold sweep behavior,
- posterior allocation analysis,
- calibration,
- separation metrics,
- minority recall recovery curves.

---

# Idea: Novel-Class Detection in ImageNet Models

## Question

Can high-performing ImageNet classifiers infer that an image belongs to a novel/unseen class?

## Motivation

Most classifiers assume:
- all inputs belong to one of the trained classes.

But real-world deployment often violates this assumption.

Interesting possibility:
- classifiers may internally encode uncertainty structure even when forced to output known labels.

## Questions

- Can score distributions identify unknown classes?
- Do entropy or margin metrics reveal novelty?
- Can embeddings separate unseen classes?
- How early in the network hierarchy does novelty emerge?
- Does calibration correlate with novelty detection?

## Potential Directions

- MSP (maximum softmax probability)
- Energy-based OOD scoring
- Embedding density estimation
- Contrastive representation analysis
- Open-set recognition literature

---

# Idea: Rare Signal Detection in Time Series

## Question

Can sequence models reliably detect meaningful but extremely rare signals embedded in noisy temporal data?

## Motivation

Many operational problems involve:
- sparse anomalies,
- weak precursors,
- rare regime changes,
- low signal-to-noise ratios.

Possible domains:
- finance,
- cybersecurity,
- medical monitoring,
- infrastructure telemetry.

## Questions

- Are RNNs/LSTMs actually good at rare-event temporal allocation?
- Does attention help weak-signal amplification?
- Does severe imbalance create the same threshold-collapse behavior seen in tabular classifiers?
- Can sequence models localize precursor structure?
- Are probability outputs meaningful operationally?

## Potential Architectures

- LSTM
- GRU
- Temporal CNN
- Transformer-based sequence models
- State-space models (Mamba/S4)

---

# Idea: Probability Allocation Geometry

## Status

This is currently the strongest emerging conceptual direction.

## Core Question

How do classifiers allocate posterior probability mass under severe imbalance?

## Emerging Observation

There may be a meaningful distinction between:

- ranking quality,
- calibration,
- operational threshold behavior,
- and probability allocation geometry.

Current evidence suggests:
- AUROC alone may substantially overstate deployable operational behavior under imbalance.

## Candidate Families

| Allocation Style | Candidate Models |
|---|---|
| Conservative allocators | RF, XGB, LightGBM |
| Moderate allocators | HDDT |
| Discrete allocators | CART |

## Potential Future Work

- calibration curves,
- reliability diagrams,
- allocation manifolds,
- prediction landscape analysis,
- posterior density visualization,
- geometric interpretation of threshold collapse.

---

# Idea: Prediction Landscape Analysis

## Question

What does the full prediction-space geometry of a classifier look like?

## Motivation

Current experiments observe:
- threshold collapse,
- posterior compression,
- conservative allocation.

But these are aggregate statistics.

A full landscape analysis might reveal:
- disconnected minority regions,
- compressed posterior manifolds,
- geometric differences between classifiers.

## Potential Approach

- generate representative synthetic grids,
- visualize posterior surfaces,
- inspect allocation continuity,
- compare decision topology across models.

---

# Meta Observation

A recurring theme across many of these ideas is:

> The distinction between ranking quality and operational uncertainty behavior.

This may ultimately become the broader long-term research direction connecting multiple projects.