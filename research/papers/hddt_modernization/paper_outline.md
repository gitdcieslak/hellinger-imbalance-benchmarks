# Ranking Is Not Deployment:
## Operational Allocation Geometry Under Severe Class Imbalance

### Optional Subtitle
A Modern Reassessment of Probability Allocation Behavior in Imbalanced Classification

---

# 1. Introduction

## 1.1 Severe Class Imbalance as an Operational Problem
- fraud
- intrusion detection
- medical diagnosis
- rare-event industrial systems

## 1.2 Ranking Quality Versus Deployment Behavior
Core argument:
- AUROC and Average Precision measure ranking,
- but deployment requires operational allocation decisions under thresholds.

Motivating observation:
- models with strong ranking metrics may exhibit catastrophic recall collapse at operational thresholds.

## 1.3 Allocation Geometry Hypothesis
Emerging thesis:
- classifier families allocate posterior probability mass differently under imbalance,
- producing distinct operational behaviors.

Introduce:
- allocation concentration,
- threshold elasticity,
- operational trajectories,
- regime structure.

## 1.4 Contributions

1. Reproducible imbalance benchmark infrastructure
2. Legacy HDDT dataset modernization
3. Operational threshold-sweep evaluation methodology
4. Allocation concentration and elasticity metrics
5. Precision–recall operational trajectory analysis
6. Empirical identification of operational allocation regimes
7. Modern reassessment of HDDT-family operational behavior

---

# 2. Related Work

## 2.1 Class Imbalance Learning
- HDDT
- cost-sensitive learning
- weighted ensembles
- boosting under imbalance
- resampling

## 2.2 Ranking Metrics Under Imbalance
- AUROC
- AP
- PR curves
- ranking vs classification

## 2.3 Calibration and Operational Decision-Making
- reliability
- thresholding
- operational risk
- posterior calibration

## 2.4 Operational Evaluation of Classifiers
Gap:
- limited study of allocation geometry and threshold dynamics under severe imbalance.

---

# 3. Benchmark Infrastructure

## 3.1 Synthetic Imbalance Framework
- configurable skew generation
- repeated stratified evaluation

## 3.2 Legacy HDDT Dataset Suite
- curated dataset registry
- ingestion pipeline
- preprocessing controls

## 3.3 Artifact and Experiment Infrastructure
- snapshotting
- reproducibility controls
- experiment manifests
- threshold reporting system

---

# 4. Experimental Design

## 4.1 Model Families

### Classical Trees
- CART
- HDDT

### Ensemble Trees
- Random Forest
- Bagged HDDT

### Boosted Ensembles
- XGBoost
- LightGBM

### Weighted Variants
- balanced RF
- weighted boosting

## 4.2 Evaluation Metrics

### Ranking Metrics
- AUROC
- Average Precision

### Operational Metrics
- precision
- recall
- F1
- balanced accuracy
- Brier score

### Allocation Metrics
- entropy
- effective support size
- Gini concentration
- threshold elasticity
- operational smoothness

## 4.3 Threshold Sweep Protocol
- fixed thresholds:
  - 0.50
  - 0.25
  - 0.10
  - 0.05
  - 0.01
- repeated 5x2 evaluation
- operational trajectory construction

---

# 5. Ranking Performance Under Imbalance

## 5.1 Synthetic Benchmarks
## 5.2 Legacy HDDT Datasets
## 5.3 Divergence Between Ranking and Deployment

Central observation:
- strong ranking does not guarantee deployable minority recall.

---

# 6. Threshold Dynamics Under Imbalance

## 6.1 Recall Recovery Curves
## 6.2 Precision Collapse Curves
## 6.3 Threshold Elasticity
## 6.4 Operational Stability

Key phenomena:
- threshold cliffs,
- abrupt recall escalation,
- conservative posterior allocation,
- threshold-invariant allocators.

---

# 7. Allocation Geometry

## 7.1 Probability Allocation Concentration
- entropy
- support size
- score compression

## 7.2 Prediction-Space Occupancy
- posterior mass occupancy
- minority allocation sparsity
- operational support regions

## 7.3 Operational Elasticity
- recall elasticity
- precision elasticity
- threshold sensitivity

## 7.4 Allocation Smoothness
- continuous vs discontinuous threshold response
- operational regime transitions

---

# 8. Precision–Recall Operational Trajectories

## 8.1 Threshold-Parameterized Trajectories
- operational movement through PR space
- deployment paths under threshold relaxation

## 8.2 Cliff Allocators
Observed behavior:
- abrupt recall recovery,
- severe threshold sensitivity.

## 8.3 Conservative Allocators
Observed behavior:
- compressed minority probability allocation,
- strong ranking with operational collapse.

## 8.4 Quantized Allocators
Observed behavior:
- threshold-invariant discrete allocation.

## 8.5 Broad Allocators
Observed behavior:
- smoother allocation behavior,
- less catastrophic threshold collapse.

---

# 9. Operational Allocation Regimes

## 9.1 Regime Synthesis

Potential regimes:
- conservative allocators
- cliff allocators
- quantized allocators
- broad allocators
- smooth allocators

## 9.2 Cross-Model Behavioral Patterns
- boosting ensembles
- HDDT-family models
- tree ensembles

## 9.3 Cross-Dataset Stability
- pathological imbalance
- transitional regimes
- healthy imbalance

---

# 10. Discussion

## 10.1 Ranking Is Not Deployment
Core argument:
- ranking metrics substantially underdescribe operational behavior.

## 10.2 Allocation Geometry as a Deployment Property
- deployment stability
- operational controllability
- threshold robustness

## 10.3 Implications for High-Risk Domains
- fraud
- cybersecurity
- medicine
- industrial rare-event detection

## 10.4 Why HDDT Still Matters
Not necessarily because:
- it dominates ranking metrics,

but because:
- it may produce materially different operational allocation behavior.

---

# 11. Limitations and Future Work

- binary classification only
- no neural tabular models
- no temporal sequence modeling
- no calibration-interaction study yet
- no full prediction-space occupancy visualization yet
- fixed threshold schedules
- limited dataset diversity

Future directions:
- calibration geometry interaction
- prediction-space occupancy analysis
- multiclass imbalance
- sequence anomaly detection
- neural tabular architectures

---

# 12. Conclusion

- severe imbalance induces distinct operational allocation geometries,
- ranking metrics alone are insufficient,
- threshold dynamics reveal hidden deployment behavior,
- classifier families exhibit distinct allocation regimes,
- HDDT-family methods remain operationally interesting under severe imbalance.