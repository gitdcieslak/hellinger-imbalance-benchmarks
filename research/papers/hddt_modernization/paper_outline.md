# Ranking Is Not Deployment:
## Operational Probability Allocation Under Severe Class Imbalance

### Optional Subtitle
A Modern Reassessment of HDDT and Threshold Dynamics

---

# 1. Introduction

- Severe class imbalance as an operational problem
- Ranking quality versus deployment behavior
- Why AUROC/AP alone are insufficient
- Operational thresholding as a first-class concern
- Contributions

## Contributions
1. Reproducible imbalance benchmark infrastructure
2. Operational threshold sweep methodology
3. Precision–recall operational trajectory analysis
4. Allocation-geometry characterization across model families
5. Modern reassessment of HDDT-family behavior

---

# 2. Related Work

## 2.1 Class Imbalance Learning
- HDDT
- cost-sensitive learning
- resampling
- weighted ensembles

## 2.2 Ranking Metrics Under Imbalance
- AUROC
- Average Precision
- Precision–Recall analysis

## 2.3 Operational Evaluation of Classifiers
- threshold analysis
- deployment behavior
- calibration and operational risk

---

# 3. Benchmark Infrastructure and Reproducibility

## 3.1 Synthetic Imbalance Framework
- configurable skew generation
- repeated stratified evaluation

## 3.2 Legacy HDDT Dataset Suite
- curated dataset registry
- ingestion and preprocessing

## 3.3 Artifact and Experiment Management
- snapshotting
- reproducibility controls
- configuration system

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
- class-weighted boosting
- balanced random forests

## 4.2 Evaluation Metrics
- AUROC
- AP
- precision
- recall
- F1
- balanced accuracy
- Brier score

## 4.3 Threshold Sweep Protocol
- fixed operational thresholds
- repeated split evaluation
- threshold trajectory construction

---

# 5. Ranking Performance Under Imbalance

## 5.1 Synthetic Benchmarks
## 5.2 Legacy HDDT Datasets
## 5.3 Divergence Between Ranking and Deployment

Core observation:
- similar ranking metrics can conceal radically different operational behavior.

---

# 6. Operational Threshold Response

## 6.1 Recall Recovery Curves
## 6.2 Precision Collapse Curves
## 6.3 Threshold Sensitivity
## 6.4 Operational Stability

Key observations:
- abrupt vs smooth recall escalation
- precision degradation regimes
- threshold cliffs in boosted ensembles

---

# 7. Allocation Geometry and Operational Trajectories

## 7.1 Allocation Geometry
- score-space concentration
- probability mass allocation
- operational elasticity

## 7.2 Precision–Recall Operational Trajectories
- threshold paths through deployment space
- smooth vs discontinuous trajectories

## 7.3 Allocation Regimes
Potential categories:
- conservative allocators
- cliff allocators
- smooth allocators
- saturating allocators

## 7.4 Cross-Model Behavioral Patterns
- boosted ensembles
- HDDT-family models
- ensemble trees

---

# 8. Cross-Dataset Synthesis

## 8.1 Threshold Elasticity Across Datasets
## 8.2 Allocation Stability
## 8.3 Operational Tradeoff Archetypes

Potential synthesis metrics:
- recall gain per threshold relaxation
- precision retention
- allocation concentration
- operational smoothness

---

# 9. Discussion

- Ranking is not deployment
- Operational allocation as a deployment concern
- Implications for fraud/security/medical systems
- Why allocation geometry matters operationally
- Interpretability of threshold dynamics

---

# 10. Limitations and Future Work

- Binary classification only
- No calibration study yet
- No neural tabular models yet
- No temporal modeling
- Limited dataset diversity
- Fixed threshold schedules

---

# 11. Conclusion

- Operational behavior under imbalance is geometric and threshold-dependent
- Ranking metrics alone are insufficient
- Allocation trajectories expose meaningful differences between classifier families
- HDDT-family models exhibit distinctive operational behavior under severe imbalance