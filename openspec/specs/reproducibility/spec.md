# Reproducibility Specification

## Requirements

### Requirement: Seed control

Every experiment SHALL specify all random seeds explicitly.

### Requirement: Environment capture

Each run SHALL capture relevant package versions, including:

- Python
- numpy
- scikit-learn
- hellinger-tree
- xgboost, if used
- lightgbm, if used

### Requirement: Generated artifacts

Generated result files SHALL be written under `results/` and excluded from default commits unless intentionally versioned.
