# Model Baselines Specification

## Requirements

### Requirement: Baseline registry

Models SHALL be defined through configuration rather than hard-coded experiment scripts.

Initial baseline families SHALL include:

- HDDT
- Bagged HDDT
- CART
- RandomForest
- XGBoost
- LightGBM

### Requirement: Optional dependencies

Optional models SHALL be skipped gracefully when their package is not installed.

### Requirement: Future baseline extension

The baseline registry SHOULD allow future addition of BalancedRandomForest and simple MLP baselines without changing experiment result schemas.
