# Synthetic Skew Suite Specification

## Requirements

### Requirement: Controlled synthetic datasets

The benchmark suite SHALL include synthetic binary classification datasets with controllable class skew.

Synthetic generation SHALL support:

- majority count
- minority count
- class separation
- noise
- random seed

### Requirement: Skew ladder

The suite SHALL support skew ratios including:

- 1:1
- 10:1
- 100:1
- 1000:1

### Requirement: Repeatability

Synthetic datasets SHALL be deterministic under fixed seeds.
