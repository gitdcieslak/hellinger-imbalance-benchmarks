# Metrics and Reporting Specification

## Requirements

### Requirement: Core imbalance metrics

Experiments SHALL compute:

- AUROC
- average precision / PR-AUC
- F1
- precision
- recall
- balanced accuracy
- Brier score

### Requirement: Result records

Each run SHALL emit a structured result record containing:

- experiment id
- dataset id
- model id
- seed
- split id
- metric values
- package versions
- timestamp

### Requirement: Generated reports

Report-ready tables and figures SHALL be generated under `reports/` and treated as generated artifacts unless intentionally versioned.
