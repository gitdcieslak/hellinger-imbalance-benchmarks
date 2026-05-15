# Dataset Manifest Specification

## Requirements

### Requirement: Dataset manifests

Each dataset used in experiments SHALL have a manifest file.

Manifest fields SHALL include:

- `dataset_id`
- `name`
- `source`
- `task_type`
- `target_column`
- `feature_columns`
- `positive_class`
- `n_samples`, when known
- `n_features`, when known
- `imbalance_ratio`, when known
- `license_notes`
- `download_instructions`
- `preprocessing_notes`

### Requirement: No large datasets committed

Raw datasets SHALL NOT be committed to the repository.

Dataset manifests MAY describe how to acquire or generate data.
