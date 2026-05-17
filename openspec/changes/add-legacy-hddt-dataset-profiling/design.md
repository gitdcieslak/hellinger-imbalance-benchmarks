# Design

## Profiling Input

The profiler reads:

- `data/processed/legacy_hddt/manifest.json`
- `data/extracted/legacy_hddt/`

Each manifest entry is profiled independently so one failure does not abort the full run.

## Supported Formats

- `csv`, `data`, `txt`: parsed with conservative delimiter inference over comma, tab, semicolon, and whitespace.
- `arff`: marked as `unsupported_arff` unless explicit ARFF parsing support is added later.
- Other formats: marked as `unsupported_format`.

## Target Inference

Candidate columns are ordered as:

1. name-based matches (`class`, `target`, `label`, `y`)
2. last column
3. first column

Chosen target criteria:

- distinct values between 2 and 20
- missingness <= 50%
- not detected as obviously continuous numeric values

If no candidate satisfies criteria, `chosen_target_column` remains null and notes record uncertainty.

## Class Profiling

When target is chosen, compute:

- `class_counts`
- `n_classes`
- `minority_class`, `minority_count`
- `majority_count`
- `imbalance_ratio = majority_count / minority_count`
- `positive_class_candidate` for binary targets (minority class)

## Outputs

- `data/processed/legacy_hddt/profile.json`
- `reports/legacy_hddt_dataset_profile.csv`
- `reports/legacy_hddt_dataset_profile.md`

Markdown includes parsed summary and failed/unsupported table plus notes on conservative target inference.
