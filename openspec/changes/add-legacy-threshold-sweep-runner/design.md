# Design

## Runner APIs

Add to `src/hib/runner.py`:

- `run_legacy_threshold_sweep(...)`
- `run_legacy_threshold_sweep_from_config(...)`

Execution reuses:

- curated legacy dataset registry
- legacy loader
- stratified repeated split generation
- model creation and split-dependent parameter handling
- fixed threshold sweep (`0.01, 0.05, 0.10, 0.25, 0.50`)

Each threshold record includes dataset/model/split metadata, class counts, and threshold metrics.

## Reporting

Add legacy threshold summary helpers in `src/hib/reporting.py`:

- aggregate by `dataset_id`, `source_group`, `model_id`, `threshold`
- compute mean/std for `precision`, `recall`, `f1`, `balanced_accuracy`
- include `n_runs`
- render markdown grouped by dataset with rows sorted by model and threshold

## CLI

Add `scripts/run_legacy_threshold_sweep.py` with options:

- `--datasets` (space-separated list)
- repeatable `--models`
- `--output`
- `--summary-csv`
- `--summary-md`

Defaults run all curated legacy datasets and default synthetic smoke model config.
