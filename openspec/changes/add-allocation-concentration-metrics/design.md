# Design

## Metrics

Per dataset/model/split, compute from predicted positive-class scores:

- `score_mean`, `score_std`, `score_min`, `score_max`
- fractions below: `0.001`, `0.005`, `0.01`, `0.05`
- fraction above: `0.50`
- `histogram_entropy` using fixed bins on `[0,1]`
- `effective_support_size = exp(histogram_entropy)`
- `gini_coefficient` over score allocations

Reuse existing fixed histogram bins from `hib.scores.HISTOGRAM_BINS`.

## Runner Integration

Add in `src/hib/runner.py`:

- `run_allocation_concentration_legacy(...)`
- `run_allocation_concentration_synthetic(...)`

Both paths reuse existing data loading/splitting/model-fitting and only add score-level post-processing.

## Reporting

Add allocation summary helpers in `src/hib/reporting.py`:

- flatten allocation records
- aggregate mean/std by `dataset_id` or `skew_ratio` and `model_id`
- include `n_runs`
- markdown rendering for grouped summaries

## CLI

Add `scripts/run_allocation_concentration.py`:

- `--source legacy|synthetic`
- `--datasets ...` (legacy)
- repeatable `--models`
- `--output`
- `--summary-csv`
- `--summary-md`

Defaults write:

- `results/allocation_concentration.jsonl`
- `reports/allocation_concentration_summary.csv`
- `reports/allocation_concentration_summary.md`
