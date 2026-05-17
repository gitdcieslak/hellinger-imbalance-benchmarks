# Design

## Module

Add `src/hib/threshold_plots.py` with:

- `load_threshold_summary_csv(...)`
- `ordered_model_ids(...)`
- `plot_metric_vs_threshold(...)`
- `plot_threshold_response_panel(...)`
- `plot_threshold_response(...)`

## Plot Semantics

- x-axis: threshold in descending order (0.50 -> 0.01)
- y-axis: fixed [0, 1] for all target metrics
- one line per model
- optional +/- 1 std shaded bands when std columns exist
- titles/labels/grid/legend included for publication-readability

## File Naming

Deterministic filename format:

`{source}_{dataset_id}_{metric}_vs_threshold.{png|svg}`

Examples:

- `legacy_boundary_recall_vs_threshold.png`
- `legacy_boundary_recall_vs_threshold.svg`

## CLI

Add `scripts/plot_threshold_response.py` with options:

- `--summary-csv`
- `--source` (`legacy` or `synthetic`)
- `--datasets ...`
- `--metrics ...`
- `--output-dir`
- `--no-error-bands`

Defaults:

- source = legacy
- output dir = `reports/plots/threshold_response`
- metrics = recall, f1, precision, balanced_accuracy
- datasets = all present in summary
