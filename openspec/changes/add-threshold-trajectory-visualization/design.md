# Design

## Plot Semantics

For each dataset (or synthetic skew group):

- x-axis: recall
- y-axis: precision
- one trajectory per model
- points ordered by threshold relaxation: `0.50 -> 0.25 -> 0.10 -> 0.05 -> 0.01`

Optional threshold annotation is enabled by default and can be disabled via CLI flag.

## Module Extensions

Extend `src/hib/threshold_plots.py` with:

- `trajectory_plot_paths(...)`
- `threshold_descending_values(...)`
- `plot_precision_recall_threshold_trajectory(...)`
- `plot_trajectory_panel(...)`
- `plot_threshold_trajectories(...)`

Validation requires columns: `model_id`, `threshold`, `precision_mean`, `recall_mean`, and either `dataset_id` or `skew_ratio`.

## Output Naming

Legacy:

- `{source}_{dataset_id}_precision_recall_trajectory.png`
- `{source}_{dataset_id}_precision_recall_trajectory.svg`

Synthetic:

- `{source}_skew_{skew_ratio}_precision_recall_trajectory.png`
- `{source}_skew_{skew_ratio}_precision_recall_trajectory.svg`

## CLI

Add `scripts/plot_threshold_trajectories.py` with options:

- `--summary-csv`
- `--source` (`legacy` or `synthetic`)
- `--datasets ...`
- `--output-dir`
- `--annotate-thresholds`
- `--no-annotate-thresholds`

Defaults:

- source = legacy
- output dir = `reports/plots/threshold_trajectories`
- datasets = all found in summary
- annotate thresholds = true
