# Design

## Per-Run Metrics

For each model/skew/split run, compute class-conditional probability separation metrics including mean/median gaps, quantile gap metrics, fraction-above-negative-tail metrics, KS statistic, and histogram-based Hellinger distance.

## Histogram Hellinger Distance

Use existing fixed bins from score analysis. Convert positive and negative score histograms to probability vectors and compute Hellinger distance.

## Aggregation

Summaries aggregate over repeated 5x2 splits by `skew_ratio` and `model_id`, returning mean/std and `n_runs`.

## Outputs

- `results/synthetic_score_separation.jsonl`
- `reports/synthetic_score_separation_summary.csv`
- `reports/synthetic_score_separation_summary.md`

Markdown groups tables by skew ratio and sorts by model id.
