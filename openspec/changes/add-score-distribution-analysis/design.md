# Design

## Data Collection

For each run and split, the runner computes predicted probabilities on the test fold and separates scores by true class label. The analysis keeps only aggregated summary records, not per-row raw prediction files.

## Class-Conditional Statistics

Each class summary includes:

- mean and standard deviation of probabilities
- min and max
- quantiles p01, p05, p10, p25, p50, p75, p90, p95, p99

## Histogram Bins

Histograms use fixed bins:

- `[0.0, 0.01)`
- `[0.01, 0.05)`
- `[0.05, 0.10)`
- `[0.10, 0.25)`
- `[0.25, 0.50)`
- `[0.50, 1.00]`

Each record includes histogram counts for the associated class.

## Outputs

- JSONL aggregated score records under `results/`
- summary CSV: `reports/synthetic_score_summary.csv`
- summary markdown: `reports/synthetic_score_summary.md`

Markdown groups tables by skew ratio and includes class labels, quantile summaries, and histogram count columns.
