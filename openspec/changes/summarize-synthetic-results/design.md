# Design

## Input

The summarizer reads JSONL records emitted by the synthetic runner. Each record is expected to contain `skew_ratio`, `model_id`, and a nested `metrics` object.

## Aggregation

Records are flattened into a tabular dataframe and grouped by `skew_ratio` and `model_id`. For each core metric, the summary contains `<metric>_mean` and `<metric>_std` columns.

Single-observation standard deviations are written as `0.0` so smoke reports remain numeric and easy to compare.

## Outputs

The CSV output is a flat machine-readable table. The markdown output is a human-readable report with one simple table per skew ratio.

Both outputs are generated under `reports/`, which remains gitignored except for `.gitkeep`.
