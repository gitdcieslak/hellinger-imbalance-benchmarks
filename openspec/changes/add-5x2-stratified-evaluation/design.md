# Design

## Split Generation

A new reusable split helper creates repeated stratified train/test splits from labels using `StratifiedShuffleSplit`.

For repeat `i`, the split seed is `base_split_seed + i` to preserve deterministic behavior while making each repeat explicit in output metadata.

## Defaults

Synthetic experiment defaults are:

- `n_repeats: 5`
- `test_size: 0.5`
- `split_seed: 0`

These defaults apply to the standard synthetic runner and threshold sweep runner.

## Run Record Metadata

Each emitted run record includes:

- `repeat_id`
- `split_id`
- `split_seed`
- `train_n`
- `test_n`
- `train_pos`
- `train_neg`
- `test_pos`
- `test_neg`

## Reporting

Summary aggregation groups stay the same (`skew_ratio`, `model_id`, and threshold where relevant), but now aggregate over repeated splits and include `n_runs` to make sample size explicit.
