# Design

## Config Inputs

The synthetic runner accepts three config categories:

- experiment config: dataset generation, seeds, split, and output path
- model config: selected model ids and model metadata
- metric config: selected metric ids

The default smoke invocation continues to use the existing synthetic smoke experiment config, synthetic smoke model config, and core imbalance metric config.

## Model Config Merging

Model configs are YAML mappings under `models`. Multiple model config files may be supplied. Model ids are taken from the mapping keys in file order.

Duplicate model ids raise a `ValueError` before any benchmark work starts.

## Validation

Missing config paths raise `FileNotFoundError` with the missing path. Unknown model ids and unknown model families raise `ValueError` with the invalid value.

## Optional Models

Optional dependency behavior remains unchanged. If a configured optional model cannot be instantiated because its package is unavailable, the runner skips that model and continues.
