# Configurable Model Config Loading

## Summary

Allow synthetic experiment runners to load experiment, model, and metric config files explicitly, including one or more model config files.

## Motivation

Boosted-tree model configs exist, but the runner still effectively uses the original smoke-test model set unless code is changed. Model selection should be config-driven so benchmark invocations can choose standard trees, boosted trees, or a merged set of model config files.

## Scope

- Add runner loading for experiment config path, model config path(s), and metric config path.
- Preserve existing synthetic smoke defaults.
- Add `--models` and `--metrics` CLI arguments to `scripts/run_synthetic_smoke.py`.
- Merge multiple model config files.
- Raise clear errors for duplicate model ids, missing config paths, and unknown model families.
- Add tests for override behavior, boosted model inclusion, duplicate ids, missing paths, and unknown families.

## Non-Goals

- Do not add neural models.
- Do not add real datasets.
- Do not add large benchmark runs.
- Do not commit generated results.
