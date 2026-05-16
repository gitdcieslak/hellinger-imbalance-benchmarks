# Add Synthetic Skew Suite

## Summary

Implement the first runnable benchmark path using deterministic synthetic Gaussian skew datasets and required tree-based baselines.

## Motivation

The project needs a small, reproducible smoke benchmark before adding real datasets or larger benchmark campaigns. Synthetic Gaussian skew data provides controlled imbalance ratios and fast verification of the runner, model registry, metrics, and result artifact conventions.

## Scope

- Add synthetic Gaussian dataset generation for 1:1, 10:1, and 100:1 skew ratios.
- Add required model registry entries for HDDT, Bagged HDDT, CART, and RandomForest.
- Add core imbalance metric computation.
- Add a config-driven runner that emits JSONL result records.
- Add a smoke script writing `results/synthetic_smoke.jsonl`.
- Add tests for deterministic generation, model registry instantiation, metric computation, and JSONL output.

## Non-Goals

- Do not add real dataset downloads.
- Do not add neural models.
- Do not commit generated benchmark results.
- Do not implement large benchmark campaigns.
