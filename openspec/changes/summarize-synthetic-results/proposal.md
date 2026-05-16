# Summarize Synthetic Results

## Summary

Add lightweight summarization for synthetic benchmark JSONL outputs, producing CSV and markdown reports grouped by skew ratio and model id.

## Motivation

The synthetic smoke benchmark now emits structured JSONL records. A small reporting path makes those records easier to inspect without adding new models, real datasets, or large generated artifacts.

## Scope

- Read `results/synthetic_smoke.jsonl`.
- Aggregate metrics by skew ratio and model id.
- Compute mean and standard deviation for core imbalance metrics.
- Write `reports/synthetic_smoke_summary.csv`.
- Write `reports/synthetic_smoke_summary.md` with simple tables grouped by skew ratio.
- Add tests for JSONL reading, aggregation columns, markdown output, and CSV output.

## Non-Goals

- Do not add new models.
- Do not add real datasets.
- Do not commit generated result or report files.
