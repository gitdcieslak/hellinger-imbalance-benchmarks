# Add Legacy Threshold Sweep Runner

## Summary

Add fixed-threshold sweep evaluation for curated legacy HDDT datasets using the same threshold set and reporting approach used in synthetic experiments.

## Motivation

Synthetic results showed threshold-collapse behavior under imbalance. Legacy datasets now need the same threshold-response analysis to validate whether those patterns transfer to original-paper data.

## Scope

- Reuse curated legacy registry and loader.
- Reuse 5x2 stratified split, model registry, and fixed threshold metrics.
- Add legacy threshold sweep runner APIs and CLI.
- Add JSONL + CSV + Markdown summary outputs for legacy threshold runs.

## Non-Goals

- No threshold optimization or calibration.
- No OpenML ingestion.
- No multiclass conversion.
- No neural models or hyperparameter tuning.
