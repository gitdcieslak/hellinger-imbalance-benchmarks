# Add Score Distribution Analysis

## Summary

Add class-conditional predicted-probability distribution analysis for synthetic imbalance benchmarks.

## Motivation

Current results show threshold sensitivity under skew and cases where ranking metrics remain strong while default-threshold recall collapses. Score distribution summaries are needed to inspect compression, minority separation, and threshold-response behavior across model families.

## Scope

- Add score distribution collection and aggregation utilities.
- Add class-conditional probability statistics with quantiles.
- Add fixed-bin histogram summaries for positive and negative classes.
- Add runner integration and a score analysis script.
- Add CSV and markdown score summary outputs.
- Add tests for aggregation, quantiles, histograms, and report generation.

## Non-Goals

- Do not add plots.
- Do not add neural models.
- Do not add real datasets.
- Do not add calibration methods.
