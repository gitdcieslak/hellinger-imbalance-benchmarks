# Add Score Separation Metrics

## Summary

Add compact class-separation metrics derived from predicted probabilities for synthetic imbalance benchmarks.

## Motivation

Score distribution analysis indicates that model families distribute probability mass differently under skew. A compact separation summary is needed to compare minority/majority score overlap and threshold sensitivity more directly.

## Scope

- Add score separation metric utilities.
- Add separation result records integrated with repeated stratified split evaluation.
- Add CSV/markdown summary reporting.
- Add a runner script for separation analysis.
- Add tests for metric behavior, bounds, and report generation.

## Non-Goals

- Do not add plots.
- Do not add calibration.
- Do not add neural models.
- Do not add real datasets.
