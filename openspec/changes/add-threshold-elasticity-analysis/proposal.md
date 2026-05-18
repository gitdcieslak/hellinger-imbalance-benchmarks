# Add Threshold Elasticity Analysis

## Summary

Add interval-level and summary-level threshold elasticity metrics to quantify threshold cliffs, smoothness, and operational stability from existing threshold sweep summaries.

## Motivation

Threshold response and precision-recall trajectories show qualitative behavior differences. Elasticity metrics provide compact quantitative descriptors for cross-model and cross-dataset comparison.

## Scope

- Add elasticity analysis module.
- Add markdown/csv reporting integration.
- Add CLI for legacy/synthetic threshold summary inputs.
- Add tests for ordering, delta signs, summary extraction, and error handling.

## Non-Goals

- No benchmark reruns.
- No model changes or calibration.
- No threshold sweep computation changes.
