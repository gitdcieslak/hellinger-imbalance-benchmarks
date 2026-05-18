# Add Allocation Regime Summary

## Summary

Add a descriptive synthesis layer that combines allocation concentration and threshold elasticity outputs into compact model-level operational regime summaries.

## Motivation

The project now focuses on operational allocation geometry under severe imbalance. We need interpretable cross-dataset regime synthesis to support paper framing without introducing formal clustering.

## Scope

- Add regime synthesis module and heuristic labels.
- Aggregate model-level regime summary metrics from existing report outputs.
- Generate csv/markdown/json summaries.
- Add lightweight regime scatter plots using matplotlib.
- Add CLI and tests.

## Non-Goals

- No benchmark reruns.
- No ML clustering/embedding/statistical significance methods.
- No new model families.
