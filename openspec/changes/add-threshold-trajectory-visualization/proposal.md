# Add Threshold Trajectory Visualization

## Summary

Add precision-recall threshold trajectory plots from existing threshold sweep summaries to visualize how models move through operational space as thresholds relax.

## Motivation

Threshold-vs-metric plots are useful, but deployment decisions trade precision against recall simultaneously. Trajectory plots make this tradeoff explicit and support the framing that ranking quality and deployment behavior diverge under severe imbalance.

## Scope

- Extend threshold plotting utilities with precision-recall trajectory plots.
- Add CLI for generating trajectory plots from legacy/synthetic threshold summary CSVs.
- Save deterministic PNG and SVG outputs.
- Add tests for ordering, filenames, filtering, and error handling.

## Non-Goals

- No benchmark reruns.
- No calibration or threshold tuning.
- No new dependencies beyond matplotlib.
