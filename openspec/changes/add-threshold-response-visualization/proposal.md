# Add Threshold Response Visualization

## Summary

Add publication-oriented threshold-response plots from existing threshold sweep summaries to visualize operational metric behavior as thresholds relax from 0.50 to 0.01.

## Motivation

Both synthetic and legacy sweeps show threshold collapse and recovery patterns. Narrative-driven visualizations are needed to communicate the difference between ranking quality and deployment-threshold behavior.

## Scope

- Add threshold plotting module and CLI.
- Plot recall/F1/precision/balanced-accuracy versus threshold.
- Support legacy summaries directly and synthetic summaries when available.
- Save deterministic PNG and SVG outputs.

## Non-Goals

- No benchmark reruns.
- No threshold optimization or calibration.
- No new model families or dependencies beyond matplotlib.
