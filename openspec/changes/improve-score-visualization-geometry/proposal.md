# Improve Score Visualization Geometry

## Summary

Improve score-distribution visualizations with zoomed views, class-aware normalization modes, and better low-probability geometry visibility under severe skew.

## Motivation

Current full-range histograms are hard to interpret when most probabilities cluster near zero. Zoomed low-score views and class-fraction normalization provide more faithful comparisons between positive and negative score geometry.

## Scope

- Add full and zoomed histogram variants.
- Add full and zoomed ECDF variants.
- Add normalization modes: `class_fraction`, `count`, and `density`.
- Add finer low-score bins for zoomed histogram views.
- Improve plot titles and axis labels with context and normalization.
- Update CLI controls for normalization and zoom variants.
- Add tests for deterministic naming, normalization behavior, zoom plot generation, and invalid options.

## Non-Goals

- Do not change model behavior.
- Do not change metric computation.
- Do not add new models, datasets, or calibration.
