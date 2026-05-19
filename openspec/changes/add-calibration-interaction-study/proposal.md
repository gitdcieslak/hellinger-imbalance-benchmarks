# Add Calibration Interaction Study for Operational Allocation Geometry

## Summary

Add a calibration interaction study that compares raw, Platt-calibrated, and isotonic-calibrated probabilities under the existing threshold-sweep framework to test regime persistence under calibration.

## Motivation

The study now distinguishes ranking quality from deployment behavior. Calibration may improve reliability while leaving operational allocation geometry largely intact. This change quantifies that interaction.

## Scope

- add leakage-safe post-hoc calibration on held-out validation slices
- run threshold sweeps for raw/platt/isotonic probabilities
- compute calibration metrics (ECE, Brier, slope/intercept, reliability curves)
- compute allocation + elasticity shifts and regime persistence summaries
- generate publication-oriented comparative plots and interpretation summary

## Non-Goals

- no model-zoo expansion
- no hyperparameter optimization
- no benchmark split changes
- no replacement of existing operational metrics
