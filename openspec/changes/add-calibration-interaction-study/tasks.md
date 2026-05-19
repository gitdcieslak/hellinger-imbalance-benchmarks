# Tasks

## 1. Calibration Pipeline

- [x] Add calibration study module.
- [x] Implement held-out validation calibration protocol (Platt + isotonic).
- [x] Emit raw/platt/isotonic threshold sweep records.

## 2. Metrics and Synthesis

- [x] Add ECE, Brier, reliability curves, and slope/intercept metrics.
- [x] Add calibration summary and regime persistence tables.
- [x] Add calibration interpretation markdown output.

## 3. Visualizations and CLI

- [x] Add reliability diagrams.
- [x] Add pre/post threshold overlays.
- [x] Add elasticity-shift scatter plots.
- [x] Add `scripts/run_calibration_interaction_study.py`.

## 4. Tests and Verification

- [x] Add tests for calibration metrics and tiny end-to-end record/artifact generation.
- [x] Run `python -m pytest`.
- [x] Run calibration interaction study CLI.
