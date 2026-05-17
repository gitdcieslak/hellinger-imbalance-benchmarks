# Tasks

## 1. Legacy Threshold Runner

- [x] Add `run_legacy_threshold_sweep(...)`.
- [x] Add `run_legacy_threshold_sweep_from_config(...)`.
- [x] Reuse fixed threshold list and existing threshold metric computation.

## 2. Reporting and CLI

- [x] Add legacy threshold summary aggregation.
- [x] Add legacy threshold markdown summary rendering.
- [x] Add `scripts/run_legacy_threshold_sweep.py`.

## 3. Tests and Verification

- [x] Add tiny fixture runner test for threshold records.
- [x] Add summary and markdown generation tests.
- [x] Preserve loud positive-class failure behavior through existing loader tests.
- [x] Run `python -m pytest`.
- [x] Run legacy threshold sweep verification command.
