# Tasks

## 1. Splits

- [x] Add reusable `StratifiedShuffleSplit` generation.
- [x] Default to 5 repeats with 50/50 test size and deterministic split seed.
- [x] Verify splits preserve both classes when feasible.

## 2. Runner Integration

- [x] Integrate repeated stratified splits into synthetic runner.
- [x] Integrate repeated stratified splits into threshold sweep runner.
- [x] Emit split metadata fields in all run records.

## 3. Reporting

- [x] Aggregate over repeated runs.
- [x] Include `n_runs` in summary outputs.

## 4. Tests and Verification

- [x] Add split generator tests.
- [x] Add run-record split metadata assertions.
- [x] Add threshold repeated-split assertions.
- [x] Run `python -m pytest`.
- [x] Run `python scripts/run_synthetic_smoke.py --models configs/models/synthetic_smoke.yaml`.
- [x] Run `python scripts/summarize_results.py`.
- [x] Run `python scripts/run_threshold_sweep.py --models configs/models/synthetic_smoke.yaml`.
