# Tasks

## 1. Separation Utilities

- [x] Add `src/hib/separation.py`.
- [x] Add required separation metrics.
- [x] Add histogram Hellinger distance on fixed bins.

## 2. Runner and Reporting

- [x] Add separation records in runner.
- [x] Add separation summary CSV/markdown reporting.
- [x] Aggregate with mean/std and `n_runs` over repeated splits.

## 3. Script

- [x] Add `scripts/run_score_separation.py`.

## 4. Tests and Verification

- [x] Add clearly separated distribution test.
- [x] Add overlapping distribution test.
- [x] Add Hellinger distance bounds test.
- [x] Add fraction-above-quantile correctness test.
- [x] Add JSONL/report generation test.
- [x] Run `python -m pytest`.
- [x] Run `python scripts/run_score_separation.py --models configs/models/synthetic_smoke.yaml`.
