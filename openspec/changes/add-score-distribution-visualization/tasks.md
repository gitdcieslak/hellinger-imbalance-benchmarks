# Tasks

## 1. Plot Utilities

- [x] Add `src/hib/plots.py`.
- [x] Add fixed-bin histogram plotting with class overlays.
- [x] Add ECDF plotting for positive/negative classes.
- [x] Add optional threshold overlays.

## 2. Runner and CLI

- [x] Add in-memory score collection for plotting.
- [x] Add `scripts/run_score_plots.py` with `--config`, `--models`, and `--metrics`.
- [x] Write deterministic plot files under `reports/plots/`.

## 3. Tests and Verification

- [x] Add plot generation test.
- [x] Add deterministic output path test.
- [x] Add empty-score handling test.
- [x] Add histogram normalization safety test.
- [x] Run `python -m pytest`.
- [x] Run `python scripts/run_score_plots.py --models configs/models/synthetic_smoke.yaml`.
