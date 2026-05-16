# Tasks

## 1. Config Loading

- [x] Add experiment config path loading.
- [x] Add model config path loading.
- [x] Add metric config path loading.
- [x] Preserve smoke defaults.
- [x] Support merging multiple model config files.

## 2. Validation

- [x] Raise clear errors for duplicate model ids.
- [x] Raise clear errors for missing config paths.
- [x] Raise clear errors for unknown model families.

## 3. CLI

- [x] Add `--models` to `scripts/run_synthetic_smoke.py`.
- [x] Add `--metrics` to `scripts/run_synthetic_smoke.py`.

## 4. Tests and Verification

- [x] Add model config override test.
- [x] Add boosted model JSONL inclusion test.
- [x] Add duplicate id test.
- [x] Add missing config path test.
- [x] Add unknown family test.
- [x] Run `python -m pytest`.
- [x] Run synthetic smoke with boosted config.
- [x] Verify xgboost and lightgbm appear in summary markdown.
