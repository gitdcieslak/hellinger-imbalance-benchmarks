# Tasks

## 1. Profiling Core

- [x] Add `src/hib/datasets/profile.py`.
- [x] Implement manifest-driven dataset profiling.
- [x] Add delimiter inference and conservative parsing for CSV-like files.
- [x] Add target inference and class/imbalance profiling.
- [x] Capture per-file parse failures without crashing full run.

## 2. Reporting and CLI

- [x] Add `scripts/profile_legacy_hddt_datasets.py`.
- [x] Write JSON profile output.
- [x] Write CSV profile report.
- [x] Write Markdown profile report with parsed and failed tables.

## 3. Tests and Verification

- [x] Add tests for parsing, delimiter inference, target uncertainty, and metrics.
- [x] Add tests for failed parse capture and report generation.
- [x] Run `python -m pytest`.
- [x] Run `python scripts/profile_legacy_hddt_datasets.py`.
