# Tasks

## 1. Data Hygiene and Documentation

- [x] Update `.gitignore` rules for local dataset artifacts.
- [x] Add `data/.gitkeep`.
- [x] Document expected `data/` layout and workflow in `data/README.md`.

## 2. Extraction and Manifest Tooling

- [x] Add `src/hib/datasets/legacy.py` with safe tar extraction.
- [x] Add archive autodetection and `--force` overwrite behavior.
- [x] Add manifest generation with file hashes and format detection.
- [x] Add CLI `scripts/extract_legacy_hddt_datasets.py`.

## 3. Registry and Tests

- [x] Add `LEGACY_HDDT_DATASET_REGISTRY` scaffold.
- [x] Add tests for safe extraction and path traversal protection.
- [x] Add tests for manifest generation, hashing, and format detection.
- [x] Add tests for archive autodetection and force overwrite handling.
- [x] Run verification commands (`python -m pytest`, extraction CLI).
