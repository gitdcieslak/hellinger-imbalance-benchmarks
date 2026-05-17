# Tasks

## 1. Curated Dataset Integration

- [x] Add curated legacy registry entries for selected binary datasets.
- [x] Add legacy loader with deterministic categorical encoding.
- [x] Enforce strict positive-class mapping with loud failure on missing class.

## 2. Runner and CLI

- [x] Integrate legacy benchmark execution into runner using existing split/model/metric stack.
- [x] Add legacy benchmark CLI script.
- [x] Add JSONL/CSV/Markdown output wiring for legacy runs.

## 3. Tests and Verification

- [x] Add tests for registry coverage and loader behavior.
- [x] Add tests for positive-class mapping, missing values, and deterministic encoding.
- [x] Add tiny end-to-end runner test with generated outputs.
- [x] Run `python -m pytest`.
- [x] Run legacy benchmark CLI verification command.
