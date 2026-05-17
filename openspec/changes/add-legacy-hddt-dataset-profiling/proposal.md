# Add Legacy HDDT Dataset Profiling

## Summary

Add dataset-level profiling for extracted legacy HDDT files to determine parseability, candidate targets, class structure, and imbalance characteristics before benchmark wiring.

## Motivation

Extraction and manifesting are in place. The next step is to understand what datasets are actually usable and how much preprocessing uncertainty remains.

## Scope

- Read `data/processed/legacy_hddt/manifest.json` and files in `data/extracted/legacy_hddt/`.
- Profile CSV-like formats (`.csv`, `.data`, `.txt`) with delimiter inference.
- Detect ARFF and mark as unsupported when scipy is unavailable.
- Infer candidate/selected target columns conservatively.
- Compute class counts and imbalance ratio when target is selected.
- Write outputs to JSON/CSV/Markdown reports.
- Add tests and CLI verification.

## Non-Goals

- No benchmark execution.
- No OpenML ingestion.
- No dataset downloads.
- No hardcoded global schema assumptions across legacy files.
