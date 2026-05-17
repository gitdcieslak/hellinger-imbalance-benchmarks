# Add Legacy HDDT Dataset Ingestion

## Summary

Add local-only ingestion tooling for the original HDDT paper dataset archive: safe extraction, manifest generation, and dataset registry scaffolding.

## Motivation

The benchmark suite is ready to expand beyond synthetic data. We need reproducible handling of legacy HDDT datasets while keeping raw artifacts out of git.

## Scope

- Add documented `data/` layout for raw/extracted/processed legacy HDDT datasets.
- Add archive extraction utilities with path traversal protection.
- Add manifest generation to `data/processed/legacy_hddt/manifest.json`.
- Add initial `LEGACY_HDDT_DATASET_REGISTRY` scaffold.
- Add tests for extraction safety, hashing, format detection, and CLI defaults.

## Non-Goals

- No OpenML ingestion.
- No dataset preprocessing pipeline.
- No benchmark execution on legacy data yet.
- No assumptions about schema/labels across legacy files.
