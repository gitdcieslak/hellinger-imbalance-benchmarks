# Add Legacy HDDT Benchmark Runner

## Summary

Add a curated legacy HDDT dataset registry and benchmark runner that applies the existing evaluation stack to selected original-paper binary datasets.

## Motivation

Profiling now provides provenance and parseability signals. The next step is controlled benchmark execution on a curated subset with explicit target and positive-class mappings.

## Scope

- Add curated registry entries for selected binary datasets.
- Add a loader for `.data` files with deterministic categorical encoding.
- Integrate legacy execution into runner using existing 5x2 stratified split flow, model registry, and metrics.
- Add a CLI to run selected legacy datasets and write JSONL + summary reports.
- Add tests for registry, loader semantics, strict positive-class handling, and tiny end-to-end runner behavior.

## Non-Goals

- No OpenML ingestion.
- No multiclass conversion.
- No neural models.
- No hyperparameter tuning.
