# Initialize Benchmark Suite

## Summary

Create the initial repository skeleton for `hellinger-imbalance-benchmarks`, including package metadata, importable Python package structure, configuration and artifact directories, and OpenSpec documentation.

## Motivation

The benchmark evidence for Hellinger Distance Decision Trees should be managed separately from the `hellinger-tree` implementation package. This separation keeps method implementation independent from experiment configuration, generated results, and reporting artifacts.

## Scope

- Add Python package skeleton under `src/hib/`.
- Add project metadata in `pyproject.toml`.
- Document repository purpose and conventions in `README.md`.
- Add directories for configs, scripts, notebooks, results, and reports.
- Add OpenSpec project, specs, and initialization change documents.

## Non-Goals

- Do not implement full benchmark experiments.
- Do not download datasets.
- Do not commit generated benchmark results.
- Do not implement the HDDT estimator in this repository.
