# Design

## Curated Registry

Add `src/hib/datasets/legacy_registry.py` with explicit dataset entries for:

- breast-w, pima, sonar, ion, mushroom
- oil, satimage, sick, boundary, compustat
- phoneme, page, cam, covtype

Each entry includes dataset path, target column, positive class, task type, source group, and profile-derived shape/imbalance metadata.

## Loader

Add `src/hib/datasets/legacy_loader.py` to:

- read `.data` files from extracted legacy directory
- split target and features
- enforce configured positive class presence (fail loudly if missing)
- map labels to binary `{0,1}` with configured positive class as `1`
- convert `?` and empty tokens to missing values
- impute numeric columns by median
- ordinal-encode categorical columns with deterministic sorted-category mapping

## Runner Integration

Add `run_legacy_hddt_benchmark(...)` in `src/hib/runner.py`:

- reuses model factory, split-dependent params, and metrics evaluation
- reuses 5x2 stratified split infrastructure
- writes records compatible with existing JSONL writer
- includes dataset metadata and source group in each record

Add legacy summary helpers for CSV/Markdown aggregation grouped by dataset and model.

## CLI

Add `scripts/run_legacy_hddt_benchmark.py` with options:

- `--datasets` (list)
- `--models` (repeatable)
- `--output`
- `--summary-csv`
- `--summary-md`

Defaults write:

- `results/legacy_hddt_benchmark.jsonl`
- `reports/legacy_hddt_benchmark_summary.csv`
- `reports/legacy_hddt_benchmark_summary.md`
