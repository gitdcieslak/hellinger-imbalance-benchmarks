# Design

## Occupancy Metrics

Per dataset/model/split from positive-class scores:

- occupancy entropy and class-conditional occupancy entropy
- occupied-bin count
- posterior sparsity index
- occupancy density ratio
- threshold occupancy persistence
- minority occupancy compression ratio
- quantization score

Alongside reusable score stats and compact distribution artifacts:

- histogram counts
- ECDF points
- threshold occupancy trajectories

## Runner Integration

Add `run_prediction_space_occupancy_legacy(...)` in `src/hib/runner.py`.

This reuses existing loaders, model registry, split generation, and scoring paths without changing benchmark semantics.

## Reporting

Add occupancy summary aggregation and markdown rendering in `src/hib/reporting.py`.

Outputs:

- `results/prediction_space_occupancy.jsonl`
- `reports/prediction_space_occupancy_summary.csv`
- `reports/prediction_space_occupancy_summary.md`

## Visualizations

Add `src/hib/occupancy_plots.py`:

- class-conditional score histograms
- ECDF occupancy curves
- threshold occupancy trajectory plots
- quantization score panels

All plots saved as PNG and SVG for publication workflows.
