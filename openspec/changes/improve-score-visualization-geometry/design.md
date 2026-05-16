# Design

## Plot Variants

For each model and skew ratio:

- histogram full range `[0, 1]`
- histogram zoom `[0, 0.10]`
- histogram medium zoom `[0, 0.25]` (optional, enabled by default)
- ECDF full range `[0, 1]`
- ECDF zoom `[0, 0.10]`

## Bins

Zoomed histograms use finer low-score bins:

- low zoom bins in `[0, 0.10]`
- medium zoom bins in `[0, 0.25]`

Full range keeps existing benchmark bins.

## Normalization Modes

- `class_fraction`: each class histogram sums to 1 independently
- `count`: raw per-bin counts
- `density`: probability density

`class_fraction` is the default for comparability under skew.

## Titles and Labels

Histogram and ECDF titles include model id, skew ratio, and score range; histogram titles also include normalization mode. Y-axis labels reflect normalization semantics.

## Output Paths

Deterministic filenames are written under `reports/plots/` using explicit variant suffixes.
