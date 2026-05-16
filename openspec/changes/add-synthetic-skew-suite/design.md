# Design

## Synthetic Data

Synthetic datasets use two Gaussian classes with configurable majority/minority counts, class separation, noise, feature count, seed, and train/test split. The minority class is labeled `1`, and generation is deterministic under a fixed seed.

The initial smoke config covers 1:1, 10:1, and 100:1 skew ratios with intentionally small sample counts.

## Model Registry

The registry maps model ids to factory functions that accept a seed. The first runnable path includes `hddt`, `bagged_hddt`, `cart`, and `random_forest`.

Optional models remain out of scope for this change.

## Metrics

Metrics are computed from fitted estimators using `predict` and `predict_proba` when available. The first metric set includes AUROC, average precision, F1, precision, recall, balanced accuracy, and Brier score.

## Runner

The runner reads a YAML config, iterates over skew ratios, seeds, and models, then writes one JSONL result record per run. Each record includes dataset id, model id, seed, split id, skew ratio, metrics, package versions, and timestamp.

Generated result files are written under `results/` and remain gitignored.
