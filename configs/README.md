# Configs

Configuration files should be deterministic, reviewable, and small enough to commit.

Expected future subdirectories:

- `datasets/` for dataset manifests
- `models/` for baseline model definitions
- `experiments/` for benchmark suite definitions
- `metrics/` for metric selections
- `reproducibility/` for seed and environment capture settings

Do not commit raw datasets or generated benchmark outputs here.
