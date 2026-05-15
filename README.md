# hellinger-imbalance-benchmarks

Reproducible benchmark infrastructure for studying class imbalance behavior across Hellinger Distance Decision Trees, modern tree ensembles, gradient boosting systems, and selected neural baselines.

This repository is separate from `hellinger-tree`. It depends on `hellinger-tree` as an installable package and does not implement the HDDT estimator itself.

## Initial Scope

- Python package skeleton under `src/hib/`
- Configuration conventions for datasets, models, experiments, metrics, and reproducibility
- Result artifact conventions under `results/` and report artifact conventions under `reports/`
- OpenSpec project structure and initial specifications
- Smoke-testable imports without requiring optional benchmark dependencies

## Planned Comparisons

- `HellingerDecisionTreeClassifier`
- Bagged HDDT
- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.ensemble.RandomForestClassifier`
- `imblearn.ensemble.BalancedRandomForestClassifier`, when `imbalanced-learn` is installed
- XGBoost
- LightGBM
- Simple MLP baseline later

## Repository Layout

```text
configs/      Configuration examples and conventions
notebooks/    Local exploratory notebooks, ignored except .gitkeep
openspec/     Project intent, specs, and proposed changes
reports/      Generated report artifacts, ignored except .gitkeep
results/      Generated benchmark results, ignored except .gitkeep
scripts/      Small operational scripts for future workflows
src/hib/      Importable benchmark infrastructure package
```

## Config Conventions

Configuration files should be text-based, reviewable, and reproducible. Prefer YAML files grouped by purpose:

- `configs/datasets/` for dataset manifests
- `configs/models/` for model baseline definitions
- `configs/experiments/` for experiment suites
- `configs/metrics/` for metric selections
- `configs/reproducibility/` for seed and environment capture settings

Configs should define what to run, not contain raw datasets or generated results.

## Result Artifact Conventions

Generated results belong under `results/` and are ignored by default. Each result record should include:

- experiment id
- dataset id
- model id
- seed
- split id
- metric values
- package versions
- timestamp

Generated report-ready tables and figures belong under `reports/` and are ignored by default unless intentionally versioned later.

## Development

Install the package in editable mode once dependencies are available:

```bash
python -m pip install -e .
```

Smoke-test the package import:

```bash
python -c "import hib; print(hib.__version__)"
```

No dataset downloads or large benchmark runs are implemented in the initial skeleton.
