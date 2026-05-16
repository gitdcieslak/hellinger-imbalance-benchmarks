# Design

## Optional Dependency Boundary

The model registry exposes `xgboost` and `lightgbm` ids, but imports their packages lazily inside model factories. This keeps `hib` importable without optional boosted-tree dependencies installed.

When an optional package is unavailable, the factory raises `OptionalDependencyUnavailable` with the model id and package name.

## Runner Behavior

The existing runner continues to consume model ids from configuration. If a configured optional model is unavailable, the runner skips that model and continues with the remaining configured models.

## Defaults

Boosted-tree defaults are intentionally modest for smoke testing:

- `n_estimators: 25`
- `max_depth: 3`
- `learning_rate: 0.1`
- deterministic `random_state`
- single-threaded `n_jobs: 1`
- binary classification objective

## Metrics

Boosted-tree baselines reuse the existing metrics infrastructure through the same `fit`, `predict`, and `predict_proba` classifier interface.
