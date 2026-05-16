# Design

## Diagnosis Approach

Use a separable 1:1 synthetic dataset with enough examples to ensure LightGBM has feasible split candidates. Verify predicted probabilities are non-constant and not collapsed at a single value.

## Parameter Changes

LightGBM smoke defaults are tuned for small synthetic folds:

- `n_estimators: 50`
- `max_depth: -1`
- `learning_rate: 0.1`
- `min_child_samples: 1`
- `min_data_in_leaf: 1`
- `objective: binary`
- deterministic seeds (`random_state`, `bagging_seed`, `feature_fraction_seed`, `data_random_seed`)
- `verbose: -1`

These changes reduce the risk of no-split behavior that can yield flat probabilities around 0.5 in tiny training sets.

## Validation

Regression test asserts LightGBM produces non-constant `predict_proba` outputs on a separable small dataset when LightGBM is installed.
