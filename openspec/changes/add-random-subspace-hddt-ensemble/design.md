# Design

## Implementation

`hddt_forest` uses:

- `BaggingClassifier`
- base estimator: `HellingerDecisionTreeClassifier`
- `bootstrap=True`
- `bootstrap_features=False`
- deterministic `random_state`

Default smoke config uses `n_estimators=50` and `max_features=sqrt`.

## Max Features Resolution

`max_features` is resolved per split with feature count awareness:

- `sqrt` -> `max(1, floor(sqrt(n_features)))`
- `log2` -> `max(1, floor(log2(n_features)))`
- `null`/`None` -> `1.0`
- int -> clamped to `[1, n_features]`
- float -> passed through (>0)

The alias `random_subspace_hddt` points to the same factory.
