# Model Naming Audit (v0.9)

## Naming Rule

- Manuscript-facing names use human-readable family names.
- Internal implementation identifiers are used only where reproducibility needs explicit ID mapping.

| Internal ID | Manuscript Name |
| --- | --- |
| `cart` | CART |
| `hddt` | HDDT |
| `hddt_forest` | Bagged HDDT |
| `random_forest` | Random Forest |
| `xgboost` | XGBoost |
| `lightgbm` | LightGBM |
| `mlp` | MLP |
| `mlp_bce` | MLP (BCE baseline) |
| `mlp_oversampled` | MLP (oversampled) |
| `mlp_weighted` | MLP (weighted) |

## Applied Change

- In `paper/manuscript_v0_9.md`, the implementation note now states that `hddt_forest` refers to Bagged HDDT throughout the manuscript.
- Main narrative uses "Bagged HDDT" consistently; `hddt_forest` appears only in the mapping note.
