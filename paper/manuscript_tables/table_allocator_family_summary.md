# Table A — Allocator Family Summary

| Model | AUROC | Average Precision | Recall@0.50 | Recall@0.01 | Recovery | Accessibility Persistence | Operational Smoothness | Max Recall Jump | Recurring Pattern |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CART | 0.6249 | 0.1458 | 0.2910 | 0.2910 | 0.0000 | 0.9429 | 1.0000 | 0.0000 | quantized_allocator |
| HDDT | 0.7810 | 0.2465 | 0.2909 | 0.8518 | 0.5609 | 0.5181 | 0.2353 | 0.2690 | broad_allocator |
| Bagged HDDT (hddt_forest) | 0.8389 | 0.3434 | 0.0954 | 0.9978 | 0.9024 | 0.1252 | 0.1741 | 0.6100 | cliff_allocator |
| Random Forest | 0.8176 | 0.3573 | 0.1531 | 0.9109 | 0.7578 | 0.4275 | 0.2249 | 0.3458 | broad_allocator |
| XGBoost | 0.8010 | 0.3090 | 0.1209 | 0.9854 | 0.8645 | 0.2032 | 0.1753 | 0.4887 | cliff_allocator |
| LightGBM | 0.8479 | 0.3462 | 0.2026 | 0.7511 | 0.5485 | 0.6753 | 0.2624 | 0.2218 | conservative_allocator |
| MLP | 0.7068 | 0.2410 | 0.1346 | 0.7867 | 0.6522 | 0.4086 | 0.4178 | 0.4061 | cliff_allocator |

Sources:

- `reports/neural_mlp/legacy_benchmark_summary.csv`
- `reports/neural_mlp/legacy_threshold_sweep_summary.csv`
- `reports/neural_mlp/allocation_regime_summary.csv`

Notes:

- `Recovery = Recall@0.01 - Recall@0.50`.
- Accessibility Persistence is reported as `mean_fraction_below_0_01` from the existing regime summary artifact.
- `hddt_forest` is presented as “Bagged HDDT (hddt_forest)” for artifact traceability and naming consistency with current branch outputs.
