# Table A (v2) — Allocator Family Summary (Persistence Remediated)

| Model | AUROC | Average Precision | Recall@0.50 | Recall@0.01 | Recovery | Threshold Occupancy Persistence | Operational Smoothness | Max Recall Jump | Recurring Pattern | Low-Score Mass (<0.01) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| CART | 0.6249 | 0.1458 | 0.2910 | 0.2910 | 0.0000 | 0.2910 | 1.0000 | 0.0000 | quantized_allocator | 0.9429 |
| HDDT | 0.7810 | 0.2465 | 0.2909 | 0.8518 | 0.5609 | 0.5006 | 0.2353 | 0.2690 | broad_allocator | 0.5181 |
| Bagged HDDT (hddt_forest) | 0.8389 | 0.3434 | 0.0954 | 0.9978 | 0.9024 | 0.5397 | 0.1741 | 0.6100 | cliff_allocator | 0.1252 |
| Random Forest | 0.8176 | 0.3573 | 0.1531 | 0.9109 | 0.7578 | 0.5597 | 0.2249 | 0.3458 | broad_allocator | 0.4275 |
| XGBoost | 0.8010 | 0.3090 | 0.1209 | 0.9854 | 0.8645 | 0.4915 | 0.1753 | 0.4887 | cliff_allocator | 0.2032 |
| LightGBM | 0.8479 | 0.3462 | 0.2026 | 0.7511 | 0.5485 | 0.4413 | 0.2624 | 0.2218 | conservative_allocator | 0.6753 |
| MLP | 0.7068 | 0.2410 | 0.1346 | 0.7867 | 0.6522 | 0.4868 | 0.4178 | 0.4061 | cliff_allocator | 0.4086 |

Sources:

- `reports/neural_mlp/legacy_benchmark_summary.csv`
- `reports/neural_mlp/legacy_threshold_sweep_summary.csv`
- `reports/neural_mlp/prediction_space_occupancy_summary.csv`
- `reports/neural_mlp/allocation_regime_summary.csv`

Definitions:

- `Recovery = Recall@0.01 - Recall@0.50`.
- `Threshold Occupancy Persistence` = model-level mean of `threshold_occupancy_persistence_mean` across datasets.
- `Low-Score Mass (<0.01)` = `mean_fraction_below_0_01` (kept as secondary concentration descriptor).
