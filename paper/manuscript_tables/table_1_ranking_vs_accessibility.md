# Table 1 — Ranking vs Accessibility Anchors

| model_id | AUROC | Average Precision | Recall@0.50 | Recall@0.01 | Recovery | Operational Smoothness | Max Recall Jump | Recurring Pattern |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mlp_bce | 0.7068 | 0.2410 | 0.1346 | 0.7867 | 0.6522 | 0.4178 | 0.4061 | cliff_allocator |
| mlp_oversampled | 0.8351 | 0.2989 | 0.4973 | 0.7041 | 0.2067 | 0.5391 | 0.0678 | smooth_allocator |
| mlp_weighted | 0.7732 | 0.2658 | 0.5789 | 0.8506 | 0.2717 | 0.6080 | 0.1783 | smooth_allocator |
| xgboost | 0.8010 | 0.3090 | 0.1209 | 0.9854 | 0.8645 | 0.1753 | 0.4887 | cliff_allocator |
| hddt | 0.7810 | 0.2465 | 0.2909 | 0.8518 | 0.5609 | 0.2353 | 0.2690 | broad_allocator |

Sources:

- MLP perturbation variants: `reports/neural_mlp_objective_perturbation_summary.md`, `reports/neural_mlp_objective_perturbation/legacy_benchmark_summary.csv`, `reports/neural_mlp_objective_perturbation/legacy_threshold_sweep_summary.csv`, `results/geometry_transition_analysis/geometry_transition_model_means.csv`
- XGBoost/HDDT anchors: `reports/neural_mlp/legacy_benchmark_summary.csv`, `reports/neural_mlp/legacy_threshold_sweep_summary.csv`, `reports/neural_mlp/allocation_regime_summary.csv`

Notes:

- `Recovery = Recall@0.01 - Recall@0.50`.
- Values are dataset means over the severe-imbalance slice used in manuscript v0.3.
