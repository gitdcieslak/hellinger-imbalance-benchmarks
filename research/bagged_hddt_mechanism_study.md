# Bagged HDDT Mechanism Study (v1)

## Scope

This note compares four allocator families (`hddt`, `hddt_forest`, `random_forest`, `lightgbm`) to explain why Bagged HDDT combines high low-threshold reachability with weak high-threshold recall.

Sources used:

- `paper/manuscript_tables/table_allocator_family_summary_v2.md`
- `reports/neural_mlp/legacy_threshold_sweep_summary.csv`
- `reports/neural_mlp/threshold_elasticity_summary.csv`
- `reports/neural_mlp/prediction_space_occupancy_summary.csv`

## Evidence Snapshot

From `table_allocator_family_summary_v2.md`:

- Bagged HDDT (`hddt_forest`) has the highest recovery (`0.9024`) and largest max recall jump (`0.6100`).
- Bagged HDDT is least smooth (`operational_smoothness = 0.1741`) among the four studied families.
- Bagged HDDT persistence is moderate-high (`0.5397`), above HDDT (`0.5006`) and LightGBM (`0.4413`), below Random Forest (`0.5597`).

Threshold trajectory means (`legacy_threshold_sweep_summary.csv`):

- `hddt_forest`: Recall@0.50 `0.0954`, @0.25 `0.2249`, @0.10 `0.4845`, @0.05 `0.8959`, @0.01 `0.9978`.
- `random_forest`: `0.1531`, `0.3387`, `0.6361`, `0.7594`, `0.9109`.
- `hddt`: `0.2909`, `0.3155`, `0.4486`, `0.5961`, `0.8518`.
- `lightgbm`: `0.2026`, `0.2788`, `0.4225`, `0.5516`, `0.7511`.

Max-jump interval counts (`threshold_elasticity_summary.csv`, 5 datasets/model):

- `hddt_forest`: `0.10->0.05` (2), `0.25->0.10` (2), `0.50->0.25` (1), `0.05->0.01` (0).
- `hddt`: mostly late jump at `0.05->0.01` (4/5).
- `lightgbm`: mostly late jump at `0.05->0.01` (3/5).
- `random_forest`: mixed middle/upper intervals.

Occupancy geometry (`prediction_space_occupancy_summary.csv`, mean over datasets):

- Bagged HDDT has lower occupancy entropy (`1.0782`) than Random Forest (`1.3469`) and HDDT (`1.1462`), above LightGBM (`0.8027`).
- Bagged HDDT has the highest posterior sparsity index (`0.2467`) among the four, indicating stronger concentration into fewer posterior bins.
- Bagged HDDT uses fewer occupied bins (`4.52`) than HDDT (`5.92`), Random Forest (`5.80`), and LightGBM (`5.88`).

## Hypothesis Assessment (H1-H5)

### H1: Bagging increases low-score accessibility mass

Assessment: **Supported (strong)**.

- Recall gain vs HDDT is largest in permissive thresholds: +`0.2998` at `0.05` and +`0.1460` at `0.01`.
- Recovery rises from `0.5609` (HDDT) to `0.9024` (Bagged HDDT).

### H2: Bagging induces a cliff-like threshold profile

Assessment: **Supported (strong)**.

- Max recall jump is much larger than comparators (`0.6100` vs `0.3458` RF, `0.2690` HDDT, `0.2218` LightGBM).
- Smoothness is lowest (`0.1741`), and jump intervals shift to middle thresholds (`0.25->0.10`, `0.10->0.05`) rather than ultra-low only.

### H3: Bagging improves threshold-survival persistence

Assessment: **Supported (moderate)**.

- Persistence improves from HDDT `0.5006` to Bagged HDDT `0.5397`.
- However, Random Forest remains slightly higher (`0.5597`), so this is an improvement not a best-in-class effect.

### H4: Bagging changes score geometry toward concentrated occupancy

Assessment: **Supported (moderate to strong)**.

- Fewer occupied bins (`4.52`) and high posterior sparsity index (`0.2467`) indicate concentration.
- Entropy sits between LightGBM and tree ensembles, consistent with concentration without full conservative collapse.

### H5: Bagged HDDT and LightGBM are non-equivalent despite similar AP/AUROC band

Assessment: **Supported (strong)**.

- Similar ranking quality (AP `0.3434` vs `0.3462`, AUROC `0.8389` vs `0.8479`) masks operational divergence.
- Bagged HDDT has much larger recovery (`0.9024` vs `0.5485`) and jump (`0.6100` vs `0.2218`), lower smoothness (`0.1741` vs `0.2624`), and higher persistence (`0.5397` vs `0.4413`).

## Working Mechanistic Interpretation

Bagging appears to convert single-tree HDDT broad allocation into an ensemble gate: probability mass remains concentrated enough to create threshold cliffs, but aggregate voting broadens minority reachability once thresholds become permissive. This explains the joint pattern:

- very low strict-threshold recall,
- very high relaxed-threshold recall,
- high jump magnitude,
- moderate-high threshold persistence.

This interpretation is consistent with current aggregate metrics, but it is still inferential (no per-instance vote decomposition yet).

## Uncertainty and Limits

- Dataset count per model is small (`n=5`), so directional claims are stronger than fine-grained effect-size claims.
- `minority_occupancy_compression_ratio_mean` is unstable for LightGBM due extreme values; use medians/robust summaries before making direct ratio comparisons.
- No calibration-curve or per-instance margin decomposition is included yet; mechanism remains phenomenological.
