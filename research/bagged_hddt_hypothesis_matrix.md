# Bagged HDDT Hypothesis Matrix

## Legend

- Evidence strength: `Strong`, `Moderate`, `Weak`, `Mixed`.
- Status: `Supported`, `Partially supported`, `Not supported`, `Undetermined`.

| Hypothesis | Observable prediction | Key metrics used | Observed pattern | Evidence strength | Status |
| --- | --- | --- | --- | --- | --- |
| H1: Bagging increases low-threshold minority reachability vs HDDT | Higher recall gains at permissive thresholds (0.05, 0.01) than strict thresholds | Recall@{0.50,0.25,0.10,0.05,0.01}, Recovery | `hddt_forest` exceeds `hddt` most at 0.05 (+0.2998) and 0.01 (+0.1460); Recovery `0.9024` vs `0.5609` | Strong | Supported |
| H2: Bagging creates cliff behavior | Larger max recall jump and lower smoothness than comparators | Max Recall Jump, Operational Smoothness, jump-interval frequencies | Largest jump (`0.6100`) and lowest smoothness (`0.1741`); jump intervals concentrate in middle bands | Strong | Supported |
| H3: Bagging improves threshold survival | Higher threshold occupancy persistence than HDDT | Threshold Occupancy Persistence | `0.5397` (Bagged HDDT) vs `0.5006` (HDDT) | Moderate | Supported |
| H4: Bagging changes occupancy geometry toward concentrated support | Fewer occupied bins, higher sparsity index, lower entropy than broad allocators | Occupied bins, Posterior Sparsity Index, Occupancy Entropy | `occupied_bin_count_mean=4.52` (lowest), `posterior_sparsity=0.2467` (highest), entropy reduced vs HDDT/RF | Moderate-Strong | Supported |
| H5: Bagged HDDT and LightGBM are operationally non-equivalent | Similar AP/AUROC but materially different threshold dynamics | AP, AUROC, Recovery, Jump, Smoothness, Persistence | AP/AUROC are close; threshold behavior differs substantially across all dynamic metrics | Strong | Supported |

## Cross-Hypothesis Reading

- The evidence supports a coherent package: Bagged HDDT is not merely "better HDDT" or "LightGBM-like"; it occupies a distinct high-recovery, high-jump regime.
- H3 is the softest result: persistence improvement exists but is incremental relative to Random Forest.
- Mechanistic claims should remain framed as **operational interpretation** pending per-instance decomposition.
