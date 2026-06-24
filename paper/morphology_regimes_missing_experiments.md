# Morphology Regimes Missing Experiments and Claim Hierarchy

## Submission Readiness

The existing evidence is sufficient for a first complete draft. It is not yet sufficient for a strong final submission if the manuscript makes broad allocator-general or regime-law claims without qualification.

The smallest additional analysis set for a credible submission is:

1. Native clustering robustness: rerun atlas with HDBSCAN and UMAP installed, or explicitly reframe the atlas as DBSCAN/PCA clustering.
2. Model-family confounding check: evaluate whether regimes predict cliffiness beyond allocator family and skew.
3. Cross-validated within-regime local equations: verify local laws are not only in-sample improvements.

## Claim Hierarchy

### Tier 1 — Core Paper Claims

These are essential; the paper fails without them.

| Claim | Current Status | Evidence | Risk |
| --- | --- | --- | --- |
| Accessibility is an operational property distinct from AUROC/AP/calibration | Strong | Dropout and oversampling optima differ from ranking optima | Needs concise main-table demonstration |
| Breadth/elevation morphology state explains accessibility level | Strong | Survival AUC high under linear morphology and robustness studies | Source-family extrapolation caveat |
| Cliffiness is not linear/scalar morphology | Strong | Linear, scalar, topology-only, frontier-only failures | Low risk |
| Nonlinear morphology explains cliffiness robustly | Strong | Kernel robustness: all RBF bandwidths beat linear; many nonlinear smoothers beat linear | Source-family holdout caveat |
| Morphology regimes provide an interpretable structure for cliffiness | Moderate-to-Strong | Atlas and consolidation results, 4-regime compression, between/within variance ratio | Needs clustering and confounding robustness |

### Tier 2 — Supporting Claims

These strengthen the story but are not individually essential.

| Claim | Current Status | Evidence | Risk |
| --- | --- | --- | --- |
| Classical allocators occupy the shared manifold | Strong | 6/6 inside prior support, 480-row classical sweep | Synthetic-only limitation |
| Four macro-regimes are a useful operational vocabulary | Strong for current data | Compression loss only 0.0382 vs 14 clusters | Naming and k selection may be challenged |
| Compact equation partially explains cliffiness | Moderate | Stepwise equation CV R2 about 0.596 | Below nonlinear reference |
| Frontier distance is not sufficient | Strong negative result | Frontier-only grouped CV R2 below linear | Low risk if framed as rejected explanation |
| Topology is mechanism candidate, not descriptor explanation | Moderate | Topology features do not improve cliffiness much | Could need deeper topology features |

### Tier 3 — Exploratory Findings

Move these to appendices unless space permits.

| Finding | Current Status | Suggested Placement |
| --- | --- | --- |
| SHAP/fallback contribution plots | Exploratory; SHAP unavailable | Appendix or omit until SHAP installed |
| Diffusion/manifold single-coordinate tests | Negative/exploratory | Appendix |
| Source-family holdout results | Important limitation, not core result | Limitations and appendix |
| Transition graph between regimes | Early future-work seed | Future work |
| Individual cluster-level local RF/tree fits | Exploratory | Appendix after CV validation |

## Missing Experiments

| Experiment | Expected Contribution | Implementation Complexity | Estimated Paper Impact | Classification |
| --- | --- | --- | --- | --- |
| Native HDBSCAN + UMAP atlas rerun | Removes major clustering-method objection and validates regime discovery with intended methods | Low to Medium; install packages and rerun atlas | High | Required before submission |
| Family/skew confounding analysis | Tests whether regimes predict cliffiness beyond allocator family and skew ratio | Medium; multinomial/logistic/RF residualization or grouped models | High | Required before submission |
| Cross-validated within-regime local equations | Validates “regime selection + local law” claim without in-sample overfit | Medium; grouped or repeated CV inside regimes, handle small n | High | Required before submission if local laws are main text |
| Publication-quality figure regeneration | Makes figures coherent and reviewer-readable | Medium; styling pass across selected figures | High | Required before submission |
| Atlas on pooled full morphology dataset, not only classical | Tests whether regimes generalize beyond classical allocator atlas | Medium; combine prior + classical and rerun clustering/consolidation | High | Strongly recommended |
| Repeat atlas using alternative cliffiness-related targets | Checks metric dependence using max_drop/effective_drop_count | Low to Medium | Medium | Strongly recommended |
| Real or semi-real imbalanced dataset validation | Addresses external validity and synthetic artifact objection | Medium to High depending data availability | High | Strongly recommended |
| Exact KernelRidge/Gaussian Process subset check | Confirms Nystroem approximation does not drive kernel result | Medium; use subset or Nyström comparison | Medium | Nice to have |
| LightGBM variant comparison | Tests sensitivity to severe-skew-safe LightGBM settings | Low to Medium | Low to Medium | Nice to have |
| Causal morphology perturbation | Moves from descriptive regime theory to intervention theory | High | Very High but likely future paper | Nice to have / Future work |

## Minimal Pre-Submission Analysis Set

Do these before final submission:

1. Install `hdbscan` and `umap-learn`; rerun `scripts/report_morphology_atlas.py` and `scripts/report_morphology_regime_consolidation.py`.
2. Add a model-family/skew confounding check for regime predictive power.
3. Add cross-validated within-regime local equation scores.
4. Regenerate the main six figures with consistent styling.

## Bottom Line

The first complete draft should be written now. The central narrative is coherent and supported: accessibility morphology reveals regimes of threshold behavior under extreme skew. The paper should not wait for broad new sweeps. It needs targeted robustness and presentation work, not more exploratory experiments.
