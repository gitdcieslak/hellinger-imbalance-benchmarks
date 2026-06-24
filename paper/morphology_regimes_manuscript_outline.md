# Morphology Regimes of Accessibility: Manuscript Outline

## Draft Readiness Decision

The existing evidence is sufficient for a first complete manuscript draft. The paper should be framed around a regime-based accessibility morphology theory rather than only a global kernel-smoothing result.

The smallest credible submission package still needs three targeted additions: regenerate final publication-quality figures, add native HDBSCAN/UMAP or explicitly label the current atlas as DBSCAN/PCA fallback, and run one robustness check showing that the regime consolidation is not an artifact of the classical-only atlas construction.

## Working Thesis

Accessibility under extreme class imbalance is governed by morphology regimes in allocator score-allocation space. Accessibility level is largely determined by position on a shared breadth/elevation morphology manifold, while accessibility dynamics, especially cliffiness, are better explained by regime membership and local nonlinear accessibility laws than by a single global linear equation or frontier distance.

## Abstract

### Key Claims

- Extreme class imbalance creates an operational threshold-selection problem not captured by AUROC, average precision, or calibration summaries.
- Accessibility morphology summarizes minority score allocation using breadth, elevation, persistence, and cliffiness.
- Across neural, classical, density, fragmentation, and trajectory experiments, allocators occupy a shared morphology manifold.
- Cliffiness is not explained by linear morphology, scalar manifold coordinates, topology-only descriptors, or simple frontier distance.
- Nonlinear morphology models robustly explain cliffiness, and unsupervised atlas construction reveals discrete/semi-discrete morphology regimes.
- A compact operational theory emerges: global cliffiness equals regime selection plus simple local accessibility laws.

### Supporting Figures

- `accessibility_morphology_space_with_classical.png`
- `kernel_robustness_bandwidth_sweep.png`
- `regime_phase_diagram.png`
- `regime_compression_curve.png`

### Supporting Tables

- Kernel robustness summary table from `kernel_morphology_robustness_summary.md`
- Regime characterization table from `morphology_regime_consolidation.md`
- Reviewer-facing claim/gap table from this planning packet

### Supporting Experiments

- Classical allocator integration
- Kernel robustness check
- Morphology equation discovery
- Frontier geometry investigation
- Morphology atlas construction
- Regime consolidation

## 1. Introduction

### Key Claims

- Operational threshold selection under severe skew requires more than ranking quality.
- Accessibility is the operational availability of minority examples across threshold movement.
- A learner’s minority score allocation has a morphology that determines threshold usability.
- The manuscript asks whether accessibility behavior is governed by a shared morphology manifold and regime structure across allocator families.

### Supporting Figures

- Conceptual diagram to regenerate: `Accessibility Level -> Morphology State -> Accessibility Dynamics`
- `accessibility_morphology_space_with_classical.png`

### Supporting Tables

- Gap audit from `paper2_morphology_framework.md`

### Supporting Experiments

- Weighted dropout sweep: accessibility-optimal dropout differs from AUROC/AP optima.
- Weighted oversampling grid: accessibility optimum differs from ranking-metric optimum.
- Classical allocator placement: tree/ensemble/boosted/Hellinger models occupy the same morphology space.

## 2. Related Work

### Key Claims

- Imbalance learning literature emphasizes ranking, resampling, reweighting, and calibrated probabilities, but less often threshold-path accessibility.
- Calibration and ranking metrics describe different model properties than accessibility morphology.
- Prior decision-tree and Hellinger-distance work motivates skew-aware splitting but does not provide a morphology/regime account of threshold dynamics.
- Manifold, clustering, and phase-diagram ideas provide a useful language for allocator behavior, but the paper’s contribution is applying them to operational accessibility under skew.

### Supporting Figures

- No primary empirical figure required.

### Supporting Tables

- Optional related-work positioning table, likely appendix.

### Supporting Experiments

- None directly; this section should contextualize the empirical contribution.

## 3. Accessibility Morphology Framework

### Key Claims

- Accessibility breadth measures spread of minority-positive score mass across accessible threshold regions.
- Accessibility elevation measures how much minority-positive mass reaches high-score/high-accessibility regions.
- Accessibility persistence is measured by minority survival AUC.
- Accessibility dynamics are measured by cliffiness, max drops, and effective drop count.
- Breadth/elevation define the morphology state space; persistence and cliffiness are outcomes induced by that state.

### Supporting Figures

- `accessibility_morphology_space_with_classical.png`
- `classical_allocator_overlay.png`
- `accessibility_operational_regime_map.png`

### Supporting Tables

- Morphology explanatory power table from `accessibility_morphology_synthesis_summary.md`
- Classical placement audit from `accessibility_morphology_with_classical.md`

### Supporting Experiments

- Morphology synthesis report
- Classical allocator integration
- Manifold coordinate experiments

## 4. Competing Explanations of Cliffiness

### Key Claims

- Linear morphology fails for cliffiness.
- Scalar manifold coordinates fail for cliffiness.
- Topology-only and topology-augmented descriptors do not explain cliffiness directly.
- Compact nonlinear equations partly explain cliffiness but do not recover the full nonlinear signal.
- Frontier distance and simple boundary distance are not sufficient explanations.

### Supporting Figures

- `frontier_model_comparison.png`
- `frontier_distance_vs_cliffiness.png`
- `boundary_distance_vs_cliffiness.png`
- `kernel_robustness_model_comparison.png`

### Supporting Tables

- Equation discovery summary from `morphology_equation_discovery.md`
- Frontier predictive tests from `frontier_morphology_summary.md`
- Topology-augmented morphology performance from `topology_augmented_morphology_summary.md`

### Supporting Experiments

- Morphology equation discovery
- Frontier geometry investigation
- Kernel robustness study
- Topology-augmented morphology

## 5. Morphology Atlas Construction

### Key Claims

- Classical allocator behavior is not a featureless continuum in morphology space.
- Unsupervised clustering identifies stable/semi-stable morphology phases.
- Cluster assignment predicts cliffiness and survival buckets substantially above baseline.
- Between-cluster cliffiness variance exceeds within-cluster variance.

### Supporting Figures

- `regime_phase_diagram.png`
- `regime_cliffiness_overlay.png`
- `cluster_cliffiness_distribution.png`

### Supporting Tables

- `cluster_summary_table.md`
- `cluster_stability_report.md`

### Supporting Experiments

- Morphology atlas construction over classical allocators.

### Caveat

- Current atlas used DBSCAN fallback because native HDBSCAN was unavailable, and PCA(2) fallback because UMAP was unavailable. Before submission either install and rerun native HDBSCAN/UMAP or state the fallback honestly and demote HDBSCAN/UMAP language.

## 6. Morphology Regime Consolidation

### Key Claims

- Four macro-regimes preserve most of the 14-cluster cliffiness predictability.
- The selected four-regime representation explains 0.6861 of cliffiness variance.
- Cliffiness-bucket accuracy drops only from 0.7710 with cluster ID to 0.7328 with four regimes.
- The four regimes provide an operational vocabulary: Quantized floor, Broad-flat high-cliff basin, Elevated broad plateau, and Mixed/transitional morphology.

### Supporting Figures

- `cluster_distance_heatmap.png`
- `cluster_dendrogram.png`
- `regime_compression_curve.png`

### Supporting Tables

- Regime characterization table from `morphology_regime_consolidation.md`
- Compression curve table from `morphology_regime_consolidation.md`
- `within_regime_equation_scores.json`

### Supporting Experiments

- Regime consolidation from atlas clusters.

## 7. Local Accessibility Laws

### Key Claims

- A single global equation is weaker than regime-conditioned local equations.
- Global ridge equation over breadth/elevation has R2 around 0.4494 in regime-consolidation analysis.
- Within-regime ridge equations improve in every nontrivial regime, with R2 values around 0.5077, 0.5439, and 0.6570, plus a small near-degenerate zero-cliffiness regime.
- The best theory statement is: global cliffiness = regime selection + simple local equation.

### Supporting Figures

- `regime_compression_curve.png`
- Optional new figure to regenerate: within-regime equation score bar plot.

### Supporting Tables

- `within_regime_equation_scores.json`
- Equation discovery table from `morphology_equation_discovery.md`

### Supporting Experiments

- Morphology equation discovery
- Regime consolidation

## 8. Operational Consequences

### Key Claims

- CART occupies a quantized/low-accessibility floor regime.
- Bagged HDDT, HDDT, random forest, and XGBoost often occupy broad low-elevation or transitional high-cliff regimes.
- LightGBM spans mixed and low-elevation regions under the current severe-skew-safe setup.
- Operational threshold risk depends on regime, not just ranking metrics.
- Morphology regimes can guide allocator selection and training interventions.

### Supporting Figures

- `classical_allocator_overlay.png`
- `regime_phase_diagram.png`
- `cluster_cliffiness_distribution.png`

### Supporting Tables

- Classical placement audit from `accessibility_morphology_with_classical.md`
- Cluster/regime summary tables

### Supporting Experiments

- Classical allocator morphology sweep
- Morphology atlas and regime consolidation

## 9. Limitations

### Key Claims

- Many experiments are synthetic; external real-data validation is still limited.
- Atlas construction currently uses classical allocator data as primary input; the regime story should be validated on the pooled full dataset or explicitly scoped.
- Native HDBSCAN/UMAP were unavailable during atlas generation; the current atlas uses DBSCAN/PCA fallbacks.
- Kernel and RF models identify nonlinear structure but do not prove causality.
- Source-family holdout is much harsher than experiment-family/model-family CV and exposes extrapolation limits.
- Regime labels are empirically derived and need stability checks across datasets and skew settings.

### Supporting Figures

- Validation-scheme sensitivity figure from kernel robustness.

### Supporting Tables

- Kernel robustness validation-scheme table
- Reviewer objection matrix

## 10. Future Work

### Key Claims

- Validate morphology regimes on real or semi-real imbalanced datasets.
- Rerun atlas with native HDBSCAN/UMAP and bootstrap consensus clustering.
- Develop intervention studies that intentionally move allocators between regimes.
- Extend regime dynamics to a transition/flow-field model.
- Use morphology objectives during training to control accessibility regimes.

### Supporting Figures

- Transition graph from regime consolidation as future-work seed.

### Supporting Tables

- Missing experiments document.

## 11. Conclusion

### Key Claims

- Accessibility morphology is not a neural-network-specific phenomenon.
- Allocators occupy a shared morphology space, but cliffiness is best understood through local nonlinear/regime structure.
- The evidence now supports a manuscript centered on morphology regimes of accessibility.
- The strongest current theory is: accessibility level is a morphology-position effect, while accessibility dynamics arise from morphology regime selection plus local accessibility laws.

### Final Draft Thesis

Accessibility under extreme class imbalance is governed by morphology regimes in minority score-allocation space. Across trees, ensembles, boosted learners, Hellinger-based allocators, and neural objectives, allocators occupy a shared breadth/elevation manifold. Persistence is largely a morphology-position variable, while cliffiness is not a global linear, scalar, frontier-distance, or topology-only effect. Instead, cliffiness is explained by a hybrid structure: regime selection plus local nonlinear accessibility laws.
