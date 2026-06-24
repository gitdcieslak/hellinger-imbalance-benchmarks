# Morphology Regimes Reviewer Objection Matrix

| Major Claim | Likely Reviewer Objection | Existing Evidence | Remaining Evidence Gap | Response Strategy |
| --- | --- | --- | --- | --- |
| Accessibility is distinct from ranking/calibration | AUROC/AP already characterize threshold behavior sufficiently | Weighted oversampling and dropout reports show accessibility optima differ from AUROC/AP optima | Add one concise table in main paper comparing ranking optima vs accessibility optima | Emphasize operational threshold path, not global ranking |
| Breadth/elevation define a meaningful morphology space | These are arbitrary engineered features | Survival AUC linear morphology CV R2 around 0.858 before classical; classical-inclusive linear survival remains high in robustness report | Need cleaner definition and ablation against conventional metrics | Define metrics operationally and compare to AUROC/AP baselines |
| Classical allocators occupy same morphology manifold | Classical rows may be forced into the same 2D plot by construction | Classical integration: 6/6 model families inside prior support; PC1 variance shift modest | Validate on full pooled atlas, not only classical-only atlas | Present manifold support diagnostics and avoid overclaiming universality |
| Cliffiness is nonlinear morphology geometry | Kernel result may be tuning artifact | Robustness report: 18 nonlinear models beat linear under grouped CV; 9/9 RBF bandwidths beat linear by 0.25+ | Source-family holdout remains poor | Use experiment-family/model-grouped CV as main; discuss source-family as extrapolation stress test |
| Kernel robustness is sufficient | Nystroem approximation is not exact KernelRidge | Multiple nonlinear smoothers agree: RF, kNN, boosting, polynomial, RBF | Exact KernelRidge/GPR not run on full data due runtime | State the goal is robustness of nonlinear morphology, not one kernel implementation |
| Compact equation explains cliffiness | Equation R2 is below nonlinear reference | Stepwise equation `log_breadth + breadth + breadth2_x_elevation` reaches grouped CV R2 about 0.596 | Need validation of equation stability under bootstraps or holdout | Frame as partial interpretable approximation, not final law |
| Frontier distance explains cliffiness | Frontier-only model performs worse than linear morphology | Frontier report: frontier-only grouped CV R2 -0.2265; frontier+morphology 0.0553 | None; evidence is negative | Present frontier as rejected competing explanation |
| Boundary-distance hypothesis failed | Boundary construction may be too simple | SVM/logistic and frontier-specific boundaries both weak | More sophisticated transition-boundary learning could be tested | State current simple boundary hypotheses are not supported |
| Atlas clusters are stable regimes | Clustering instability/projection artifacts | Atlas ARI moderate, bootstrap ARI mean 0.5922, between/within cliffiness ratio 4.9351 | Native HDBSCAN/UMAP unavailable; current primary cluster is DBSCAN fallback | Required before submission: rerun with HDBSCAN/UMAP or relabel method clearly |
| Four macro-regimes are not arbitrary | k chosen post hoc | k=4 has best silhouette among 3-6, strong variance explained 0.6861, low compression loss 0.0382 | Need sensitivity to alternative k and naming rules in appendix | Include compression curve and dendrogram with k cut line |
| Regime ID explains cliffiness | Regime ID may proxy model family | Regime ID cliffiness bucket accuracy 0.7328; cluster accuracy 0.7710 | Need family-held-out regime predictive test | Add model-family confounding check before submission |
| Morphology coordinates alone perform better than regimes in bucket prediction | This weakens regime theory | Regime predictive table: coordinates accuracy 0.8015 vs regime 0.7328 | Need clarify regimes are explanatory vocabulary, not always strongest classifier | Frame regimes as compression/interpretability, not pure prediction optimization |
| Within-regime equations improve global equation | In-sample R2 can overfit, especially small regimes | Within-regime ridge improves over global ridge in every regime; small 28-row regime likely degenerate | Need cross-validated within-regime equations | Required before submission if local laws are a main claim |
| Model-family confounding drives regimes | Clusters dominated by specific allocators | Cluster summaries show dominant families; regimes mix some families but not all | Need stratified analysis by family/skew | Present as allocator-regime relationship, not independent causal law yet |
| Dataset-specific artifact | All data are synthetic/severe-skew generated | Multiple synthetic families and classical/neural allocators included | Real or semi-real dataset missing | Strongly recommended before submission if targeting broad ML venue |
| Metric dependence | Cliffiness metric may encode the discovered regimes mechanically | Multiple survival-shape metrics exist, but atlas focuses on cliffiness | Need repeat atlas with max_drop/effective_drop_count | Nice-to-have robustness appendix |
| Interpolation vs extrapolation | Grouped CV may still interpolate within morphology support | Source-family holdout poor in kernel robustness | Need explicit statement: theory is supported inside observed support | Scope claim to observed severe-skew morphology manifold |
| LightGBM settings affect conclusions | Severe-skew-safe LightGBM settings may bias morphology | Model registry preserves min_child/min_data settings to avoid collapse | Need compare LightGBM variants if claiming LightGBM-specific behavior | Keep LightGBM as one allocator placement, not a main claim |

## Highest-Risk Objections

1. Clustering instability due to missing native HDBSCAN/UMAP.
2. Regime ID confounded with model family.
3. Within-regime equation improvements are in-sample.
4. No real-data validation.
5. Source-family holdout limits extrapolation claim.
