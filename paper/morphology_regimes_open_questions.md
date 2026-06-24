# Morphology Regimes Open Questions

## Unresolved Reviewer Concerns

| Concern | Current Status | Suggested Response |
| --- | --- | --- |
| External validity | Evidence is mostly synthetic severe-skew plus controlled classical allocator sweeps. | Add one real or semi-real imbalanced benchmark before broad submission, or frame as controlled morphology theory. |
| Cluster identity stability | Native HDBSCAN/UMAP preserves macro-regimes but not point-level labels; DBSCAN vs HDBSCAN ARI is 0.1705. | Claim macro-regime robustness only. Do not claim exact cluster reproducibility. |
| Local-law instability | Fallback regimes show partial within-regime improvement; native HDBSCAN does not replicate it. | Present local laws as hypothesis/future work. Main claim should be regime-structured cliffiness. |
| Regime projection in transition analysis | Most pooled transition rows are classifier projections into native regime vocabulary. | Label projection clearly; keep transition analysis exploratory or appendix-level. |
| Family/skew confounding | Regime adds R2 beyond family/skew, but bucket-accuracy gains are modest. | Frame regimes as explanatory compression, not optimal classifiers. Repeat confounding after final native atlas if needed. |
| Source-family extrapolation | Kernel robustness is poor under source-family holdout. | Scope claims to observed severe-skew morphology support. Treat source-family holdout as extrapolation stress test. |
| Metric dependence | Atlas/regime results focus on cliffiness. | Repeat analysis for max drop and effective drop count, or include as future work. |
| SHAP claims | SHAP was unavailable; plots are fallback importance/dependence plots. | Remove SHAP language or rerun after installing SHAP. |

## Weak Sections In v0.2

1. The abstract is complete but dense; advisor may want a simpler one-paragraph version for workshop submission.
2. The morphology atlas section describes both fallback and native clustering; it may need a cleaner chronology in final form.
3. Regime names are still provisional, especially the two native mixed regimes.
4. The transition analysis is strong numerically but depends on projected labels and proxy orderings for objectives and allocator families.
5. Figure references are still placeholders and need publication-quality regeneration.
6. The local-law theory has been softened; the paper needs a clear choice between emphasizing regimes alone or regime plus local morphology.

## Claims Needing Additional Support

| Claim | Support Needed |
| --- | --- |
| Regimes generalize beyond synthetic severe skew | Real or semi-real imbalanced dataset validation. |
| Regime transitions are causal | Deliberate interventions that target movement between regimes. |
| Local equations characterize regimes | Native-regime local models that outperform global baselines, or alternative small-regime handling. |
| Breadth/elevation are sufficient morphology coordinates | Coordinate ablation against AUROC/AP/calibration/topology features. |
| Regimes are stable under metric choice | Repeat atlas with `minority_survival_max_drop` and `minority_survival_effective_drop_count`. |

## Candidate Submission Venues

| Venue Type | Candidate Venues | Fit |
| --- | --- | --- |
| Workshop | NeurIPS/ICML/ICLR workshops on trustworthy ML, imbalanced learning, evaluation, or data-centric AI | Best near-term fit if framed as new evaluation geometry and exploratory regime theory. |
| Applied ML conference | KDD workshop, SDM workshop, ECML/PKDD workshop | Good if operational threshold/accessibility framing is emphasized. |
| Journal | Machine Learning, Data Mining and Knowledge Discovery, ACM TKDD | Possible after real-data validation and publication-quality figures. |
| Domain venue | Rare-event detection, fraud, medical screening, or risk modeling workshops | Good if paired with a concrete real/semi-real benchmark. |

## Recommended Next Steps Before Advisor Review

1. Regenerate priority figures with consistent style and final numbering.
2. Decide whether transition analysis belongs in main Results or Appendix.
3. Remove or relabel all SHAP references unless SHAP is installed and rerun.
4. Add a one-page methods appendix listing every input CSV and feature transformation.
5. Prepare a short advisor memo stating the main claim: regime structure is robust; local laws remain unresolved.

## Recommended Next Steps Before Submission

1. Add at least one real or semi-real imbalanced dataset.
2. Repeat regime analysis for related survival-shape metrics.
3. Repeat family/skew confounding on final native regime labels, not fallback labels.
4. Improve local-law validation or move local laws entirely to future work.
5. Create final figure set with colorblind-safe palettes, larger labels, and consistent captions.
