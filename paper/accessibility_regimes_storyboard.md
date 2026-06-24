# Accessibility Regimes Storyboard

| Figure | Title | Source Path | Keep / Appendix / Cut | Reason | Claim Supported | Review Concern |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Shared Accessibility Morphology Space | `../reports/topology/accessibility_morphology_space_with_classical.png` | Keep | Establishes Act I morphology space and shows classical allocators inside support. | Accessibility can be represented in shared breadth/elevation space. | Needs publication styling and possibly clearer family labels. |
| 2 | Native UMAP Morphology Atlas | `../reports/topology/morphology_umap_space.png` | Keep | Central visual for native HDBSCAN/UMAP robustness. | Macro-regime regions remain visible under native embedding. | Must state point-level labels are not stable. |
| 3 | Regime Consolidation Dendrogram | `../reports/topology/cluster_dendrogram.png` | Keep | Shows how clusters are compressed into macro-regimes. | Four-regime vocabulary is a structured consolidation, not arbitrary naming. | Needs cut line and labels. |
| 4 | Regime Compression Curve | `../reports/topology/regime_compression_curve.png` | Keep | Quantifies predictive retention under regime compression. | Four regimes retain most cluster-level cliffiness signal. | Should include cluster baseline and selected k marker. |
| 5 | Nonlinear Model Comparison | `../reports/topology/kernel_robustness_model_comparison.png` | Keep | Shows linear/global explanations fail relative to nonlinear morphology. | Cliffiness needs nonlinear/local structure. | Needs grouped-CV-only emphasis in final version. |
| 6 | Frontier Geometry Falsification | `../reports/topology/frontier_model_comparison.png` | Keep or Appendix | Important negative result, but may be secondary. | Cliffiness is not primarily frontier distance. | Could be moved to appendix if main paper is too long. |
| 7 | Transition Paths in Morphology Space | `../reports/topology/regime_transition_paths_morphology_space.png` | Keep for review; maybe Appendix later | Main Act III visual showing paths through regimes. | Interventions induce structured movement through regime space. | Projected labels and dense arrows require caveat. |
| 8 | Regime Stability by Intervention | `../reports/topology/regime_stability_by_intervention.png` | Keep for review; maybe Appendix later | Summarizes smooth vs phase-like movement. | Different interventions have different regime-change probabilities. | Needs clearer labels and possibly fewer categories. |
| 9 | Cliffiness Change by Intervention | `../reports/topology/regime_transition_delta_cliffiness.png` | Appendix candidate | Useful but less central than transition graph/stability. | Regime movement connects to cliffiness changes. | Boxplot may be redundant with Table 5. |

## Story Arc

| Act | Figures | Narrative Role |
| --- | --- | --- |
| Act I: Morphology | Figure 1 | Introduce breadth/elevation as shared accessibility state space. |
| Act II: Regimes | Figures 2-6 | Show that cliffiness is not global/linear/frontier-based and that macro-regimes explain allocator states. |
| Act III: Transitions | Figures 7-9 | Show interventions move allocators through regime space and help explain cliffiness changes. |

## Main-Paper Minimal Figure Set

| Priority | Figure | Rationale |
| --- | --- | --- |
| Essential | Figure 1 | Without morphology space, regime argument has no visual foundation. |
| Essential | Figure 2 | Native HDBSCAN/UMAP robustness is a key reviewer concern. |
| Essential | Figure 4 | Compression curve justifies macro-regime vocabulary. |
| Essential | Figure 5 | Shows why nonlinear/regime explanation is needed. |
| Essential | Figure 7 | Establishes transition dynamics as the new Act III result. |
| Optional Main | Figure 3 | Useful if space permits; otherwise appendix with Figure 4. |
| Optional Main | Figure 6 | Strong falsification result but can be appendix. |
| Optional Main | Figure 8 | Useful if transition analysis stays in main text. |
| Appendix | Figure 9 | Supports intervention effects but likely redundant with Table 5. |

## Regeneration Checklist

1. Apply one colorblind-safe palette across regime figures.
2. Use consistent regime names, not only numeric IDs.
3. Add source/caption text directly below each figure.
4. Highlight k=4 in dendrogram and compression curve.
5. Simplify transition figures if they remain in the main manuscript.
