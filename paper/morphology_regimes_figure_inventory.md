# Morphology Regimes Figure Inventory

| Figure | Figure Title | Purpose | Source Artifact | Manuscript Section | Status | Regeneration Needed |
| --- | --- | --- | --- | --- | --- | --- |
| Fig. 1 | Accessibility Morphology Framework | Introduce Level -> Morphology State -> Dynamics conceptual model | New schematic needed | Introduction / Framework | Not yet publication-ready | Create clean vector schematic |
| Fig. 2 | Shared Accessibility Morphology Space | Show pooled breadth/elevation manifold with classical allocators included | `reports/topology/accessibility_morphology_space_with_classical.png` | Accessibility Morphology Framework | Draft-ready | Regenerate with consistent typography, larger labels, colorblind-safe palette |
| Fig. 3 | Classical Allocator Overlay | Demonstrate classical allocators occupy existing morphology support | `reports/topology/classical_allocator_overlay.png` | Operational Consequences | Draft-ready | Regenerate with family labels and less point overlap |
| Fig. 4 | Kernel Robustness Bandwidth Sweep | Show nonlinear cliffiness result is stable across bandwidths | `reports/topology/kernel_robustness_bandwidth_sweep.png` | Competing Explanations / Nonlinear Geometry | Draft-ready | Regenerate with linear baseline line and grouped-CV emphasis |
| Fig. 5 | Nonlinear Model Comparison | Compare linear, polynomial, RF, kNN, RBF, boosting for cliffiness | `reports/topology/kernel_robustness_model_comparison.png` | Competing Explanations | Draft-ready | Regenerate as sorted horizontal bar plot with grouped CV only |
| Fig. 6 | Frontier Geometry Test | Show frontier/slack features do not explain cliffiness alone | `reports/topology/frontier_model_comparison.png` | Competing Explanations | Draft-ready | Regenerate with compact equation and RF/RBF reference clearly marked |
| Fig. 7 | Morphology Atlas Phase Diagram | Show discovered cluster/regime structure in breadth/elevation space | `reports/topology/regime_phase_diagram.png` | Morphology Atlas Construction | Draft-only | Regenerate after native HDBSCAN/UMAP or relabel as DBSCAN/PCA fallback |
| Fig. 8 | Cliffiness Overlay | Show cliffiness varies by morphology region | `reports/topology/regime_cliffiness_overlay.png` | Morphology Atlas Construction | Draft-ready | Regenerate with shared axes and same color scale as phase diagram |
| Fig. 9 | Cluster Cliffiness Distribution | Show between-cluster cliffiness separation | `reports/topology/cluster_cliffiness_distribution.png` | Morphology Atlas Construction | Draft-ready | Regenerate with macro-regime labels once finalized |
| Fig. 10 | Cluster Distance Heatmap | Show similarity between discovered atlas clusters | `reports/topology/cluster_distance_heatmap.png` | Regime Consolidation | Draft-ready | Regenerate with regime grouping annotations |
| Fig. 11 | Cluster Dendrogram | Motivate hierarchical regime consolidation | `reports/topology/cluster_dendrogram.png` | Regime Consolidation | Draft-ready | Regenerate with selected 4-regime cut line |
| Fig. 12 | Regime Compression Curve | Show 4 regimes retain most cluster predictability | `reports/topology/regime_compression_curve.png` | Regime Consolidation | Draft-ready | Regenerate with cluster baseline and chosen k highlighted |
| Fig. 13 | Within-Regime Local Laws | Show local equations outperform global equation | New figure from `within_regime_equation_scores.json` | Local Accessibility Laws | Missing | Create bar plot of global vs within-regime R2 |
| Appendix A1 | PDP Breadth | Show nonlinear breadth effect on cliffiness | `reports/topology/pdp_breadth.png` | Appendix / Equation Discovery | Appendix-ready | Optional cleanup only |
| Appendix A2 | PDP Elevation | Show elevation effect on cliffiness | `reports/topology/pdp_elevation.png` | Appendix / Equation Discovery | Appendix-ready | Optional cleanup only |
| Appendix A3 | PDP Interaction | Show breadth/elevation interaction | `reports/topology/pdp_interaction.png` | Appendix / Equation Discovery | Appendix-ready | Optional cleanup only |
| Appendix A4 | SHAP/Fallback Summary | Feature contribution diagnostic | `reports/topology/shap_summary.png` | Appendix / Equation Discovery | Not publication-ready | Install SHAP or relabel as feature importance fallback |
| Appendix A5 | Boundary Distance vs Cliffiness | Show simple boundary distance failure | `reports/topology/boundary_distance_vs_cliffiness.png` | Appendix / Competing Explanations | Appendix-ready | Optional cleanup only |
| Appendix A6 | Frontier Distance vs Cliffiness | Show frontier slack is insufficient | `reports/topology/frontier_distance_vs_cliffiness.png` | Appendix / Frontier Geometry | Appendix-ready | Optional cleanup only |
| Appendix A7 | Normalized Frontier Position vs Cliffiness | Show envelope position diagnostic | `reports/topology/normalized_frontier_position_vs_cliffiness.png` | Appendix / Frontier Geometry | Appendix-ready | Optional cleanup only |
| Appendix A8 | Validation Scheme Sensitivity | Show random vs grouped CV behavior | `reports/topology/kernel_robustness_validation_schemes.png` | Limitations / Appendix | Draft-ready | Regenerate with source-family holdout caveat highlighted |

## Publication-Ready Priority

- Must regenerate before submission: Fig. 1, Fig. 2, Fig. 4, Fig. 7, Fig. 12, Fig. 13.
- Can use as draft figures immediately: Fig. 3, Fig. 5, Fig. 6, Fig. 8, Fig. 9, Fig. 10, Fig. 11.
- Appendix-only unless needed: PDPs, SHAP/fallback, boundary/frontier diagnostics, validation-scheme sensitivity.
