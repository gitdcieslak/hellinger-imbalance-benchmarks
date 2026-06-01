# Accessibility Regime Analysis

## Objective

Assess whether recurring allocator/morphology patterns occupy distinguishable regions in operational metric space, using existing family-level artifacts only.

## Generated Figure Assets

- Primary map: `paper/manuscript_figures/figure_accessibility_regime_map.png`
- Variant A: `paper/manuscript_figures/figure_accessibility_regime_map_variant_a.png`
- Variant B: `paper/manuscript_figures/figure_accessibility_regime_map_variant_b.png`

## Data Sources

- `paper/manuscript_tables/table_allocator_family_summary.md`
- `paper/manuscript_tables/_allocator_family_summary_values.csv`
- `reports/neural_mlp/allocation_regime_summary.csv`
- `reports/neural_mlp/legacy_benchmark_summary.csv`
- `reports/neural_mlp/legacy_threshold_sweep_summary.csv`

## Primary Projection

Axes:

- X = Operational Smoothness
- Y = Accessibility Persistence (as reported in current artifacts)

Models:

- CART
- HDDT
- Bagged HDDT (`hddt_forest`)
- Random Forest
- XGBoost
- LightGBM
- MLP

## Findings

### Do morphology labels cluster?

Partially yes.

- `broad_allocator` points (HDDT, Random Forest) are close in the lower-middle smoothness region with mid persistence.
- `cliff_allocator` points (XGBoost, Bagged HDDT, MLP) are more dispersed, but occupy a generally lower-smoothness region than CART and MLP exception aside.
- `conservative_allocator` (LightGBM) appears as a separate single-point region with moderate smoothness and high persistence.

Interpretation: labels are not perfectly separable convex clusters, but they do show structured regional tendencies rather than random scatter.

### Do cliff allocators occupy a coherent region?

Moderately coherent, not tightly compact.

- XGBoost and Bagged HDDT align on low smoothness with high jump behavior (seen most clearly in Variant A).
- MLP is cliff-labeled but displaced toward higher smoothness than XGBoost/Bagged HDDT, indicating a softer cliff profile under this aggregate projection.

Interpretation: the cliff family appears as a tendency band, not a single compact island.

### Do broad allocators occupy a coherent region?

Yes, relatively coherent for this sample.

- HDDT and Random Forest appear close on smoothness and persistence in the primary map.
- They also remain close in Variant A (jump vs persistence), with lower jump intensity than cliff exemplars.

### Does CART appear as an outlier?

Yes, strongly.

- CART sits at maximal smoothness with maximal persistence and zero jump, consistent with quantized/invariant threshold response.
- It is visually isolated from all other families in all three projections.

### Does LightGBM separate from Bagged HDDT despite similar AUROC?

Yes.

- AUROC values are both high, but operational geometry differs:
  - LightGBM: higher smoothness and conservative pattern.
  - Bagged HDDT: lower smoothness with high jump intensity and cliff pattern.
- Variant A (Max Recall Jump vs Persistence) makes this separation especially clear.

## Variant Utility

- Variant A (`max_recall_jump` vs `accessibility_persistence`) is the clearest for cliff-versus-broad separation.
- Variant B (`recovery` vs `accessibility_persistence`) shows non-equivalence structure but with more overlap.

## Conclusion

The morphology language is supported by metric geometry beyond trajectory visuals, with two caveats:

1. Separation is **structured but imperfect** (expected in small family-level samples).
2. Cliff-labeled models are **heterogeneous in intensity**, so region-style interpretation is stronger than hard class boundaries.

Overall, the regime map supports using recurring accessibility patterns as provisional empirical organization in manuscript framing.
