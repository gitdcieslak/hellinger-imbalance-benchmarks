# Persistence Metric Remediation

## Section 1 — Current Metric Audit Summary

From `paper/metric_semantics_audit.md`, the core issue is:

- The manuscript/table label **Accessibility Persistence** was populated using `mean_fraction_below_0_01`.
- That metric captures low-score mass concentration, not threshold-survival persistence.

Why this was problematic:

- Reader intuition for "persistence" is survival-like behavior across threshold changes.
- The old values created semantic inversions (e.g., CART appearing highly persistent, Bagged HDDT appearing weakly persistent).

Conclusion of audit (confirmed):

- Computation was not the primary bug; **naming/semantics mismatch** was.

---

## Section 2 — Candidate Persistence Metrics Comparison

## Candidate A: `mean_fraction_below_0_01` (old field)

- Source: `reports/neural_mlp/allocation_regime_summary.csv`
- Meaning: concentration of scores below 0.01.
- Strength: useful compression/concentration descriptor.
- Weakness: not threshold-survival persistence.
- Reader alignment: low.

## Candidate B: `threshold_occupancy_persistence_mean` (recommended)

- Source: `reports/neural_mlp/prediction_space_occupancy_summary.csv`
- Underlying computation (`src/hib/occupancy.py`): mean positive-class survival across threshold grid.
- Strength: directly survival-like and threshold-mediated.
- Weakness: still grid-dependent (as all threshold-grid metrics are).
- Reader alignment: high.

## Candidate C: Recovery (`Recall@0.01 - Recall@0.50`)

- Source: threshold sweep summaries.
- Strength: intuitive and stable summary of recoverable accessibility.
- Weakness: endpoint-only; ignores path smoothness.
- Reader alignment: medium-high, but it is not persistence.

## Candidate D: Reachability trajectory area/mean (not separately stored as named artifact)

- Conceptually close to persistence and highly interpretable.
- Not currently exposed as a stable named summary field in the family table pipeline.

### Best match to reader intuition

`threshold_occupancy_persistence_mean` is the best available existing metric for the term **Persistence**.

---

## Section 3 — Recommended Persistence Definition

Recommended manuscript definition:

**Threshold Occupancy Persistence**: the mean minority occupancy above threshold across the operational threshold grid (`0.50, 0.25, 0.10, 0.05, 0.01`), aggregated across runs and then datasets.

Rationale:

- It measures what readers think "persistence" should measure.
- It is already computed in existing artifacts.
- It aligns with reachability/survival semantics and threshold-policy framing.

Terminology update:

- Use **Threshold Occupancy Persistence** as primary persistence metric.
- Keep `mean_fraction_below_0_01` but rename to **Low-Score Mass (<0.01)** and treat as secondary concentration signal.

---

## Section 4 — Migration Plan

## 4.1 Implemented outputs

Rebuilt table:

- `paper/manuscript_tables/table_allocator_family_summary_v2.md`

Rebuilt regime maps with remediated persistence axis:

- `paper/manuscript_figures/figure_accessibility_regime_map_v2.png`
- `paper/manuscript_figures/figure_accessibility_regime_map_variant_a_v2.png`
- `paper/manuscript_figures/figure_accessibility_regime_map_variant_b_v2.png`

Supporting numeric artifact:

- `paper/manuscript_tables/_allocator_family_summary_values_v2.csv`

Generation script:

- `scripts/remediate_persistence_metric.py`

## 4.2 Reassessment after remediation

### Morphology separation

- Still present, but less exaggerated by semantic artifact.
- Cliff/broad distinctions now rely more cleanly on smoothness + jump + recovery combinations.

### Family clustering

- Broad allocators (HDDT, Random Forest) remain close.
- Cliff allocators remain heterogeneous (XGBoost, Bagged HDDT, MLP), consistent with prior caveat.
- CART remains an outlier (quantized behavior), but no longer looks "maximally persistent" by definition mismatch.

### Bagged HDDT vs LightGBM

- The old persistence contrast flips meaningfully under corrected persistence:
  - Bagged HDDT persistence (`0.5397`) > LightGBM (`0.4413`).
- The pair still demonstrates divergence via smoothness/jump/recovery profile differences, but persistence can no longer be used in the old direction.

### Narrative impact

- Positive: "persistence" language is now semantically defendable.
- Required adjustment: update Bagged HDDT vs LightGBM exposition to avoid citing old persistence contrast direction.

## 4.3 Minimal manuscript patch checklist

1. Replace references to old family table with `paper/manuscript_tables/table_allocator_family_summary_v2.md`.
2. In text/captions, reserve "persistence" for threshold-survival metrics.
3. Replace old regime-map figure references with `_v2` versions where persistence is discussed.
4. In `paper/bagged_hddt_vs_lightgbm.md`, revise persistence comparison sentence to match v2 values.

---

## Final Decision

After remediation, the manuscript can use the term **Persistence** without predictable reviewer confusion, provided v2 table/figures and wording updates are adopted consistently.
