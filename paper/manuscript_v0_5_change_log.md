# manuscript_v0_5 Change Log

Base file: `paper/manuscript_v0_4.md`
Output file: `paper/manuscript_v0_5.md`
Pass type: persistence-semantic alignment and artifact integration

## Summary

This pass integrates persistence-metric remediation findings into manuscript text, figures, and tables, and aligns related exposition files.

## 1) Persistence language alignment

- Updated core terminology so **accessibility persistence** is explicitly defined as threshold-survival persistence across the operational threshold grid.
- Added clarifying wording in the introduction and reachability framing to avoid ambiguity with low-score concentration measures.

## 2) Family-level artifact integration (v2)

Added family-level table and regime-map references in the Accessibility Trajectory Observations section:

- `paper/manuscript_tables/table_allocator_family_summary_v2.md`
- `paper/manuscript_figures/figure_accessibility_regime_map_v2.png`
- `paper/manuscript_figures/figure_accessibility_regime_map_variant_a_v2.png`
- `paper/manuscript_figures/figure_accessibility_regime_map_variant_b_v2.png`

These now appear before neural perturbation discussion to preserve evidence hierarchy.

## 3) Discussion update

- Added a brief **metric semantics clarification** paragraph in Discussion framing:
  - persistence = threshold-survival persistence,
  - low-score mass (`<0.01`) = secondary descriptor.

Framed as clarification, not substantive claim revision.

## 4) Checklist updates

- Updated final manuscript checklist to include Figure 6-8 v2 and Table A (v2).

## 5) Companion exposition update

Updated `paper/bagged_hddt_vs_lightgbm.md` to match persistence-remediated values and interpretation:

- switched reference table to v2,
- corrected persistence-direction statements,
- preserved core non-equivalence argument through smoothness/jump/recovery divergence.

## What did not change

- No new experiments were run.
- No new scientific claims were introduced.
- Core thesis (ranking/reliability non-equivalence with operational accessibility behavior) remains unchanged.
