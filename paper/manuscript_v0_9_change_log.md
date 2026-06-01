# manuscript_v0_9 Change Log

Base file: `paper/manuscript_v0_8.md`
Output file: `paper/manuscript_v0_9.md`
Pass type: reviewer-integration emphasis and readability hardening

## Summary

This pass improves structure and emphasis using existing evidence only. No new experiments, metrics, contributions, or claim classes were introduced.

## 1) Introduction strengthening

- Added a paragraph after the motivating example and before the contributions list.
- New paragraph makes family-level observations explicit early (CART, HDDT, Bagged HDDT, Random Forest, LightGBM, neural models).

## 2) Running example promotion

- Replaced the prior short callout with a stronger, recurring anchor:
  - **Running Example: Similar Ranking, Different Accessibility (Bagged HDDT vs LightGBM)**.
- Callout includes AUROC/AP similarity plus recovery/jump/smoothness/persistence interpretation and stability qualification.

## 3) Naming consistency

- Added Experimental Framework note:
  - `hddt_forest` is the implementation identifier for Bagged HDDT.
- Normalized narrative usage to "Bagged HDDT" in manuscript-facing text.

## 4) Section 9 emphasis upgrade

- Reopened section with practical implication statement:
  - reliability and accessibility are distinct operational axes.
- Moved operational question to the top of the section.
- Added explicit figure lead-in describing what to observe before Figure 5.

## 5) Regime-map streamlining

- Main-text regime-map emphasis now centers on Figure 7 (jump vs persistence).
- Figures 6 and 8 are framed as complementary appendix candidates in narrative text.

## 6) Reachability sequencing cleanup

- Kept Section 4 example and added forward reference clarifying that `mlp_bce` is defined in Section 5 (Option B).

## 7) Figure-path cleanup

- Removed manuscript-body filesystem path references from figure/table captions.
- Captions now use manuscript figure/table labels only.

## 8) Companion artifacts added

- `paper/model_naming_audit.md`
- `paper/figure_main_vs_appendix_recommendation.md`
- `paper/reviewer_feedback_resolution_round1.md`

## What did not change

- No new experiments.
- No new metrics.
- No new contributions.
- No expansion of scientific scope.
