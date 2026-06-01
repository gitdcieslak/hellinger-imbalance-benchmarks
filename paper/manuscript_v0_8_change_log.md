# manuscript_v0_8 Change Log

Base file: `paper/manuscript_v0_7.md`
Output file: `paper/manuscript_v0_8.md`
Pass type: external-circulation cleanup and review hardening

## Summary

This pass removes internal draft markers, adds one concise running-example anchor, and preserves all existing claims/scope.

## 1) TODO / internal-marker cleanup

- Removed manuscript-body TODO items:
  - methods wording verification note,
  - appendix recurring-pattern expansion note,
  - synthesis figure/table reminder.
- Removed end-of-manuscript internal `Figure and Citation Checklist` block.

## 2) Converted notes

- Added `paper/future_work_notes.md` to preserve useful non-manuscript-ready follow-up items from removed TODO markers.

## 3) Running example anchor

- Added concise callout in Section 6:
  - **Running Example: Bagged HDDT vs LightGBM**
  - emphasizes similar ranking quality with different accessibility geometry using already-reported values.

## 4) Citation closure confirmation

- No new citation gaps found in v0.8.
- Existing closure status retained from `paper/references_audit.md`.

## 5) Figure/table cross-reference validation

- Added `paper/figure_reference_audit.md` confirming referenced assets exist and numbering is consistent.

## 6) Review readiness artifact

- Added `paper/review_readiness_checklist.md` with pass/fail status on placeholders, citations, claims, references, and terminology consistency.

## What did not change

- No new experiments.
- No new claims.
- No new manuscript sections.
