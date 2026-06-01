# manuscript_v0_7 Change Log

Base file: `paper/manuscript_v0_6.md`
Output file: `paper/manuscript_v0_7.md`
Pass type: operational citation closure + HDDT lineage strengthening

## Summary

This pass closes unresolved operational-ML citation gaps, strengthens HDDT intellectual lineage context, and tightens Bagged HDDT interpretation guardrails without introducing new experiments or new claim classes.

## 1) Operational ML citation closure

- Replaced unresolved placeholders in Related Work Section 2.5:
  - `[CITATION: operational ML monitoring]`
  - `[CITATION: human-in-the-loop triage]`
  - `[CITATION: queue-aware ML deployment]`

with concrete references:

- `[CITATION: Sculley et al. 2015]`
- `[CITATION: Breck et al. 2017]`
- `[CITATION: Amershi et al. 2019]`

## 2) HDDT lineage strengthening

- Expanded Section 2.1 HDDT context to include:
  - original HDDT introduction (`Cieslak and Chawla 2008`),
  - robustness/skew-insensitivity framing (`Cieslak et al. 2012`),
  - explicit statement that prior HDDT work is not claimed to have studied accessibility geometry in current form.

## 3) Experimental framework enhancement

- Added explicit sentence in model-family description clarifying why HDDT-family models were included:
  - skew-insensitive design under severe imbalance makes them natural comparators for accessibility-oriented evaluation.

## 4) Citation checklist update

- Added citation checklist entries for:
  - `[CITATION: Cieslak et al. 2012]`
  - `[CITATION: Sculley et al. 2015]`
  - `[CITATION: Breck et al. 2017]`
  - `[CITATION: Amershi et al. 2019]`

## 5) Bibliography updates

Added to `paper/references.bib`:

- `CieslakEtAl2012`
- `Sculley2015`
- `Breck2017`
- `Amershi2019`

## 6) Companion review artifacts added/updated

- Updated `paper/references_audit.md` (all placeholders resolved).
- Added `research/hddt_lineage_notes.md`.
- Added `paper/reviewer_risk_check_v0_7.md`.

## What did not change

- No new model training, no new benchmark runs.
- No new primary contribution claims.
- No causal mechanism claims were added for Bagged HDDT.
