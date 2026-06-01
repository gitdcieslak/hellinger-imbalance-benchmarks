# Claim Coverage Matrix

| Claim | Figure | Coverage Strength |
| --- | --- | --- |
| Claim 1 — Ranking quality is not operational accessibility | Figure 3, Figure 11, Figure 1 | Strong |
| Claim 2 — AUROC/AP can obscure default-threshold collapse | Figure 3, Figure 11 | Strong |
| Claim 3 — Reachability adds trajectory information | Figure 1, Figure 2, Figure 4 | Strong |
| Claim 4 — Distinct recurring morphology regimes appear | Figure 5, Figure 11 | Moderate |
| Claim 5 — CART-like quantized behavior appears | Figure 5, Figure 11 (model-specific panel needed) | Weak |
| Claim 6 — XGBoost cliff-like collapse appears consistently | Figure 11 (model-specific panel), Figure 5 | Moderate |
| Claim 7 — HDDT / Bagged HDDT broad persistence | Figure 5, Figure 8 (family-specific occupancy panel recommended) | Weak |
| Claim 8 — Baseline MLP analyzable in same framework | Figure 3, Figure 11 | Moderate |
| Claim 9 — BCE-trained MLP is cliff-like | Figure 1, Figure 2, Figure 6 | Strong |
| Claim 10 — Oversampling/weighting move MLP cliff -> smooth | Figure 1, Figure 2, Figure 6, Figure 4 | Strong |
| Claim 11 — Morphology is not architecture-only | Figure 6, Figure 1, Figure 4 | Strong |
| Claim 12 — Smoothness not reducible to support breadth alone | Figure 7, Figure 4 | Strong |
| Claim 13 — Calibration can improve ECE/Brier while worsening morphology | Figure 9, Figure 10 | Strong |
| Claim 14 — Calibration quality and smoothness partially non-equivalent | Figure 9, Figure 10 | Strong |
| Claim 15 — Regime taxonomy is provisional and empirical | Figure 5 (with explicit caveat in caption/body) | Moderate |
| Claim 16 — Empirical + conceptual framing, not complete theory | Figure support is indirect; communicated primarily via scope language | Weak |
| Claim 17 — Bagged HDDT mechanism unresolved | No direct figure; best handled as limitation/future-work note | Missing |
| Claim 18 — Reachability-aware objectives as cautious future direction | No current figure (future experiment required) | Missing |

## Coverage Summary

- Strongly visualized core claims: 1, 2, 3, 9, 10, 11, 12, 13, 14.
- Moderately visualized claims needing tighter model-family panels: 4, 6, 8, 15.
- Weak or non-visual claims mostly rhetorical/scope-oriented or unresolved: 5, 7, 16, 17, 18.

# Missing Figure Opportunities

## Claims lacking strong visuals

- Claim 5 (CART quantized behavior): add a dedicated model-family threshold-evolution small-multiple.
- Claim 7 (HDDT/bagged HDDT broad persistence): add explicit family comparison panel separating `hddt`, `bagged_hddt`, and `hddt_forest` to avoid naming drift confusion.
- Claim 16 (framing scope): not naturally figure-first; keep as explicit text box/scope table.
- Claim 17 and Claim 18: genuinely future-work claims; absence of strong figures is expected.

## Figures worth generating later

- Family-specific threshold evolution atlas (CART, XGBoost, LightGBM, HDDT, bagged HDDT, MLP variants) with shared axes.
- Raw vs calibrated reachability overlays per model to complement aggregate calibration deltas.
- Threshold-grid sensitivity analysis figure (coarse vs finer grids) to bound localization dependence.
- Mechanism probe plot for bagged HDDT (if follow-up experiments separate bagging vs split criterion effects).

## Appendix candidates

- Figure 6 (transition composite alternative layouts).
- Figure 8 (occupancy persistence/support breadth extended panels).
- Figure 10 (raw vs calibrated regime-shift examples as full table panel).
- Figure 11 (full recall-evolution panel set, with main text showing only representative slices).

## Visual Narrative Alignment Check

- Reachability arc: Figure 1 -> Figure 2 establishes global then dataset-conditioned trajectory behavior.
- Accessibility dynamics arc: Figure 3 -> Figure 4 -> Figure 11 shows collapse, localization, and recall evolution.
- Morphology arc: Figure 5 and Figure 6 provide recurring pattern framing and controlled MLP transition.
- Occupancy mechanism arc: Figure 7 and Figure 8 support persistence/support interpretation.
- Calibration arc: Figure 9 and Figure 10 close with reliability-accessibility non-equivalence.

This sequence aligns with `paper/manuscript_v0_2.md` section flow and supports all major empirical claims with at least one direct visual anchor.
