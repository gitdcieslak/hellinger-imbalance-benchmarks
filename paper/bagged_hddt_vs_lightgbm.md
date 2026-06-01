# Bagged HDDT vs LightGBM: Comparative Exposition

## Technical Interpretation

Using `paper/manuscript_tables/table_allocator_family_summary_v2.md`, Bagged HDDT (`hddt_forest`) and LightGBM have similar ranking quality (`AUROC=0.8389` vs `0.8479`, `AP=0.3434` vs `0.3462`), but materially different operational-accessibility geometry:

- **Threshold-survival accessibility persistence:** `0.5397` (Bagged HDDT) vs `0.4413` (LightGBM)
- **Operational smoothness:** `0.1741` (Bagged HDDT) vs `0.2624` (LightGBM)
- **Max recall jump:** `0.6100` (Bagged HDDT) vs `0.2218` (LightGBM)
- **Recovery (`Recall@0.01 - Recall@0.50`):** `0.9024` (Bagged HDDT) vs `0.5485` (LightGBM)

Interpretation: both models rank well, but they differ in operational profile. Bagged HDDT shows higher threshold-survival persistence yet much sharper threshold volatility (`max_recall_jump=0.6100`) and lower smoothness (`0.1741`), while LightGBM shows lower volatility (`0.2218`) and higher smoothness (`0.2624`) with lower net recovery pressure. This remains a direct empirical non-equivalence example: near-neighbor ranking performance does not imply near-neighbor operational controllability.

## Reviewer-Friendly Interpretation

If a reviewer looks only at AUROC/AP, Bagged HDDT and LightGBM look nearly tied. But when threshold policy changes, they behave very differently:

- Bagged HDDT behaves like a **volatile high-recovery model**: minority access can recover strongly but through larger threshold jumps.
- LightGBM behaves more **conservatively and steadily**: accessibility changes are less abrupt, with smoother threshold response.

So the practical decision question changes from “which model ranks better?” to “which model is easier to control safely under threshold adjustments?”

## Manuscript Paragraph Draft

Bagged HDDT and LightGBM provide a compact non-equivalence example in our family-level results. Their ranking summaries are close (`AUROC 0.8389` vs `0.8479`; `AP 0.3434` vs `0.3462`), yet their operational accessibility behavior diverges substantially. Bagged HDDT shows higher threshold-survival persistence (`0.5397`) but lower operational smoothness (`0.1741`) and higher jump intensity (`max_recall_jump=0.6100`), whereas LightGBM shows lower persistence (`0.4413`) but smoother threshold evolution (`0.2624`) and lower jump intensity (`0.2218`). This contrast indicates that similar ranking quality can mask materially different threshold-policy controllability under severe imbalance.

## Figure Callout Draft

- In `paper/manuscript_figures/figure_ranking_vs_accessibility_scatter.png`, Bagged HDDT and LightGBM appear close on AUROC yet remain operationally separable in smoothness/jump behavior.
- In `paper/manuscript_figures/figure_family_reachability_comparison.png`, their trajectory shapes diverge: Bagged HDDT exhibits sharper accessibility redistribution, while LightGBM follows a more conservative transition profile.

Suggested in-text callout sentence:

“Bagged HDDT and LightGBM illustrate the central non-equivalence signal: despite near-matched AUROC/AP, their smoothness and jump structure differ enough to imply different threshold-policy risk profiles.”

## Why This Example Matters

This pair is one of the cleanest motivating examples in the manuscript because it minimizes the usual objection (“you are just comparing a weak model to a strong model”). Here, ranking is similar, but operational behavior is not. That makes the inference sharper:

1. The accessibility argument is not a proxy for poor discrimination.
2. Trajectory-aware evaluation adds decision-relevant information beyond AUROC/AP.
3. The later neural perturbation section reads as a mechanism-oriented extension, not the first evidence of the phenomenon.

## Assessment

Yes: **Bagged HDDT vs LightGBM is a strong candidate for the cleanest motivating example** of ranking-quality similarity with operational-accessibility divergence in the current manuscript evidence stack.
