# External Review Packet

## 1) Paper Summary

This manuscript studies evaluation under severe class imbalance for threshold-mediated deployment settings. The core claim is that ranking quality and calibration quality can be partially non-equivalent to threshold-mediated minority accessibility behavior. We frame this behavior through reachability trajectories and related operational diagnostics (elasticity localization, smoothness, jump intensity, threshold-survival persistence).

Scope is intentionally bounded:

- empirical + conceptual framing paper,
- constrained dataset/model slice,
- no new learner/theory/taxonomy claim.

## 2) Contribution Summary

### Empirical

- Family-level accessibility trajectories differ materially under shared protocol.
- Similar ranking summaries can map to different operational behavior.
- Fixed-architecture MLP perturbations (`mlp_bce -> mlp_oversampled/mlp_weighted`) produce cliff-to-smoother transitions.
- Calibration interaction: ECE/Brier gains can coincide with degraded smoothness/jump profile in this setting.

### Analytical

- Reachability trajectory framing (`R(t)=P(\hat p(x)\ge t|y=1)`).
- Elasticity localization and max-jump diagnostics.
- Threshold-survival persistence framing for operational controllability.

### Conceptual

- Operational accessibility as a first-class deployment interpretation object.
- Ranking, calibration, and accessibility as separate but interacting views.

## 3) Evidence Hierarchy

The manuscript now follows this order:

1. **Family-level accessibility phenomena** (non-neural baseline families).
2. **Reachability lens** to interpret threshold evolution.
3. **Neural morphology transition** as controlled mechanism-oriented extension.
4. **Calibration interaction** as additional non-equivalence axis.

This ordering is intended to avoid MLP-centric interpretation.

## 4) Figure Gallery

### Reachability

- `paper/manuscript_figures/figure_1_reachability_mean_transition.png`
- `paper/manuscript_figures/figure_2_reachability_by_dataset.png`

Use: establish threshold-evolution behavior and dataset-modulated heterogeneity.

### Family comparison

- `paper/manuscript_figures/figure_family_reachability_comparison.png`
- `paper/manuscript_figures/figure_ranking_vs_accessibility_scatter.png`

Use: show family-level divergence and ranking/accessibility non-equivalence.

### Regime map (persistence-remediated)

- `paper/manuscript_figures/figure_accessibility_regime_map_v2.png`
- `paper/manuscript_figures/figure_accessibility_regime_map_variant_a_v2.png`
- `paper/manuscript_figures/figure_accessibility_regime_map_variant_b_v2.png`

Use: test whether recurring patterns occupy structured regions in operational metric space.

### Calibration

- `paper/manuscript_figures/figure_5_calibration_geometry_deltas.png`

Use: bounded evidence of calibration-reliability vs accessibility-geometry tension.

## 5) Anchor Example: Bagged HDDT vs LightGBM

This is a motivating pair because ranking summaries are close while operational behavior diverges.

- Similar ranking: AUROC/AP are near-neighbors.
- Different operational profile: smoothness, jump intensity, recovery, and threshold-survival persistence differ.

This supports the central thesis without relying on “weak vs strong model” contrast.

## 6) Metrics Clarification (Important)

Persistence semantics were remediated:

- **Current persistence meaning:** threshold-survival persistence across threshold grid.
- **Low-score mass (`<0.01`)** retained only as secondary concentration descriptor.

This change is intended to align metric naming with natural reviewer interpretation.

## 7) Reviewer Questions

For ML researchers:

1. Is the reachability framing useful, or should the paper lean harder on class-conditional coverage language?
2. Are the operational metrics meaningful and interpretable as defined now?
3. Is morphology language justified as provisional empirical shorthand?
4. What literature are we missing (especially selective prediction overlap and decision-focused calibration work)?

Additional targeted questions:

5. Is the evidence hierarchy convincing (family-level first, neural second, calibration third)?
6. Does Bagged HDDT vs LightGBM function as a clean motivating example?
7. Where does the current exposition still risk overclaim or misread?

## 8) Risk Register

### Selective prediction overlap

- Risk: reachability read as rebranded coverage.
- Mitigation: explicit overlap acknowledgment; operational reinterpretation framing.

### Terminology concerns

- Risk: morphology labels interpreted as ontology.
- Mitigation: consistent “recurring empirical patterns” language + non-claims.

### External validity

- Risk: constrained dataset/model slice.
- Mitigation: bounded claims, explicit limitations, replication-forward future work.

### Metric semantics

- Risk: persistence confusion.
- Mitigation: v2 persistence definition and explicit low-score-mass separation.

## 9) Open Questions for Next Iteration

1. Do we need one additional cross-domain replication before main-track submission?
2. Should we add a compact appendix translating metrics to queue/KPI intuition?
3. Should we include a short “mapping to selective prediction terminology” subsection/table?
4. Should we add one calibration robustness replication (extra calibrator/split regime)?

## 10) What We Need From You

Please focus feedback on:

- thesis clarity,
- evidence credibility,
- terminology safety,
- and likely reviewer attack points.

If possible, provide:

1. one “strong accept” argument,
2. one “major concern” argument,
3. one concrete revision that would most improve acceptance odds.
