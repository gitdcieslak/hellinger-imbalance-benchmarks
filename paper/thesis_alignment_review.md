# Thesis Alignment Review

## Section A — Thesis Statement

1. What the paper currently appears to claim:
   - The manuscript argues that under severe imbalance, ranking and calibration metrics can miss threshold-mediated minority accessibility behavior.
   - It also presents reachability/morphology language, family-level regime contrasts, MLP perturbation transitions, and calibration-accessibility interaction.
2. What the paper should claim:
   - Primary claim should remain: conventional performance summaries (ranking + reliability) are insufficient to characterize threshold-mediated minority accessibility trajectories under severe imbalance.
3. Do these differ?
   - Slightly. Current text is mostly aligned, but some sections can read as "framework/terminology introduction" first and "empirical non-equivalence" second.

Primary claim:

- Empirical non-equivalence between conventional summaries and threshold-mediated minority accessibility.

Secondary claims:

- Reachability trajectories are a useful analytical lens.
- Family-level evidence demonstrates cross-model divergence beyond neural-only settings.
- Fixed-architecture MLP perturbation changes trajectory morphology.
- Calibration improvements can coincide with worse accessibility controllability.

Supporting observations:

- Bagged HDDT vs LightGBM running example.
- Stability check under bootstrap/leave-one-dataset-out.
- Oil near-flat trajectory as a boundary case for threshold leverage.

---

## Section B — Section-by-Section Audit

| Section | Role | Contribution to Thesis | Classification | Recommendation |
| --- | --- | --- | --- | --- |
| Abstract | Thesis summary | Direct | Core Thesis | Keep; continue leading with phenomenon before machinery |
| Introduction | Problem setup + thesis declaration | Direct | Core Thesis | Keep |
| Related Work | Positioning + overlap guardrails | Indirect | Context | Keep, but keep compact and overlap-explicit |
| Reachability | Defines key object and distinguishes from single-threshold recall | Direct | Necessary Machinery | Keep; this is required for interpreting evidence |
| Accessibility Metrics | Persistence/smoothness/elasticity semantics | Indirect | Necessary Machinery | Keep but compress repeated metric caveats |
| Family-Level Evidence | Main empirical support across model families | Direct | Core Evidence | Keep and prioritize early in empirical flow |
| Bagged HDDT | High-salience anchor example | Direct | Core Evidence | Keep as running example; avoid over-expanding mechanism prose in main text |
| MLP Perturbation | Controlled supporting test of morphology sensitivity | Direct | Supporting Evidence | Keep; ensure it supports (not replaces) family-level thesis |
| Calibration | Distinct operational-axis evidence | Direct | Core Evidence | Keep and emphasize as major finding |
| Discussion | Synthesis and implications | Direct | Core Thesis | Keep; prune any framework-heavy digressions |
| Conclusion | Final thesis reinforcement | Direct | Core Thesis | Keep and keep non-equivalence sentence first |

High-level judgment:

- The manuscript is largely thesis-aligned.
- Dilution risk comes from over-weighting terminology/morphology scaffolding relative to non-equivalence evidence.
