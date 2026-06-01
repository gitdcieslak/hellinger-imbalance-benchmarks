# Recommended Contribution Statement

This document provides manuscript-ready contribution language calibrated to the current evidence base.

## Primary Contributions

1. **Evidence of metric non-equivalence under severe imbalance.**
   We show that strong ranking and calibration summaries can coexist with substantial degradation in threshold-mediated minority accessibility, indicating that AUROC/AP/ECE/Brier alone may underdescribe operational behavior in this regime.

2. **Trajectory-centric operational analysis.**
   We operationalize minority accessibility as a threshold trajectory (reachability-focused), and use elasticity localization and persistence/smoothness descriptors to characterize where accessibility is gained, lost, or stabilized.

3. **Fixed-architecture imbalance-pressure sensitivity result.**
   Under a controlled MLP architecture, changing objective/sampling pressure (BCE -> oversampled/weighted) systematically shifts accessibility morphology from cliff-like to smoother and more controllable trajectories.

4. **Calibration interaction finding.**
   In this setting, reliability improvements (ECE/Brier) can coincide with reduced operational smoothness and larger local accessibility jumps, supporting a multi-axis evaluation view.

## Secondary Contributions

1. **Provisional morphology vocabulary.**
   We introduce an interpretable shorthand (e.g., cliff-like, smooth-elevated, broad-ramp) for recurring trajectory patterns to improve comparative discussion.

2. **Operational framing for threshold policy.**
   We connect trajectory behavior to threshold-adjustment controllability, highlighting practical implications for systems that tune operating points under class imbalance.

3. **Structured framing with explicit limits.**
   We position the work as an empirical and conceptual framework contribution, not a new learner, full theory, or universal taxonomy.

## Explicit Non-Claims

- We do **not** propose a new classification algorithm or training objective.
- We do **not** claim a complete formal theory of accessibility morphology.
- We do **not** claim universal generalization across all datasets, model families, or imbalance regimes.
- We do **not** claim that calibration improvements are generally harmful.
- We do **not** claim a definitive causal mechanism for Bagged HDDT accessibility behavior.

## Manuscript-Ready Compact Version

Use this short block where space is limited (e.g., introduction contribution bullets):

1. We provide evidence that, under severe class imbalance, ranking and calibration summaries can underdescribe threshold-mediated minority accessibility.
2. We contribute a trajectory-centric analytical lens (reachability, elasticity localization, persistence/smoothness) for operational accessibility assessment.
3. We show fixed-architecture MLP accessibility morphology shifts under imbalance-pressure perturbations (BCE vs oversampled/weighted).
4. We document a calibration interaction in which reliability gains can coexist with degraded operational smoothness in this setting.
5. We frame these as empirical and conceptual contributions with explicit non-claims on universality and theory completeness.

## Wording Guardrails

Prefer:

- “in this severe-imbalance setting”
- “current evidence suggests”
- “partially non-equivalent”
- “provisional morphology vocabulary”

Avoid:

- “proves universal independence”
- “new foundational mathematical object”
- “calibration is operationally harmful”
- “complete theory of accessibility behavior”
