# External Review Packet (Short)

## What problem is being studied?

Under severe class imbalance, model quality is often summarized with ranking and calibration metrics (AUROC/AP/ECE/Brier). In threshold-mediated deployments (triage, review queues, sparse-event monitoring), teams actually act through thresholds. We study whether strong ranking/reliability summaries can still hide operational accessibility failures for minority cases.

## What was observed?

- Similar or acceptable ranking quality can coexist with very different threshold-accessibility behavior.
- Family-level trajectories show recurring operational patterns (quantized, cliff-like, broad, conservative) before any neural-specific analysis.
- In fixed-architecture MLP perturbations, changing imbalance pressure (BCE vs oversampled/weighted) shifts trajectories from cliff-like toward smoother behavior.
- Calibration can improve ECE/Brier while worsening smoothness/jump behavior in this setting.

## Why does it matter?

If two models look similar on AUROC/AP but react very differently to threshold changes, deployment risk differs even before utility optimization. The practical question is not just "which model ranks better?" but "which model is controllable under policy adjustments?"

## Core findings

1. Ranking/reliability and operational accessibility are partially non-equivalent under severe imbalance.
2. Reachability trajectories (`R(t)=P(\hat p(x)\ge t|y=1)`) provide useful threshold-evolution structure beyond single-threshold recall.
3. Family-level evidence establishes the phenomenon before neural perturbation analysis.
4. Bagged HDDT vs LightGBM is a compact motivating contrast: similar ranking summaries, different operational profiles.
5. Metric semantics were clarified: persistence now refers to threshold-survival persistence (not low-score mass concentration).

## Open questions

- How broadly does this pattern replicate across more imbalance domains?
- How sensitive are conclusions to threshold-grid choices and calibrator families?
- Can we connect these diagnostics to explicit queue/KPI simulation outcomes?
- What is the mechanism behind bagged HDDT behavior in this protocol?

## Requested feedback

1. Is the central thesis clear without overclaiming?
2. Is reachability framing useful, or should it be reframed in more standard coverage terms?
3. Are operational metrics intuitive and reviewer-safe after persistence remediation?
4. Does morphology language feel justified as provisional empirical shorthand?
5. What citations or adjacent literatures are missing for strongest reviewer resilience?
