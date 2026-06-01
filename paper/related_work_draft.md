## Related Work

### 1. Severe Class Imbalance Learning

Research on severe class imbalance has long emphasized cost-sensitive learning, reweighting, resampling, and specialized tree/ensemble approaches for rare-event detection [CITATION: severe imbalance survey], [CITATION: cost-sensitive learning foundation], [CITATION: oversampling foundation], [CITATION: HDDT original paper]. This literature establishes both the practical importance of skewed-class settings and the sensitivity of minority performance to training pressure.

Our manuscript is directly situated in this setting and uses this tradition’s core tools (weighting, oversampling, threshold-based evaluation). We therefore build upon imbalance learning practice rather than replace it. The difference in emphasis is that we do not propose a new learner; instead, we analyze how minority accessibility behaves under threshold evolution in deployment-relevant conditions.

This distinction is important for positioning. In imbalance work, improvements are often interpreted through endpoint metrics or algorithmic gains. Here, the contribution is empirical and conceptual: we examine trajectory behavior and operational controllability under severe skew, including fixed-architecture perturbations (`mlp_bce`, `mlp_oversampled`, `mlp_weighted`) that change accessibility morphology without changing architecture class.

### 2. Evaluation Under Severe Imbalance

Classifier evaluation under imbalance commonly relies on AUROC, average precision, precision-recall summaries, and selected threshold operating points [CITATION: PR vs ROC under imbalance], [CITATION: threshold evaluation under class imbalance]. These metrics remain valuable and we use them throughout.

Our results do not dispute their usefulness; rather, they suggest they may be insufficient for threshold-mediated deployment interpretation on their own. Because these metrics summarize discrimination at either aggregate or selected operating points, they may not reveal where accessibility transitions concentrate across threshold space.  In our setting, models with acceptable ranking quality can still show default-threshold accessibility collapse, with substantial recovery only under threshold relaxation. This motivates an explicit trajectory-centered view of minority accessibility.

Accordingly, we treat standard ranking metrics as part of a broader operational assessment stack, alongside reachability trajectories, elasticity localization, occupancy persistence, and calibration interaction. The framing is additive: it extends in emphasis from static endpoints to threshold-evolution behavior.

### 3. Selective Prediction, Coverage, and Threshold Analysis

Our closest conceptual neighbors are reject-option classification, selective prediction, and risk-coverage analysis [CITATION: reject-option foundation], [CITATION: selective prediction foundation], [CITATION: risk-coverage analysis]. These traditions study confidence-thresholded action sets and how predictive risk changes as coverage varies.

We explicitly acknowledge this overlap. Reachability, as used here, is closely related to class-conditional coverage/threshold trajectory analysis. A selective-prediction reviewer could reasonably interpret `R(t)=P(\hat p(x)\ge t\mid y=1)` as a minority-conditioned coverage trajectory.

Where our focus differs is operational emphasis under severe imbalance:

- minority accessibility persistence rather than aggregate acceptance coverage,
- localization of threshold sensitivity (elasticity concentration and jump placement),
- and threshold-policy robustness in constrained deployment settings.

In other words, we do not position reachability as disconnected from coverage traditions. We position it as an operational reinterpretation and elevation of class-conditional threshold trajectories for severe imbalance analysis.

This also addresses a likely reviewer question (“Is reachability just class-conditional coverage?”). Our answer is: it is closely related in mathematical form, but the manuscript contributes an integrated empirical perspective that couples minority-conditioned trajectory behavior with occupancy, smoothness, and calibration interaction in threshold-mediated deployment contexts.

### 4. Calibration and Decision-Focused Probability Estimation

Calibration literature has established post-hoc methods and reliability diagnostics such as Platt scaling, isotonic regression, ECE, and Brier score [CITATION: Platt scaling], [CITATION: isotonic calibration], [CITATION: calibration foundations], [CITATION: modern calibration analysis]. We rely on these standard tools and treat calibration as valuable.

Our contribution in this area is not a new calibration method. Instead, we examine how calibration interacts with operational accessibility trajectories under severe imbalance. Current evidence suggests that reliability improvements (lower ECE/Brier) can coexist with less smooth threshold-accessibility behavior in the same setting. To our knowledge, this interaction is not typically foregrounded in standard calibration reporting and motivates treating reliability and accessibility as partially distinct operational considerations.

This is a careful non-equivalence claim, not an anti-calibration claim. Calibration remains important for probability quality; our findings suggest that reliability and operational threshold controllability may be partially distinct axes in some severe-imbalance deployments.

This framing is related to decision-focused probability estimation, but extends it by emphasizing trajectory morphology and accessibility persistence under threshold movement rather than reliability at a single decision surface.

### 5. Decision-Theoretic and Operational Perspectives

Decision-theoretic classification and utility-sensitive thresholding emphasize choosing thresholds under asymmetric costs and action utilities [CITATION: cost-sensitive threshold theory], [CITATION: decision-theoretic classification]. Operational ML traditions similarly emphasize deployment constraints, human-in-the-loop workflows, and threshold policy management in triage-like systems [CITATION: operational ML monitoring], [CITATION: human-in-the-loop triage], [CITATION: queue-aware ML deployment].

Our perspective is complementary to both. Decision-theoretic work often centers on selecting good thresholds for a utility model. We focus on how minority accessibility evolves as thresholds change, because real systems frequently adjust threshold policy in response to workload, precision targets, or operational incidents.

This is where accessibility persistence and operational smoothness become deployment-relevant: they describe whether policy adjustments produce controlled changes or cliff-like shifts. In this sense, the manuscript is aligned with operational ML concerns about controllability and stability, while retaining a model-evaluation orientation.

### 6. Summary Positioning

Taken together, this manuscript is most closely related to severe imbalance evaluation, selective prediction/coverage analysis, threshold-sensitive decision analysis, calibration-for-decision discussions, and operational ML deployment thinking. Many of the analytical objects discussed here have conceptual antecedents in selective prediction, coverage analysis, calibration, and decision-theoretic classification. Our contribution is primarily an empirical and operational reframing that brings these perspectives together around minority accessibility trajectories under severe imbalance.

We do not claim a new learner, a complete operational morphology theory, or a universal taxonomy. We also do not frame the contribution as a benchmark competition.

Instead, we provide an empirical and operationally grounded perspective on minority accessibility trajectories under severe class imbalance, with a focus on threshold-evolution behavior, optimization-sensitive morphology transitions, and calibration-accessibility interaction.
