# Beyond AUROC and Calibration: Operational Accessibility Trajectories Under Severe Class Imbalance

## Abstract

Severe class imbalance is common in threshold-mediated deployments such as triage, fraud review, and sparse-event surveillance, yet evaluation is often dominated by static ranking and reliability metrics. Current evidence from repeated-split severe-imbalance experiments suggests that AUROC, average precision, ECE, and Brier can be partially non-equivalent to threshold-mediated minority accessibility in this setting. We analyze reachability trajectories, closely related to class-conditional coverage trajectories but used here as an operational deployment object under severe imbalance, to characterize accessibility evolution across threshold policy changes. Using this lens, we observe recurring empirical patterns, including cliff-like and smoother transitions, and show that within a fixed MLP architecture, objective/sampling perturbations (`mlp_bce` vs `mlp_oversampled`/`mlp_weighted`) can move accessibility morphology from cliff-like to smoother forms. We also observe a calibration interaction: reliability metrics can improve while operational smoothness declines and jump intensity increases in this constrained severe-imbalance setting. We present these results as an empirical + conceptual framing contribution rather than a complete theory or universal taxonomy. This manuscript is not a new learner, not a complete operational morphology theory, not a universal classifier taxonomy, and not a benchmark competition paper.

## 1. Introduction

How should we reason about operational accessibility under severe class imbalance? In many real systems, decisions are threshold-mediated rather than rank-only: teams tighten or relax thresholds to manage review load, precision constraints, or policy shifts. In that context, the key deployment question is not only whether positives are ranked above negatives, but whether minority cases remain accessible as threshold policy changes.

Our empirical program repeatedly surfaced a consistent signal: ranking quality does not reliably determine threshold-level accessibility in severe-imbalance settings. In our MLP baseline analysis over `boundary`, `cam`, `compustat`, `oil`, and `satimage`, dataset-mean values show `auroc_mean=0.7068` and `average_precision_mean=0.2410`, while `recall@0.50=0.1346` and `recall@0.01=0.7867` (recovery `+0.6522`) [source: `reports/neural_mlp_allocation_geometry_summary.md`]. On several severe datasets in the same run, default-threshold recall collapses near zero while low-threshold recovery remains substantial.

To study this behavior, we treat threshold-mediated accessibility trajectories as operationally relevant analytical objects. We use reachability trajectories, elasticity localization, and accessibility persistence/operational smoothness to describe where accessibility is lost, where it recovers, and how controllable threshold policy appears under perturbation.

Our contributions are:

1. We provide evidence that, under severe class imbalance, ranking and calibration summaries can underdescribe threshold-mediated minority accessibility.
2. We contribute a trajectory-centric analytical lens (reachability, elasticity localization, persistence/smoothness) for operational accessibility assessment.
3. We show fixed-architecture MLP accessibility morphology shifts under imbalance-pressure perturbations (BCE vs oversampled/weighted).
4. We document a calibration interaction in which reliability gains can coexist with degraded operational smoothness in this setting.
5. We frame these as empirical and conceptual contributions with explicit non-claims on universality and theory completeness.

### Scope delimitations

This manuscript is:

- an empirical + conceptual framing paper,
- focused on severe imbalance and threshold-mediated deployment,
- based on constrained model and dataset slices.

This manuscript is not:

- a complete operational morphology theory,
- a universal classifier taxonomy,
- a formal topology paper,
- a benchmark leaderboard paper,
- or a production-ready objective proposal.

## 2. Related Work

### 2.1 Severe Class Imbalance Learning

Research on severe class imbalance has long emphasized cost-sensitive learning, reweighting, resampling, and specialized tree/ensemble approaches for rare-event detection [CITATION: severe imbalance survey], [CITATION: cost-sensitive learning foundation], [CITATION: oversampling foundation], [CITATION: HDDT original paper]. This literature establishes both practical importance and minority-performance sensitivity to training pressure.

Our manuscript is directly situated in this setting and uses these core tools (weighting, oversampling, threshold-based evaluation). We do not propose a new learner; we analyze how minority accessibility behaves under threshold evolution in deployment-relevant conditions.

### 2.2 Evaluation Under Severe Imbalance

Classifier evaluation under imbalance commonly relies on AUROC, average precision, precision-recall summaries, and selected threshold operating points [CITATION: PR vs ROC under imbalance], [CITATION: threshold evaluation under class imbalance]. These metrics remain valuable and are used throughout.

Our results do not dispute their usefulness; they suggest these summaries can be insufficient on their own for threshold-mediated deployment interpretation. In this setting, models with acceptable ranking quality can still show default-threshold accessibility collapse, with substantial recovery only under threshold relaxation.

### 2.3 Selective Prediction, Coverage, and Threshold Analysis

Our closest conceptual neighbors are reject-option classification, selective prediction, and risk-coverage analysis [CITATION: reject-option foundation], [CITATION: selective prediction foundation], [CITATION: risk-coverage analysis]. These traditions analyze confidence-thresholded action sets and risk/coverage behavior.

We explicitly acknowledge overlap. Reachability, as used here, is closely related to class-conditional coverage/threshold trajectory analysis (`R(t)=P(\hat p(x)\ge t\mid y=1)`). Our emphasis differs in operational focus under severe imbalance: minority accessibility persistence, elasticity localization, and threshold-policy robustness.

### 2.4 Calibration and Decision-Focused Probability Estimation

Calibration literature has established post-hoc methods and reliability diagnostics such as Platt scaling, isotonic regression, ECE, and Brier score [CITATION: Platt scaling], [CITATION: isotonic calibration], [CITATION: calibration foundations], [CITATION: modern calibration analysis]. We rely on these tools, and calibration remains valuable for probability reliability.

Our contribution is not a new calibration method. We study calibration-accessibility interaction under severe imbalance, where reliability improvements can coexist with less smooth threshold-accessibility behavior.

### 2.5 Decision-Theoretic and Operational Perspectives

Decision-theoretic classification and utility-sensitive thresholding emphasize threshold choice under asymmetric costs [CITATION: cost-sensitive threshold theory], [CITATION: decision-theoretic classification]. Operational ML traditions emphasize deployment constraints, human-in-the-loop workflows, and threshold policy management [CITATION: operational ML monitoring], [CITATION: human-in-the-loop triage], [CITATION: queue-aware ML deployment].

Our perspective is complementary: we focus on accessibility evolution as thresholds move, because real systems regularly adjust policy in response to workload or risk constraints.

### 2.6 Summary Positioning

This manuscript is most closely related to severe-imbalance evaluation, selective prediction/coverage analysis, threshold-sensitive decision analysis, calibration-for-decision discussions, and operational ML deployment thinking. We do not claim a new learner or complete theory; we provide an empirical and operationally grounded framing for minority accessibility trajectories under severe class imbalance.

## 3. Operational Accessibility Under Severe Imbalance

Under severe imbalance, deployment systems are often constrained by review capacity, cost asymmetry, and threshold policy. In these contexts, operationally relevant performance is not just class ordering but accessibility persistence: whether minority cases remain available to action as thresholds move.

This distinction matters because threshold policies are not static in practice. Teams tighten or relax thresholds due to queue pressure, precision constraints, policy updates, or incident response. If behavior is cliff-like, small policy changes can produce disproportionate accessibility shifts. If behavior is smoother, policy control is more stable.

We therefore use operational accessibility to refer to threshold-conditioned minority access, and treat ranking quality as necessary but insufficient for deployment interpretation.

Potential objection: “is this just threshold tuning?” Our answer is no in two senses. First, the object of study is trajectory shape, not only an optimized point. Second, models with similar ranking quality can produce sharply different accessibility trajectories, implying different policy controllability.

To anchor this framing, we provide a conceptual map from ranking/calibration summaries to threshold-mediated reachability and resulting accessibility persistence.

[Figure 0 about here]

**Figure 0:** Operational accessibility lens (`paper/manuscript_figures/figure_0_operational_accessibility_lens.png`).

## 4. Reachability and Threshold Accessibility

We define reachability as:

\[
R(t) = P(\hat{p}(x) \ge t \mid y=1)
\]

where `t` is a decision threshold and `\hat{p}(x)` is the model score interpreted on `[0,1]`.

Intuitively, `R(t)` answers: what fraction of minority instances remains operationally accessible at threshold `t`?

Recall at a fixed threshold is one point on `R(t)`. Reachability analysis studies trajectory behavior across thresholds, including:

- where accessibility loss concentrates,
- how quickly accessibility decays,
- whether transition behavior is smooth or phase-like,
- and how stable threshold-policy control is under perturbation.

This lens is closely related to class-conditional coverage trajectories, but we use it here as an operational deployment object centered on accessibility persistence, operational smoothness, and elasticity localization under severe imbalance.

In this project, threshold grid analyses (`0.50`, `0.25`, `0.10`, `0.05`, `0.01`) and interval derivatives reveal patterns that single-threshold recall cannot. In transition artifacts, `mlp_bce` often concentrates reachability gains in narrow intervals (e.g., strong `0.50->0.25` jump on `boundary`; large `0.05->0.01` jump on `compustat`), while perturbed variants distribute gains more broadly [source: `results/geometry_transition_analysis/reachability_derivatives.csv`].

To understand threshold-mediated accessibility evolution, we first examine aggregate reachability trajectories.

[Figure 1 about here]

**Figure 1:** Mean reachability transition (`paper/manuscript_figures/figure_1_reachability_mean_transition.png`).

To evaluate whether the aggregate pattern is stable or dataset-specific, we then inspect per-dataset reachability panels.

[Figure 2 about here]

**Figure 2:** Dataset-level reachability panels (`paper/manuscript_figures/figure_2_reachability_by_dataset.png`).

Operational implication: same-threshold recall values can conceal very different threshold sensitivity structures.

## 5. Experimental Framework

We use a constrained, reproducible framework built around repeated splits, fixed threshold sweeps, and shared metric/reporting pathways.

### Data and protocol

- Legacy severe-imbalance datasets emphasized in this paper: `boundary`, `cam`, `compustat`, `oil`, `satimage`.
- Repeated split style follows project legacy protocol (5 repeated splits used in current reports).
- Operational thresholds: `0.50`, `0.25`, `0.10`, `0.05`, `0.01`.

### Model slices used in manuscript argument

- Baseline family comparators: CART, HDDT variants, RandomForest, XGBoost, LightGBM.
- Neural extension: MLP baseline (`mlp`).
- Fixed-architecture perturbation: `mlp_bce`, `mlp_oversampled`, `mlp_weighted`.

### Analysis artifact families

- threshold sweep summaries,
- elasticity summaries,
- allocation concentration summaries,
- occupancy summaries,
- calibration interaction summaries,
- recurring-pattern synthesis outputs.

This is intentionally not a broad architecture benchmark. No new deep-learning stack, neural architecture zoo, or hyperparameter sweep campaign is required for the claims developed here.

TODO: verify manuscript methods table with exact split-seed wording from runner configs before external circulation.

## 6. Accessibility Trajectory Observations

The first empirical layer is ranking-accessibility divergence under threshold evolution.

From the MLP baseline report, dataset-mean ranking quality is moderate (`auroc_mean=0.7068`, `average_precision_mean=0.2410`), but default-threshold accessibility is low (`recall@0.50=0.1346`) with strong recovery under relaxation (`recall@0.01=0.7867`) [source: `reports/neural_mlp_allocation_geometry_summary.md`]. At dataset level, severe collapse cases appear (`boundary=0.0000`, `compustat=0.0000` at `0.50`) with large recovery under lower thresholds.

These observations are consistent with earlier model-family findings: high AUROC/AP can coexist with near-zero minority accessibility at operational defaults.

To ground the non-equivalence claim with concrete anchors, we next present representative threshold-collapse and recovery values.

[Table 1 about here]

**Table 1:** Ranking vs accessibility anchors (`paper/manuscript_tables/table_1_ranking_vs_accessibility.md`).

Potential objection: “are these isolated examples?” We mitigate that concern by repeated-split summaries and multi-dataset consistency in severe subsets, while treating generalization as bounded.

## 7. Recurring Morphology Patterns (Provisional)

After trajectory evidence is established, recurring empirical patterns become useful shorthand. We use descriptors such as quantized, cliff-like, smooth, broad, and conservative to summarize recurring behavior; we do not treat these as formal classes.

In baseline analyses, CART often aligns with quantized behavior (minimal threshold response), XGBoost is consistently cliff-like in current evidence, LightGBM appears conservative in the cited runs, and HDDT/bagged HDDT often align with broader accessibility patterns. These labels remain provisional empirical organization.

To localize where threshold sensitivity concentrates, we summarize interval-level elasticity patterns.

[Figure 3 about here]

**Figure 3:** Elasticity interval concentration heatmap (`paper/manuscript_figures/figure_3_elasticity_interval_heatmap.png`) used as recurring-pattern evidence rather than taxonomy proof.

Reviewer-facing caveat: these recurring empirical patterns are heuristic and threshold-grid dependent.

TODO(APPENDIX): include expanded recurring-pattern tables and alternative threshold-grid sensitivity checks.

## 8. Neural Perturbation and Morphology Transition

This section provides a key inferential pivot: with architecture held fixed, imbalance-pressure perturbation changes accessibility morphology.

From the perturbation summary (dataset means):

- `mlp_bce`: `AUROC=0.7068`, `AP=0.2410`, `recall@0.50=0.1346`, `recall@0.01=0.7867`, `operational_smoothness=0.4178`, `max_recall_jump=0.4061`, recurring pattern `cliff_allocator`.
- `mlp_oversampled`: `AUROC=0.8351`, `AP=0.2989`, `recall@0.50=0.4973`, `recall@0.01=0.7041`, `operational_smoothness=0.5391`, `max_recall_jump=0.0678`, recurring pattern `smooth_allocator`.
- `mlp_weighted`: `AUROC=0.7732`, `AP=0.2658`, `recall@0.50=0.5789`, `recall@0.01=0.8506`, `operational_smoothness=0.6080`, `max_recall_jump=0.1783`, recurring pattern `smooth_allocator`.

These shifts are consistent with imbalance-pressure sensitivity and support the claim that operational accessibility is not architecture-only.

To interpret mechanism-level structure, we next contrast support breadth with occupancy persistence.

[Figure 4 about here]

**Figure 4:** Support vs persistence scatter (`paper/manuscript_figures/figure_4_support_vs_persistence.png`).

Interpretive nuance from transition analysis: operational smoothness is not reducible to support breadth alone. `mlp_weighted` broadens support more strongly, but `mlp_oversampled` also smooths trajectories without equivalent support expansion. This is consistent with threshold-morphology redistribution as a primary mechanism.

Potential objection: “does oversampling merely shift thresholds?” Interval-derivative evidence suggests more than pure shift: jump localization and elasticity concentration change materially.

## 9. Calibration vs Operational Accessibility

This section consolidates a central finding: reliability improvement and accessibility behavior can diverge.

Across MLP perturbation variants, calibration summaries indicate ECE/Brier improvements, while transition deltas show negative smoothness shifts and positive jump shifts on average [source: `results/geometry_transition_analysis/calibration_transition_model_means.csv`]. In recurring-pattern persistence artifacts, `mlp_oversampled` and `mlp_weighted` can appear smooth in raw form but cliff-like after calibration transforms [source: `reports/neural_mlp_objective_perturbation/calibration_interaction/regime_persistence_table.csv`].

Reliability and accessibility appear to be partially non-equivalent operational axes in this severe-imbalance setting. Calibration remains valuable for probability reliability; the point is that reliability gains do not guarantee preservation of accessibility persistence or operational smoothness.

To make this interaction explicit, we next examine calibration geometry deltas.

[Figure 5 about here]

**Figure 5:** Calibration geometry deltas (`paper/manuscript_figures/figure_5_calibration_geometry_deltas.png`).

Operationally, this matters because teams may improve reliability metrics while unintentionally increasing threshold sensitivity and reducing accessibility controllability.

Potential objection: “is this just calibration pathology in one setup?” We treat this as bounded evidence requiring replication across calibrators and data regimes, not a universal anti-calibration claim.

## 10. Toward Operational Morphology

Current evidence supports four points. First, operational accessibility trajectories deserve direct analysis under severe imbalance. Second, recurring empirical patterns can summarize behavior when treated as provisional shorthand. Third, fixed-architecture perturbation results suggest imbalance-pressure mechanisms can materially alter behavior. Fourth, calibration-quality improvements can coexist with accessibility degradation.

Taken together, these findings support a synthesis in which ranking quality, calibration quality, and accessibility behavior are separate but interacting views of model behavior in threshold-mediated deployment.

Why should deployment teams care? In queue-managed review systems and human-in-the-loop triage, threshold policy is often a control knob. If accessibility behavior is cliff-like, small threshold moves can trigger large changes in minority access and workload composition. If behavior is smoother and persistence is higher, teams gain more stable control under policy adjustments. A trajectory-aware view therefore improves operational risk interpretation even when ranking and reliability summaries look acceptable.

We present operational morphology as an emerging framing perspective, not a complete formal theory. The intent is to improve deployment interpretability, sharpen hypothesis generation, and structure follow-up studies without overclaiming.

TODO: add one synthesis figure/table mapping each central claim to direct evidence artifact and confidence level.

## 11. Limitations

This draft has clear limitations.

1. **Dataset scope:** evidence comes from a constrained severe-imbalance legacy slice; external validity is unproven.
2. **Threshold grid dependence:** analyses use a fixed threshold set; finer grids may alter localization details.
3. **Terminology overlap risk:** reachability partially overlaps with selective prediction/class-conditional coverage traditions.
4. **Provisional labels:** recurring-pattern labels are heuristic shorthand, not formal ontology.
5. **Neural scope:** only sklearn-style MLP baseline and perturbations are included.
6. **Mechanism uncertainty:** bagged HDDT broad behavior is empirically visible but mechanistically unresolved.
7. **Calibration interaction uncertainty:** observed reliability-smoothness divergence may vary by calibrator/data scale.
8. **No formal theory claims:** the manuscript does not provide complete theoretical characterization or causal proof.

Accordingly, we do not claim universal behavior, complete morphology characterization, or definitive causal decomposition.

## 12. Future Work

We prioritize bounded extensions aligned to current evidence strength.

1. **Bagged HDDT mechanism disambiguation:** separate effects of Hellinger splitting, bagging stabilization, and feature/subsample structure.
2. **Calibration-morphology decomposition:** characterize when reliability correction re-steepens accessibility and when it does not.
3. **Reachability-aware objective exploration (cautious):** investigate whether trajectory-aware training pressure can improve operational controllability without overfitting threshold grids.
4. **Cross-domain replication:** test whether trajectory non-equivalence patterns hold in additional imbalance domains.

These are hypothesis-generating directions, not validated solutions.

## 13. Conclusion

In threshold-mediated severe-imbalance settings such as those studied here, operational accessibility should be analyzed as a first-class empirical object alongside ranking and reliability summaries. Current evidence suggests that static AUROC/AP and calibration metrics can underdescribe trajectory behavior that is operationally consequential. Reachability trajectories, elasticity localization, accessibility persistence, and operational smoothness provide a practical lens for this analysis.

Empirically, fixed-architecture MLP perturbations (`mlp_bce` vs `mlp_oversampled`/`mlp_weighted`) show cliff-to-smoother recurring-pattern transitions, supporting the view that accessibility morphology is sensitive to imbalance-pressure mechanisms rather than architecture alone. Calibration results further indicate that reliability gains can coexist with degraded accessibility smoothness in this setting, reinforcing that reliability and accessibility are partially non-equivalent operational axes.

This trajectory lens is closely related to class-conditional coverage analysis but emphasizes operational accessibility persistence and smoothness under threshold policy change. We present this work as an empirical + conceptual framing contribution, not a new learner, not a complete operational morphology theory, not a universal classifier taxonomy, and not a universal anti-calibration claim.

A natural next step is targeted replication and mechanism disambiguation across broader datasets, calibrators, and model families to determine when these recurring patterns persist and when they break.

---

## Figure and Citation Checklist

- Figure 0: `paper/manuscript_figures/figure_0_operational_accessibility_lens.png`
- Figure 1: `paper/manuscript_figures/figure_1_reachability_mean_transition.png`
- Figure 2: `paper/manuscript_figures/figure_2_reachability_by_dataset.png`
- Figure 3: `paper/manuscript_figures/figure_3_elasticity_interval_heatmap.png`
- Figure 4: `paper/manuscript_figures/figure_4_support_vs_persistence.png`
- Figure 5: `paper/manuscript_figures/figure_5_calibration_geometry_deltas.png`
- Table 1: `paper/manuscript_tables/table_1_ranking_vs_accessibility.md`

- [CITATION: severe imbalance deployment literature]
- [CITATION: HDDT original paper]
- [CITATION: calibration literature]
- [CITATION: threshold decision policy / risk management literature]
- [CITATION: class imbalance evaluation literature]
- [CITATION: selective prediction foundational paper]
