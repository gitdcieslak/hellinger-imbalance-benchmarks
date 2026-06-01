# Beyond AUROC and Calibration: Operational Accessibility Trajectories Under Severe Class Imbalance

## Abstract

Severe class imbalance is common in threshold-mediated deployments such as triage, fraud review, and sparse-event surveillance, yet model evaluation is often dominated by static ranking and reliability metrics. Current evidence from repeated-split legacy imbalance experiments suggests that AUROC, average precision, and calibration quality can be partially non-equivalent to operational accessibility. We frame reachability trajectories, defined as minority accessibility across threshold evolution, as a central analytical object for this setting. We observe recurring accessibility morphology patterns, including cliff-like and smoother transitions, and show that within a fixed MLP architecture, objective/sampling perturbations can move morphology from cliff-like to smoother forms. We also observe a calibration tension: ECE/Brier can improve while operational smoothness declines and jump intensity increases. We treat these findings as empirical and conceptually organizing rather than definitive or universal. This manuscript is an empirical + conceptual framing paper, not a complete operational morphology theory, not a universal classifier taxonomy, and not a benchmark competition paper.

## 1. Introduction

Severe class imbalance creates a practical deployment problem that is easy to understate: a model may rank examples well, yet still fail to make minority cases operationally accessible at realistic decision thresholds. In many systems, decisions are threshold-mediated rather than rank-only. The operational question is therefore not only “can the model rank positives above negatives?” but also “how does minority accessibility evolve as threshold policy changes?”

Our empirical program began from a conventional benchmark motivation, but repeatedly produced a different signal: ranking quality does not reliably determine threshold-level accessibility under severe imbalance. For example, in our MLP baseline analysis over `boundary`, `cam`, `compustat`, `oil`, and `satimage`, dataset-mean values show `auroc_mean=0.7068` and `average_precision_mean=0.2410`, while `recall@0.50=0.1346` and `recall@0.01=0.7867` (recovery `+0.6522`) [source: `reports/neural_mlp_allocation_geometry_summary.md`]. On some severe datasets in the same run, default-threshold recall collapses to near-zero while low-threshold recovery remains substantial.

This paper argues that threshold-mediated accessibility trajectories are operationally relevant analytical objects in their own right. We position reachability trajectories as a practical lens to study this behavior, and we organize recurring empirical patterns using provisional morphology language. We then extend this analysis to a fixed-architecture neural perturbation setting (`mlp_bce`, `mlp_oversampled`, `mlp_weighted`) to test whether accessibility morphology is sensitive to imbalance-pressure mechanisms.

Our central narrative is six-fold: (1) static ranking metrics can obscure accessibility behavior; (2) reachability trajectories reveal accessibility evolution; (3) recurring empirical morphology patterns appear; (4) fixed-architecture neural perturbations induce morphology transitions; (5) calibration can improve reliability while degrading accessibility smoothness; and (6) operational accessibility appears partially independent from both ranking and calibration quality.

### Scope delimitations

We explicitly delimit scope. This manuscript is:

- an empirical + conceptual framing paper,
- focused on severe imbalance and threshold-mediated deployment,
- based on constrained model and dataset slices.

This manuscript is not:

- a complete operational morphology theory,
- a universal classifier taxonomy,
- a formal topology paper,
- a benchmark leaderboard paper,
- or a production-ready objective proposal.

## 2. Operational Accessibility Under Severe Imbalance

Under severe imbalance, deployment systems are often constrained by review capacity, cost asymmetry, and threshold policy. In these contexts, the operationally relevant outcome is not just class ordering but accessibility persistence: whether minority cases remain available to action as thresholds move.

This distinction matters because threshold policies are not static in practice. Teams tighten or relax thresholds due to queue pressure, precision constraints, policy updates, or incident response. If model behavior is cliff-like, small policy changes can produce disproportionate accessibility shifts. If behavior is smoother, policy control is more stable.

We therefore use “operational accessibility” to refer to threshold-conditioned minority access, and treat ranking quality as necessary but insufficient for deployment interpretation. This framing is consistent with the project’s evolution documented in the study journal and with the framework document’s emphasis on threshold-mediated behavior [CITATION: severe imbalance deployment literature].

Potential objection: “Is this just threshold tuning?” Our answer is no in two senses. First, the object of study is the trajectory shape, not the best tuned point. Second, models with similar ranking quality can produce sharply different trajectory morphology, implying distinct operational control properties.

## 3. Reachability and Threshold Accessibility

We define reachability as:

\[
R(t) = P(\hat{p}(x) \ge t \mid y=1)
\]

where `t` is a decision threshold and `\hat{p}(x)` is the model score interpreted on `[0,1]`.

Intuitively, `R(t)` answers: “what fraction of minority instances remains operationally accessible at threshold `t`?”

This section addresses a key reviewer concern directly: “isn’t reachability just recall?” Recall at a fixed threshold is one point on `R(t)`. Reachability analysis studies the trajectory across thresholds, including:

- where accessibility loss concentrates,
- how fast accessibility decays,
- whether transition behavior is smooth or phase-like,
- and how stable policy control is under threshold perturbation.

In this project, threshold grid analyses (`0.50`, `0.25`, `0.10`, `0.05`, `0.01`) and interval derivatives reveal patterns that single-threshold recall cannot. In transition analysis artifacts, `mlp_bce` often concentrates reachability gains in narrow intervals (e.g., strong `0.50->0.25` jump on `boundary`; large `0.05->0.01` jump on `compustat`), while perturbed variants distribute gains more broadly [source: `results/geometry_transition_analysis/reachability_derivatives.csv`].

[Figure 1 about here]

**Figure 1 (working):** Mean reachability transition (`reports/geometry_transition_analysis/plots/reachability_transition_mean.png`).

[Figure 2 about here]

**Figure 2 (working):** Dataset-level reachability panels (`reports/geometry_transition_analysis/plots/reachability_transition_by_dataset.png`).

Operational implication: same-threshold recall values can conceal very different threshold sensitivity structures. We therefore treat reachability trajectory analysis as central, not auxiliary.

## 4. Experimental Framework

We use a constrained, reproducible empirical framework built around repeated splits, fixed threshold sweeps, and shared metric/reporting pathways.

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
- regime synthesis outputs.

This is intentionally not a broad architecture benchmark. No new deep-learning stack, no neural architecture zoo, and no hyperparameter sweep campaign are required for the claims developed here.

TODO: verify manuscript methods table with exact split-seed wording from runner configs before external circulation.

## 5. Accessibility Trajectory Observations

The first empirical layer is the ranking-accessibility divergence under threshold evolution.

From the MLP baseline report, dataset-mean ranking quality is moderate (`auroc_mean=0.7068`, `average_precision_mean=0.2410`), but default-threshold accessibility is low (`recall@0.50=0.1346`) with strong recovery under relaxation (`recall@0.01=0.7867`) [source: `reports/neural_mlp_allocation_geometry_summary.md`]. At dataset level, severe collapse cases appear (`boundary=0.0000`, `compustat=0.0000` at `0.50`) with large recovery under lower thresholds.

These observations are consistent with the journal’s earlier model-family findings: high AUROC/AP can coexist with near-zero minority accessibility at operational defaults.

This section is not a “which model wins” section. Its purpose is to establish that trajectory-aware interpretation is necessary.

[Table 1 about here]

**Table 1 (working):** Representative ranking/accessibility anchors (AUROC, AP, recall@0.50, recall@0.01, recovery) for selected models and datasets.

Potential objection: “are these isolated examples?” We mitigate that concern by repeated-split summaries and multi-dataset consistency in severe subsets, while still treating generalization as bounded.

## 6. Recurring Morphology Patterns (Provisional)

After trajectory evidence is established, recurring empirical patterns become interpretable shorthand. We use labels such as quantized, cliff-like, smooth, broad, and conservative to summarize recurring behavior; we do not present these as fixed ontological classes.

In baseline analyses, CART often aligns with quantized behavior (minimal threshold response), XGBoost is consistently cliff-like in current evidence, LightGBM appears conservative in the cited runs, and HDDT/bagged HDDT often align with broader accessibility patterns. These labels are empirical conveniences that compress multi-metric behavior into discussable forms.

[Figure 3 about here]

**Figure 3 (working):** Elasticity interval concentration heatmap (`reports/geometry_transition_analysis/plots/elasticity_interval_heatmap.png`) used here as morphology-pattern evidence rather than taxonomy proof.

Reviewer-facing caveat: regime labels are heuristic and threshold-grid dependent. We therefore treat them as provisional shorthand for recurring trajectory morphology patterns.

TODO(APPENDIX): include expanded regime tables and alternative threshold-grid sensitivity checks.

## 7. Neural Perturbation and Morphology Transition

This section provides a key inferential pivot: with architecture held fixed, imbalance-pressure perturbation changes accessibility morphology.

From the perturbation summary (dataset means):

- `mlp_bce`: `AUROC=0.7068`, `AP=0.2410`, `recall@0.50=0.1346`, `recall@0.01=0.7867`, `smoothness=0.4178`, `max_recall_jump=0.4061`, regime `cliff_allocator`.
- `mlp_oversampled`: `AUROC=0.8351`, `AP=0.2989`, `recall@0.50=0.4973`, `recall@0.01=0.7041`, `smoothness=0.5391`, `max_recall_jump=0.0678`, regime `smooth_allocator`.
- `mlp_weighted`: `AUROC=0.7732`, `AP=0.2658`, `recall@0.50=0.5789`, `recall@0.01=0.8506`, `smoothness=0.6080`, `max_recall_jump=0.1783`, regime `smooth_allocator`.

These shifts are consistent with optimization/imbalance-pressure sensitivity and support the claim that accessibility morphology is not architecture-only.

However, we avoid strong causal language. The evidence is controlled but still empirical: one architecture family, constrained perturbation mechanisms, bounded dataset slice.

[Figure 4 about here]

**Figure 4 (working):** Support vs persistence scatter (`reports/geometry_transition_analysis/plots/support_vs_persistence.png`).

Interpretive nuance from transition analysis: smoothness is not reducible to support breadth alone. `mlp_weighted` broadens support more strongly, but `mlp_oversampled` also smooths trajectories without equivalent support expansion. This appears consistent with threshold-morphology redistribution as a primary mechanism.

Potential objection: “does oversampling merely shift thresholds?” The interval-derivative evidence suggests more than pure shift: jump localization and elasticity concentration structure change materially.

## 8. Calibration vs Operational Accessibility

This section consolidates one of the strongest empirical findings: reliability improvement and accessibility smoothness can diverge.

Across MLP perturbation variants, calibration summaries indicate ECE/Brier improvements, while transition deltas show negative smoothness shifts and positive jump shifts on average [source: `results/geometry_transition_analysis/calibration_transition_model_means.csv`]. In regime persistence artifacts, `mlp_oversampled` and `mlp_weighted` can appear smooth in raw form but cliff-like after calibration transforms [source: `reports/neural_mlp_objective_perturbation/calibration_interaction/regime_persistence_table.csv`].

This does not imply calibration is undesirable. Calibration remains valuable for probability reliability. The empirical tension is that reliability correction and threshold-trajectory behavior are partially non-equivalent under severe imbalance.

[Figure 5 about here]

**Figure 5 (working):** Calibration geometry deltas (`reports/geometry_transition_analysis/plots/calibration_geometry_deltas.png`).

Operationally, this matters because teams may improve reliability metrics while unintentionally increasing threshold sensitivity and reducing accessibility controllability.

Potential objection: “is this just calibration pathology in one setup?” We treat this as a bounded observation requiring replication across calibrators and data regimes, not a universal anti-calibration claim.

## 9. Toward Operational Morphology

What does the current evidence support?

First, operational accessibility trajectories deserve direct analysis under severe imbalance. Second, recurring morphology patterns can summarize behavior if treated as empirical shorthand. Third, fixed-architecture perturbation results suggest optimization and imbalance-pressure mechanisms can materially alter morphology. Fourth, calibration-quality improvements can coexist with accessibility degradation.

Taken together, these findings appear consistent with a jointly emergent view: ranking, calibration, and accessibility are related but partially independent axes in threshold-mediated deployment.

We therefore present “operational morphology” as an emerging framing perspective, not a complete formal theory. The goal is to improve interpretability of deployment behavior, sharpen hypothesis generation, and structure follow-up studies without overclaiming.

TODO: add one synthesis figure/table mapping each central claim to direct evidence artifact and confidence level.

## 10. Limitations

This draft has clear limitations.

1. **Dataset scope:** evidence comes from a constrained severe-imbalance legacy slice; external validity is unproven.
2. **Threshold grid dependence:** analyses use a fixed threshold set; finer grids may alter localization details.
3. **Taxonomy provisionality:** regime labels are heuristic summaries, not definitive classes.
4. **Neural scope:** only sklearn-style MLP baseline and perturbations are included.
5. **Mechanism uncertainty:** bagged HDDT broad behavior is empirically visible but mechanistically unresolved.
6. **Calibration interaction uncertainty:** observed reliability-smoothness divergence may vary by calibrator/data scale.

Accordingly, we do not claim universal behavior, complete morphology characterization, or definitive causal decomposition.

## 11. Future Work

We prioritize bounded extensions aligned to current evidence strength.

1. **Bagged HDDT mechanism disambiguation:** separate effects of Hellinger splitting, bagging stabilization, and feature/subsample structure.
2. **Calibration-morphology decomposition:** characterize when reliability correction re-steepens accessibility and when it does not.
3. **Reachability-aware objective exploration (cautious):** investigate whether trajectory-aware training pressure can improve operational controllability without overfitting threshold grids.
4. **Cross-domain replication:** test whether the trajectory non-equivalence pattern holds in additional imbalance domains.

These are hypothesis-generating directions, not implied validated solutions.

## 12. Conclusion

This manuscript draft argues that under severe class imbalance, threshold-mediated operational accessibility should be analyzed as a first-class empirical object. Current evidence suggests that static ranking and calibration summaries can obscure trajectory behavior that is operationally consequential. Reachability trajectories, elasticity localization, and occupancy persistence provide a practical lens for this analysis. Within a fixed MLP architecture, objective/sampling perturbations can move behavior from cliff-like to smoother patterns, supporting the view that accessibility morphology is sensitive to imbalance-pressure mechanisms. Calibration findings further suggest a meaningful tension between reliability improvements and operational smoothness.

We present these results as an empirical + conceptual framing contribution. We treat morphology taxonomy as provisional, preserve explicit non-claims, and frame this draft as a foundation for deeper, more targeted follow-up rather than a complete theory or benchmark endpoint.

---

## Figure and Citation Placeholders Checklist

- [Figure 1 about here] Reachability mean transition
- [Figure 2 about here] Reachability by dataset panels
- [Figure 3 about here] Elasticity interval heatmap
- [Figure 4 about here] Support vs persistence scatter
- [Figure 5 about here] Calibration geometry deltas
- [Table 1 about here] Ranking vs accessibility anchors

- [CITATION: severe imbalance deployment literature]
- [CITATION: HDDT original paper]
- [CITATION: calibration literature]
- [CITATION: threshold decision policy / risk management literature]
- [CITATION: class imbalance evaluation literature]
