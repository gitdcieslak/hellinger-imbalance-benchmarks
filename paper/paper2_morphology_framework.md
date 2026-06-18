# Paper 2 Narrative Assembly: Accessibility Morphology Framework

## Executive Decision

The Accessibility Morphology Framework is sufficiently supported to begin manuscript drafting. The core paper claim survives the completed experiments:

> Accessibility level is largely explained by position along a shared morphology manifold, while accessibility dynamics emerge from nonlinear geometry on that manifold.

The evidence is strongest for synthetic severe-skew MLP-family experiments and morphology-field analyses. The remaining gaps are not reasons to delay drafting, but they should shape the manuscript scope: classical allocator coverage is incomplete in the current manifold synthesis, topology currently acts more as a mechanism candidate than as a predictive descriptor, and the kernel field is diagnostic rather than causal.

## 1. Motivation

Extreme class imbalance creates an operational problem that standard performance summaries do not resolve. Under severe skew, two models can have similar AUROC, average precision, or calibration error while exposing very different threshold behavior for the minority class. The question facing a practitioner is not only whether positives are ranked ahead of negatives on average, but whether there exists a usable threshold region where minority examples remain accessible without catastrophic loss.

Ranking metrics summarize ordering. Calibration metrics summarize probability fidelity. Neither directly describes the shape of minority accessibility as the threshold moves. In a deployment setting, threshold selection is an operational act: a model must provide a threshold path over which minority support persists, changes smoothly enough to control, and avoids abrupt cliffs. This is especially important under extreme skew because small score reallocations can make the minority class appear either recoverable or operationally invisible.

This motivates accessibility as a distinct operational property. Accessibility asks how a classifier allocates minority-positive score mass across the decision-threshold landscape. It is not reducible to AUROC, AP, or calibration because it concerns threshold usability rather than global ranking or probability accuracy.

The framework assembled here treats accessibility as a morphology problem. Instead of asking only whether a learner ranks well, we ask what morphology state its minority score distribution occupies and what threshold dynamics that state induces.

## 2. Accessibility Morphology

Accessibility morphology is the low-dimensional shape of the minority-positive score allocation induced by a learner. The completed experiments focus on two core state variables.

Accessibility Breadth measures how broadly minority-positive mass is distributed across accessible threshold regions. Operationally, breadth captures whether a model spreads minority accessibility across a wider score range or compresses it into a narrow band.

Accessibility Elevation measures how much minority-positive mass is lifted into the high-score, high-accessibility region. Operationally, elevation captures whether a model creates a strong accessible peak for minority examples.

Together, breadth and elevation define the working morphology space. The current synthesis uses `reports/topology/accessibility_morphology_space.png` as the main morphology map and `reports/topology/accessibility_operational_regime_map.png` as the operational interpretation map.

The key question becomes: what morphology state is an allocator occupying?

This reframes allocator behavior as a state-space problem. A broad allocator can distribute minority accessibility across a larger threshold range. A concentrated allocator can lift minority examples into a high-accessibility region but may narrow usable threshold flexibility. A quantized allocator can create discrete or brittle threshold behavior. The same AUROC or AP value may correspond to different morphology states and therefore different operational threshold behavior.

Current allocator placement is strongest for the observed MLP-family variants and dropout/weighting/oversampling settings. The synthesis table records named classical allocators such as HDDT, Bagged HDDT, CART, Random Forest, XGBoost, and LightGBM as important paper-facing targets, but the current pooled manifold file does not yet include all of them as observed rows. That is a manuscript-scope caveat rather than a rejection of the framework.

## 3. Accessibility Level

Accessibility level is the amount of minority accessibility preserved over the threshold path. The primary empirical proxy is minority survival AUC, interpreted as accessibility persistence: how long minority support survives as the decision threshold tightens.

The central level result is strong:

| Target | Linear Breadth/Elevation CV R2 | PC1 Only CV R2 | PC1/PC2 CV R2 | Kernel CV R2 |
| --- | --- | --- | --- | --- |
| Minority survival AUC | 0.8580 | 0.8008 | 0.8580 | 0.9485 |

These results support the claim that accessibility level behaves approximately as a state variable. Breadth and elevation alone predict survival AUC well under grouped cross-validation by experiment family. PC1 alone also predicts survival AUC well, which indicates that the morphology manifold has a dominant axis of accessibility level.

The PCA diagnostics reinforce this interpretation:

| Diagnostic | Value |
| --- | --- |
| PC1 explained variance ratio | 0.9301 |
| PC2 explained variance ratio | 0.0699 |
| Breadth/elevation correlation | -0.8602 |
| PC1 loading on breadth | 0.7071 |
| PC1 loading on elevation | -0.7071 |

The morphology space is therefore approximately one-dimensional in its dominant variation, but not one-dimensional enough to explain all accessibility behavior. The best scalar coordinate for level, global arc position, achieved survival AUC CV R2 of 0.8531. This is nearly identical to linear breadth/elevation, so the level component can be summarized compactly.

The paper claim for level is ready to use: accessibility persistence is largely determined by morphology position, especially the breadth/elevation state and its dominant manifold coordinate.

## 4. Accessibility Dynamics

Accessibility dynamics describe how accessibility changes along the threshold path. The key quantities are persistence and cliffiness.

Persistence is the survival of minority support across thresholds and is measured by minority survival AUC.

Cliffiness is the abruptness of accessibility loss and is measured by the shape of the minority survival curve, including sharp drops and concentrated threshold failures.

The completed experiments show that dynamics do not behave like level. Linear morphology is good for survival AUC but fails for cliffiness:

| Target | Linear Breadth/Elevation CV R2 | PC1 Only CV R2 | PC1/PC2 CV R2 |
| --- | --- | --- | --- |
| Minority survival cliffiness | -0.1839 | -0.1717 | -0.1839 |

This failure is important. It shows that cliffiness is not just a lower-quality version of survival AUC and not simply a coordinate along the same scalar manifold axis. Adding PC2 does not rescue linear cliffiness prediction; the PC1/PC2 model remains negative under grouped CV.

Topology-only and topology-augmented models also fail to explain cliffiness in their current form. In the topology-augmented morphology report, topology-only cliffiness CV R2 was -0.1398 and morphology plus topology was -0.1878, only a +0.0016 improvement over morphology alone. This argues against a simple descriptor-level topology explanation for cliffiness. Topology may still be mechanistic, but the current component and fragmentation descriptors are not sufficient as a direct predictive representation.

Scalar manifold coordinates also fail for joint accessibility behavior. The best scalar coordinate, accessibility arc, predicts survival AUC well but predicts cliffiness poorly:

| Coordinate | Survival AUC CV R2 | Cliffiness CV R2 | Mean Joint Score |
| --- | --- | --- | --- |
| Accessibility coordinate arc | 0.8531 | -0.1521 | 0.3505 |

The conclusion is that accessibility level and accessibility dynamics are different objects. Level behaves like a state variable. Dynamics behave like local geometry on the state space.

## 5. Nonlinear Morphology Geometry

Kernel morphology resolves the cliffiness failure. The RBF kernel field over standardized breadth/elevation transforms morphology from a linear coordinate system into a nonlinear surface. This nonlinear surface strongly predicts both survival AUC and cliffiness under grouped CV.

| Target | Linear Morphology CV R2 | Kernel Morphology CV R2 | Delta |
| --- | --- | --- | --- |
| Minority survival AUC | 0.8580 | 0.9485 | +0.0905 |
| Minority survival cliffiness | -0.1839 | 0.9063 | +1.0902 |

The kernel result is the strongest evidence for the paper's central theoretical distinction. Survival AUC improves modestly because level was already mostly captured by position. Cliffiness improves dramatically because the dynamics are nonlinear in morphology space.

The kernel field diagnostics show that cliffiness has coherent local structure over the manifold. The cliffiness field has high gradient magnitude and high local curvature, consistent with abrupt threshold-transition regions. The generated figures `reports/topology/kernel_morphology_survival_field.png`, `reports/topology/kernel_morphology_cliffiness_field.png`, and `reports/topology/accessibility_kernel_cliffiness_field.png` should be treated as paper-facing candidates.

Residuals after kernel smoothing are not dominated by sample density. The density-versus-residual-variance correlation was approximately -0.0046 for cliffiness, suggesting that the kernel result is not merely a data-density artifact. The largest cliffiness residual family was density threshold, which should be reported as the main family-specific correction candidate.

The paper claim for dynamics is ready to use with a careful qualifier: accessibility dynamics emerge from nonlinear geometry on the morphology manifold, as diagnosed by kernel smoothing, but causal validation remains future work.

## 6. Family Trajectories

The completed experiments indicate that different experiment families occupy a shared morphology space while tracing distinct arcs through it. This is the bridge between individual sweeps and a unified framework.

| Family | Rows | PC1 Range | PC2 Spread | Mean Survival AUC | Mean Cliffiness | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| density_threshold | 1883 | -1.5422 to 3.1586 | 0.3531 | 0.8264 | 0.4834 | shared level manifold, dynamics correction candidate |
| density_separability | 880 | -1.5422 to 3.1481 | 0.3744 | 0.7934 | 0.4460 | follows pooled morphology field |
| fragmentation | 242 | -1.5422 to -0.1738 | 0.0438 | 0.9978 | 0.1348 | compact high-elevation, low-cliffiness arc |
| allocation_trajectory | 350 | -1.3558 to 3.2370 | 0.6709 | 0.5477 | 0.5967 | wide PC2 spread, family-specific shape |
| weighted_dropout_dense | 880 | -1.2085 to 0.7775 | 0.0885 | 0.9474 | 0.8300 | high-cliffiness dropout arc |
| mlp_objectives_topology | 70 | -0.4037 to 3.1824 | 0.7170 | 0.4519 | 0.5221 | wide PC2 spread, objective-dependent shape |

The trajectory figure `reports/topology/accessibility_family_trajectories.png` should be used to explain this result visually. Families do not collapse to a single curve. Instead, they move through a common breadth/elevation state space along different paths.

The family-level synthesis is:

| Family | Shared Manifold? | Shared Dynamics? | Notes |
| --- | --- | --- | --- |
| density_threshold | yes | partial | large kernel residual |
| density_separability | yes | yes | follows pooled morphology field |
| allocation_trajectory | partial | yes | wide PC2 spread |
| weighted_dropout_dense | yes | yes | high-cliffiness dropout arc |
| mlp_objectives_topology | partial | yes | wide PC2 spread |
| fragmentation | yes | yes | compact low-cliffiness arc |

This supports the common-state-space claim while preserving family-specific nuance. The right theoretical framing is not that every allocator follows the same trajectory. It is that allocator behavior can be located in a common morphology space, and each training or data perturbation traces a characteristic trajectory through that space.

## 7. Operational Regimes

The morphology manifold can be interpreted as an operational regime map. The regimes are not arbitrary clusters; they correspond to threshold-selection behavior.

Broad Allocators distribute minority accessibility across a wider threshold region. They may offer more threshold flexibility but can have lower elevation if minority mass is spread rather than lifted.

Quantized Allocators concentrate accessibility into discrete score states. They may appear acceptable under ranking metrics but produce brittle operational behavior because small threshold changes can trigger large accessibility changes.

Concentrated Allocators lift minority mass into a high-accessibility region. They can produce high survival AUC but may also create high cliffiness if the accessible region is narrow or sharply bounded.

The operational regime figure `reports/topology/accessibility_operational_regime_map.png` places observed learners and perturbations into this space. Current observations include MLP BCE variants, weighted BCE, oversampled BCE, and dropout variants. In the current synthesis table, the named classical allocators HDDT, Bagged HDDT, CART, Random Forest, XGBoost, and LightGBM should be treated as required manuscript comparators, but not all are present in the latest pooled manifold inputs. The manuscript should either add those rows before submission or explicitly scope the morphology-manifold evidence to the available neural and synthetic allocator families.

The key operational implication is that threshold behavior can be discussed in terms of where a model sits in morphology space. A model's accessibility risk is not just whether its score is high or calibrated; it is whether its morphology regime creates usable persistence and manageable dynamics.

## 8. Accessibility Morphology Framework

The proposed framework is:

```text
Accessibility Level
    -> Morphology State
    -> Accessibility Dynamics
```

Accessibility Level is the amount of minority accessibility preserved over threshold movement. It is measured primarily by minority survival AUC and behaves approximately as a morphology-state variable.

Morphology State is the position of the allocator in breadth/elevation space. It captures the operational shape of minority score allocation rather than only ranking or probability accuracy.

Accessibility Dynamics are the local threshold behaviors induced by morphology state, especially abrupt changes in accessibility. Dynamics are measured by cliffiness and are best explained by nonlinear geometry on the morphology manifold.

Accessibility Breadth is the operational spread of minority accessibility across thresholds.

Accessibility Elevation is the lifting of minority mass into high-accessibility score regions.

Accessibility Persistence is the survival of minority support as thresholds tighten.

Accessibility Regime is the qualitative operating zone of an allocator, such as broad, quantized, concentrated, or mixed morphology.

Accessibility Trajectory is the path traced through morphology space as a data property, model objective, training intervention, or threshold perturbation changes.

The framework explains why the same intervention can improve one accessibility property while harming another. For example, the dense dropout sweep showed survival AUC increasing near-monotonically with dropout in 4/4 skew regimes, while cliffiness increased with dropout in most regimes. This is an accessibility tradeoff: dropout moves the allocator through morphology space toward higher elevation and survival, but also toward sharper dynamics.

The weighted BCE x oversampling grid reinforces the same idea. The accessibility-optimal cell differed from AUROC/AP optima: survival and low cliffiness were best at high weighting plus high oversampling, while AUROC/AP preferred lower intervention. Weighting and oversampling moved morphology in different directions, supporting the claim that accessibility is an operational property not reducible to conventional ranking metrics.

Topology studies add mechanism context. Fragmentation weakly reduced survival and increased density-adjusted cliffiness, with the largest cliffiness jump from 20 to 50 islands. Density and separability sweeps place data geometry into the morphology framework. However, topology descriptors did not directly predict cliffiness once used as simple features, so topology is currently better framed as a possible generator of morphology transitions rather than the final explanatory coordinate system.

## 9. Threats to Validity

Synthetic datasets are the largest external validity threat. The framework is supported across several synthetic families and severe-skew perturbations, but real-world datasets may introduce score-allocation structures not represented here.

Family labels are experiment-design labels. Grouped CV by family is intentionally strict, but family imbalance and protocol differences can affect residuals and model comparison.

Kernel smoothing assumptions matter. The kernel field is strong diagnostic evidence for nonlinear morphology geometry, but the result depends on bandwidth, kernel choice, and the observed support of the morphology space.

Classical allocator coverage is incomplete in the current manifold synthesis. HDDT, Bagged HDDT, CART, Random Forest, XGBoost, and LightGBM should be placed into the final state space before submission if the manuscript claims allocator universality across classical and neural families.

Causality is not established. The results show that morphology state predicts accessibility level and that nonlinear morphology geometry predicts dynamics. They do not yet prove that changing morphology causally changes accessibility dynamics in isolation.

Topology remains under-specified as a mechanism. Fragmentation, density, and component structure are plausible generators of morphology, but current topology features do not directly explain cliffiness.

## 10. Future Work

Perturbation studies should directly manipulate breadth and elevation while holding other factors fixed. This is the cleanest route from diagnostic morphology to causal morphology.

Topology should be studied as a mechanism rather than only as a descriptor. The next step is to model how density, separability, and fragmentation move allocators through morphology space.

Causal accessibility transitions should be formalized. The current framework suggests transitions between broad, concentrated, and quantized regimes, but these transitions need controlled interventions.

Accessibility control and optimization should become a design objective. If morphology state predicts operational accessibility, training procedures can target desired morphology regimes directly.

The framework should be extended to deeper neural allocators. Current neural results use MLP variants and feature-dropout approximations. Deeper architectures may create richer morphology trajectories.

Classical allocator placement should be completed. HDDT, Bagged HDDT, CART, Random Forest, XGBoost, and LightGBM should be run through the same morphology-state pipeline and included in the final operational regime map.

Kernel robustness should be tested. The RBF result should be checked against bandwidth sensitivity, Gaussian process regression, spline smoothers, and nested grouped CV.

## Gap Audit

| Claim | Supporting Evidence | Strength | Missing Evidence |
| ----- | ------------------- | -------- | ---------------- |
| Accessibility is distinct from ranking and calibration | Weighted oversampling grid shows accessibility optimum differs from AUROC/AP optima; dropout sweep shows accessibility-optimal dropout differs from AUROC/AP optima | Strongly Supported | More real-data demonstrations would strengthen external validity |
| Accessibility morphology can be represented by breadth and elevation | Pooled morphology space across 4305 rows, 6 families, 23 models; clear operational interpretation | Strongly Supported | Complete classical allocator placement in the same table |
| Morphology space is approximately one-dimensional in dominant variation | PC1 explains 0.9301 of standardized breadth/elevation variance | Strongly Supported | Robustness across external datasets |
| Accessibility level is explained by morphology position | Survival AUC CV R2 is 0.8580 for linear breadth/elevation and 0.8008 for PC1 only | Strongly Supported | Causal perturbation of morphology position |
| Survival AUC behaves approximately as a state variable | Global arc position predicts survival AUC with CV R2 0.8531 | Strongly Supported | Direct validation on held-out dataset families beyond synthetic sweeps |
| Cliffiness is not a linear morphology coordinate | Linear breadth/elevation CV R2 is -0.1839; PC1/PC2 CV R2 is -0.1839; scalar coordinate cliffiness CV R2 is -0.1521 | Strongly Supported | None for the negative linear result; still needs causal interpretation |
| Accessibility dynamics emerge from nonlinear morphology geometry | Kernel morphology predicts cliffiness with CV R2 0.9063 and improves over linear by +1.0902 | Strongly Supported | Kernel bandwidth sensitivity and alternative nonlinear smoothers |
| A universal scalar coordinate explains both level and dynamics | Best scalar coordinate predicts survival but not cliffiness | Weakly Supported | Need nonlinear or multi-coordinate state representation; scalar claim should not be made |
| Experiment families share a common morphology state space | Density, separability, fragmentation, dropout, trajectory, and objective studies all normalize into breadth/elevation space | Moderately Supported | Classical allocator families and external datasets need inclusion |
| Experiment families follow distinct trajectories | Family arcs show different PC ranges and PC2 spreads; trajectories visible in generated arc plots | Strongly Supported | More controlled trajectory definitions for each perturbation |
| Density threshold needs family-specific dynamics correction | Kernel residual analysis identifies density threshold as largest cliffiness residual family | Moderately Supported | Inspect local density-threshold subregions and rerun robustness checks |
| Allocation trajectory and MLP objective studies deviate in manifold shape | PC2 spread is largest for mlp_objectives_topology and allocation_trajectory | Moderately Supported | Determine whether deviations are meaningful or due to protocol/row-count differences |
| Topology directly explains cliffiness | Topology-only and topology-augmented models do not improve cliffiness meaningfully | Weakly Supported | Topology should be reframed as mechanism candidate and tested with richer features |
| Fragmentation affects accessibility dynamics | Fragmentation increases density-adjusted cliffiness and largest cliffiness jump occurs from 20 to 50 islands | Moderately Supported | Stronger causal controls and more geometry regimes |
| Dropout creates an accessibility tradeoff | Survival AUC increases near-monotonically with dropout in 4/4 skews; cliffiness also tends to increase | Strongly Supported | Confirm with true neural dropout/deeper networks |
| Weighting and oversampling are morphology controls | Weighting and oversampling move morphology differently; accessibility optimum differs from AUROC/AP optimum | Strongly Supported | Extend to more learners and real datasets |
| Classical allocators can all be placed into the common state space | Framework and pipeline support placement; current synthesis retains named allocators | Speculative | Run HDDT, Bagged HDDT, CART, Random Forest, XGBoost, and LightGBM through the same pooled morphology pipeline |
| The framework is publication-ready | Core level/dynamics distinction is strongly supported and coherent across multiple experiment families | Moderately Supported | Add or scope classical allocator evidence; add kernel robustness checks if time permits |

## Reader Answers

What is accessibility morphology? It is the operational shape of minority-positive score allocation across thresholds, summarized by breadth and elevation.

What is the morphology manifold? It is the shared breadth/elevation state space in which different data regimes, objectives, and training interventions trace trajectories.

Why does level behave differently from dynamics? Level is mostly determined by position on the manifold, while dynamics depend on nonlinear local geometry that creates cliffs and sharp transitions.

What evidence supports the framework? Survival AUC is strongly predicted by linear morphology, cliffiness is not, kernel morphology strongly predicts cliffiness, and multiple experiment families occupy a common state space with distinct trajectories.

What evidence is still missing? Complete classical allocator placement, real-data validation, causal perturbation studies, and robustness checks for kernel smoothing.

## Manuscript Recommendation

Begin drafting Paper 2 around the Accessibility Morphology Framework now. Do not wait for another dozen sweeps. The central contribution is already visible: accessibility is not merely ranking performance under another name; it is an operational morphology of threshold usability. The current evidence supports a manuscript whose main claim is the level/dynamics split, with morphology position explaining accessibility level and nonlinear morphology geometry explaining accessibility dynamics.

Before submission, prioritize three bounded additions if time permits: place the named classical allocators into the same morphology map, run a kernel bandwidth sensitivity check, and add one real or semi-real dataset validation. These would strengthen generality but are not required to justify the theoretical framework as the core paper narrative.
