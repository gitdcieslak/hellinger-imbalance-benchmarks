# Accessibility Regimes in Rare-Event Learning

## Abstract

Rare-event classifiers are usually evaluated by ranking, calibration, or aggregate threshold metrics, yet deployment often depends on a different question: whether rare positives remain accessible over a usable threshold region. Building on the accessibility framework from Paper 1, this manuscript studies threshold usability as morphology. We represent minority score allocation with breadth, which measures how widely minority-positive score mass is distributed, and elevation, which measures how much of that mass reaches high-score regions. Across neural objectives, density and separability sweeps, fragmentation sweeps, dropout interventions, training trajectories, and a 480-run classical allocator sweep, accessibility persistence is largely explained by morphology position. Cliffiness, the abruptness of minority survival loss, is not. Linear morphology, scalar manifold coordinates, topology-only descriptors, simple boundary distance, and frontier distance all fail to explain cliffiness under grouped validation, while nonlinear morphology models remain predictive. We therefore construct a morphology atlas and consolidate allocator behavior into macro-regimes. Native HDBSCAN/UMAP replication preserves the main regime result: four macro-regimes remain, cliffiness-bucket accuracy is 0.7152, the between/within cliffiness variance ratio is 7.9470, and the broad high-cliff basin and quantized floor survive. Regime information contributes beyond allocator family and skew in regression checks, and transition analysis shows that interventions induce structured movement through regime space. Adding start/end regime to continuous morphology deltas improves held-out prediction of cliffiness change from CV R2 0.1138 to 0.2776. The central claim is not that cliffiness follows one global equation or frontier distance. Accessibility behavior is regime-structured: cliffiness is better understood as regime plus local morphology.

## Review Navigation

### Figures

| Figure | Title | Section | Source |
| --- | --- | --- | --- |
| 1 | Shared Accessibility Morphology Space | 2. Background | `../reports/topology/accessibility_morphology_space_with_classical.png` |
| 2 | Native UMAP Morphology Atlas | 3. Morphology Atlas Construction | `../reports/topology/morphology_umap_space.png` |
| 3 | Regime Consolidation Dendrogram | 4.2 Regime Consolidation | `../reports/topology/cluster_dendrogram.png` |
| 4 | Regime Compression Curve | 4.2 Regime Consolidation | `../reports/topology/regime_compression_curve.png` |
| 5 | Nonlinear Model Comparison | 4.4 Equation Discovery | `../reports/topology/kernel_robustness_model_comparison.png` |
| 6 | Frontier Geometry Falsification | 4.5 Frontier Geometry | `../reports/topology/frontier_model_comparison.png` |
| 7 | Transition Paths in Morphology Space | 4.6 Regime Transition Analysis | `../reports/topology/regime_transition_paths_morphology_space.png` |
| 8 | Regime Stability by Intervention | 4.6 Regime Transition Analysis | `../reports/topology/regime_stability_by_intervention.png` |
| 9 | Cliffiness Change by Intervention | 4.6 Regime Transition Analysis | `../reports/topology/regime_transition_delta_cliffiness.png` |

### Tables

| Table | Title | Section |
| --- | --- | --- |
| 1 | Core Definitions | 2. Background |
| 2 | Competing Explanation Results | 4.4-4.5 |
| 3 | Regime Robustness Summary | 4.1-4.2 |
| 4 | Confounding Check | 4.3 |
| 5 | Regime Transition Summary | 4.6 |

## 1. Introduction

Rare-event learning is often framed as a prediction problem under skew. A model should rank rare positives above negatives, maintain calibrated probabilities, or optimize a cost-sensitive objective. Those goals are important, but they do not fully describe deployment. In many applications, the practitioner must choose a threshold. The relevant question is then whether rare positives remain accessible over a stable threshold region.

Ranking and accessibility can diverge. A classifier may have acceptable AUROC while placing minority examples into a narrow or quantized score band. Such a model can look useful by ranking metrics and still provide little threshold flexibility. Similarly, an intervention can improve access over part of the threshold path while creating sharp drops in minority support. We call this threshold-path behavior accessibility.

This paper asks whether accessibility behavior can be organized geometrically. We represent minority-positive score allocation by morphology coordinates, especially breadth and elevation. Breadth describes the spread of minority-positive score mass. Elevation describes how much mass reaches high-score regions. These coordinates define a morphology space. The main question is whether allocator behavior inside this space is continuous and global, or whether it is organized into regimes.

The evidence favors regimes. Accessibility persistence behaves largely like a position variable in morphology space. Cliffiness does not. Linear morphology, scalar coordinates, topology-only descriptors, boundary distance, and frontier distance fail to explain cliffiness. Nonlinear models succeed, but equation discovery and frontier analysis show that the nonlinear signal is not reducible to one global law. A morphology atlas reveals stable macro-regimes, and transition analysis shows that interventions move allocators through those regimes in structured ways.

The working thesis is:

```text
Accessibility persistence ~= morphology position
Accessibility cliffiness ~= regime + local morphology
```

This manuscript is a review-preparation draft. It includes embedded figures, tables, captions, caveats, and result summaries so that the argument can be evaluated without opening the intermediate reports.

## 2. Background: Accessibility Morphology

The accessibility framework treats a classifier as an allocator of minority-positive examples across score space. As a threshold moves from permissive to strict, the fraction of minority positives above threshold forms a minority survival curve. The survival curve describes threshold-mediated access to the rare class.

**Table 1. Core definitions.**

| Term | Definition | Operational Role |
| --- | --- | --- |
| Breadth | Spread of minority-positive score mass across accessible score regions. | Describes whether positives occupy a narrow score band or a wider threshold range. |
| Elevation | Degree to which minority-positive score mass reaches high-score regions. | Describes whether positives survive strict thresholds. |
| Survival AUC | Area under the minority survival curve over threshold movement. | Primary measure of accessibility persistence. |
| Cliffiness | Abruptness of minority survival loss along the threshold path. | Measures threshold brittleness and sharp drops in access. |
| Persistence | General concept of minority support remaining available as thresholds tighten. | Accessibility-level behavior. |
| Regime | Coherent morphology region with characteristic accessibility behavior. | Explanatory compression of allocator states. |
| Transition | Ordered movement from one regime or morphology state to another under a sweep/intervention. | Describes accessibility dynamics across training or data perturbations. |

The broader morphology dataset combines training trajectories, weighted dropout sweeps, density threshold sweeps, density/separability factorial sweeps, fragmentation sweeps, MLP objective topology experiments, and a classical allocator morphology sweep. The classical sweep contains 480 successful runs over CART, HDDT, Bagged HDDT, Random Forest, XGBoost, and LightGBM. All six classical allocator families lie inside the earlier neural/synthetic morphology support, so the morphology space is not only a neural-model artifact.

![Figure 1. Shared accessibility morphology space.](../reports/topology/accessibility_morphology_space_with_classical.png)

**Figure 1. Shared accessibility morphology space.**
Pooled breadth/elevation morphology space with classical allocators included. The figure supports the claim that classical tree and ensemble allocators occupy the same broad accessibility morphology support as prior neural and synthetic experiments. Source: `../reports/topology/accessibility_morphology_space_with_classical.png`. Section: Background.

## 3. Morphology Atlas Construction

The morphology atlas tests whether allocator behavior forms recognizable regions. The primary atlas dataset is the 480-row classical allocator morphology sweep. Each run contributes breadth, elevation, survival AUC, cliffiness, and score-allocation summaries. Frontier-derived features are also attached, including vertical slack, normalized position between envelopes, local frontier width, nearest frontier distance, and frontier curvature.

Features are robust-scaled. PCA retaining 95 percent variance is used for clustering diagnostics. The initial atlas used PCA(2) visualization and DBSCAN fallback labels because UMAP and HDBSCAN were not yet installed. The review manuscript treats the later native replication as the robustness reference: UMAP uses `n_neighbors=15`, `min_dist=0.1`, and `random_state=42`; HDBSCAN uses `min_cluster_size=15` and `min_samples=5`.

Native HDBSCAN and DBSCAN fallback do not produce identical point labels. Their ARI is 0.1705 and NMI is 0.5147. This is not hidden. The robust claim is macro-regime preservation, not point-level cluster identity. Native HDBSCAN changes assignments substantially but preserves the broad high-cliff basin, the quantized floor, and the overall macro-regime structure after consolidation.

![Figure 2. Native UMAP morphology atlas.](../reports/topology/morphology_umap_space.png)

**Figure 2. Native UMAP morphology atlas.**
Native UMAP embedding of allocator morphology features, colored by allocator family, cliffiness, and macro-regime assignment. The figure shows that allocator behavior is not a single undifferentiated cloud; broad allocator regions and low-breadth quantized regions remain visible. Source: `../reports/topology/morphology_umap_space.png`. Section: Morphology Atlas Construction.

## 4. Results

### 4.1 Morphology Atlas

Allocator morphologies form non-random regions. In the fallback atlas, cluster ID predicted cliffiness buckets with 0.7710 cross-validated accuracy against a 0.3333 baseline, and the between/within cliffiness variance ratio was approximately 4.9351. Native HDBSCAN/UMAP preserved the macro-regime result: the consolidated native atlas selected four macro-regimes, reached cliffiness-bucket accuracy 0.7152, and produced a between/within cliffiness variance ratio of 7.9470.

The atlas also clarifies why one coordinate is insufficient. Broadness alone is not good or bad. Broad low-elevation states can be high-cliff and low-persistence. Low-breadth states can represent quantized score floors. Elevated broad states can improve persistence while retaining some cliffiness. The relevant object is therefore a morphology region, not an isolated coordinate.

**Table 3. Regime robustness summary.**

| Atlas | Cluster Count | Macro-Regime Count | Cliffiness Bucket Accuracy | Between/Within Variance Ratio | Main Conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| Fallback DBSCAN/PCA atlas | 14 | 4 | 0.7328 for regimes; 0.7710 for clusters | 4.9351 at cluster level; 0.6861 variance explained by 4 regimes | Discovery atlas supports discrete/semi-discrete regimes. |
| Native HDBSCAN/UMAP atlas | 9 at `min_cluster_size=15` | 4 | 0.7152 for regimes | 7.9470 | Macro-regime structure replicates; point labels do not exactly replicate. |

### 4.2 Regime Consolidation

The fallback atlas discovered 14 clusters, which is too many for an operational theory. Cluster centroids were constructed from mean breadth, mean elevation, mean cliffiness, mean survival AUC, mean vertical slack, and mean frontier position. Agglomerative clustering over centroids was evaluated for k from 2 to 8. Four regimes provided the best operational compression: they preserved most cluster-level cliffiness information while remaining interpretable.

The working regime vocabulary is:

1. **Quantized floor:** low breadth, low elevation, low survival, often low cliffiness because positives sit near a score floor.
2. **Broad-flat high-cliff basin:** broad but low-elevation morphology with high cliffiness and low survival.
3. **Elevated broad plateau:** broad morphology with higher elevation and improved survival.
4. **Mixed transition:** intermediate or elevated states connecting the floor and broad regimes.

These names are provisional but useful. The scientific object is not the numeric label; it is the morphology state and its associated threshold behavior.

![Figure 3. Regime consolidation dendrogram.](../reports/topology/cluster_dendrogram.png)

**Figure 3. Regime consolidation dendrogram.**
Hierarchical clustering of atlas cluster centroids. The dendrogram motivates compressing many discovered clusters into a smaller operational vocabulary. Source: `../reports/topology/cluster_dendrogram.png`. Section: Regime Consolidation.

![Figure 4. Regime compression curve.](../reports/topology/regime_compression_curve.png)

**Figure 4. Regime compression curve.**
Compression curve showing how much cliffiness-bucket predictability remains as cluster labels are compressed into fewer macro-regimes. The four-regime representation retains most of the cluster-level signal. Source: `../reports/topology/regime_compression_curve.png`. Section: Regime Consolidation.

### 4.3 Confounding Analysis

Regimes are associated with allocator families, which is partly expected: different allocators induce different score morphologies. The key question is whether regime information adds anything beyond family and skew.

**Table 4. Confounding check.**

| Feature Set | Cliffiness Bucket CV Accuracy | Cliffiness CV R2 | Gain vs Family + Skew Accuracy | Gain vs Family + Skew R2 |
| --- | ---: | ---: | ---: | ---: |
| Family + skew | 0.8015 | 0.6361 | 0.0000 | 0.0000 |
| Family + skew + regime | 0.8117 | 0.7317 | 0.0102 | 0.0956 |
| Family + skew + coordinates | 0.8142 | 0.6354 | 0.0127 | -0.0007 |
| Family + skew + regime + coordinates | 0.8346 | 0.7772 | 0.0331 | 0.1411 |

The result is mixed but useful. Regime ID adds substantial held-out R2 beyond family and skew, but bucket-accuracy gains are modest. This supports the manuscript framing: regimes are explanatory compression, not necessarily optimal classifiers.

### 4.4 Equation Discovery and Nonlinear Morphology

Equation discovery asks whether cliffiness can be explained without regimes. The best compact stepwise equation is:

```text
log_breadth + breadth + breadth^2 * elevation
```

It reaches grouped CV R2 of 0.5964. This is meaningful: nonlinear breadth effects and breadth/elevation interactions matter. But random forests and RBF morphology models perform better, indicating that the compact equation is an approximation rather than a complete law.

![Figure 5. Nonlinear model comparison.](../reports/topology/kernel_robustness_model_comparison.png)

**Figure 5. Nonlinear model comparison.**
Comparison of linear, polynomial, tree, kNN, boosting, and RBF morphology models for cliffiness. Nonlinear models outperform linear morphology under grouped validation, supporting the claim that cliffiness requires local nonlinear structure. Source: `../reports/topology/kernel_robustness_model_comparison.png`. Section: Equation Discovery.

**Table 2. Competing explanation results.**

| Explanation | Representative Result | Interpretation |
| --- | ---: | --- |
| Linear morphology | CV R2 -0.1786 | Fails for cliffiness under experiment-family grouped validation. |
| Scalar coordinate | Arc-coordinate CV R2 -0.1232 | Cliffiness is not a one-dimensional morphology coordinate. |
| Compact equation | CV R2 0.5964 | Useful partial approximation, not a complete law. |
| Frontier-only | CV R2 -0.2265 | Frontier distance alone fails. |
| Frontier + morphology | CV R2 0.0553 | Adds little under grouped validation. |
| RBF/kernel reference | Best grouped CV R2 0.7422 | Strong nonlinear morphology reference. |
| Random forest reference | CV R2 0.6830 in equation discovery; 0.6831 in frontier report | Local nonlinear structure is predictive. |

### 4.5 Frontier Geometry

Frontier geometry was tested as a falsification attempt. If cliffiness were mainly a distance-to-frontier phenomenon, frontier-only features should predict it well. They do not. Under experiment-family grouped CV, frontier-only features have R2 -0.2265, worse than linear morphology. Frontier plus morphology improves only to 0.0553, far below compact equations and nonlinear references.

![Figure 6. Frontier geometry falsification.](../reports/topology/frontier_model_comparison.png)

**Figure 6. Frontier geometry falsification.**
Grouped-CV comparison of frontier-only, frontier-plus-morphology, compact equation, random forest, and RBF references. Frontier distance is not sufficient to explain cliffiness. Source: `../reports/topology/frontier_model_comparison.png`. Section: Frontier Geometry.

### 4.6 Regime Transition Analysis

The static atlas supports regime structure. Transition analysis asks whether interventions move allocators through regime space coherently. Native HDBSCAN labels were reconstructed on the classical atlas and consolidated to the four-regime vocabulary. Exact matched classical rows are marked `native_cluster`; all other pooled rows are `classifier_projection` labels from a RandomForest classifier trained on breadth/elevation.

**Table 5. Regime transition summary.**

| Intervention Type | Regime-Change Probability | Median Delta Cliffiness | Dominant Transition | Interpretation |
| --- | ---: | ---: | --- | --- |
| Allocator family change | 1.0000 | 0.0000 | 0->2 | Model class changes move through regimes, but median cliffiness change is neutral because directions vary. |
| Objective intervention | 1.0000 | -0.1176 | 0->3 | Objective changes induce strong regime movement and often reduce cliffiness. |
| Training epoch | 0.9800 | -0.4251 | 1->3 | Training trajectories frequently move between regimes and often reduce cliffiness. |
| Density increase | 0.7500 | 0.4564 | 1->3 | Density perturbations produce frequent regime changes but may increase cliffiness in this projected vocabulary. |
| Separability increase | 0.3786 | 0.0000 | 3->1 | Separability often moves allocators back toward lower-cliff states, but many paths remain in-regime. |
| Dropout increase | 0.0000 | 0.3000 | none | Dropout changes cliffiness within regime rather than changing projected regime. |
| Fragmentation increase | 0.0000 | 0.0000 | none | Fragmentation effects appear mostly within-regime in this projection. |

The strongest transition result is predictive. Continuous morphology deltas alone predict path-level delta cliffiness with CV R2 0.1138. Regime transition features alone reach 0.0672. Combining morphology deltas with start/end regime raises CV R2 to 0.2776. This supports the claim that intervention paths reveal structured movement through regime space.

![Figure 7. Transition paths in morphology space.](../reports/topology/regime_transition_paths_morphology_space.png)

**Figure 7. Transition paths in morphology space.**
Projected transition paths over breadth/elevation space, with arrows showing observed regime-changing movements. The figure supports the Act III claim that interventions do not only perturb metrics; they move allocators through morphology regimes. Source: `../reports/topology/regime_transition_paths_morphology_space.png`. Section: Regime Transition Analysis.

![Figure 8. Regime stability by intervention.](../reports/topology/regime_stability_by_intervention.png)

**Figure 8. Regime stability by intervention.**
Probability of regime change by intervention and step size. Density, separability, objective, and training axes show different stability patterns, distinguishing smooth within-regime movement from phase-like transitions. Source: `../reports/topology/regime_stability_by_intervention.png`. Section: Regime Transition Analysis.

![Figure 9. Cliffiness change by intervention.](../reports/topology/regime_transition_delta_cliffiness.png)

**Figure 9. Cliffiness change by intervention.**
Distribution of path-level cliffiness changes by intervention type. The figure helps connect regime movement to operational changes in threshold brittleness. Source: `../reports/topology/regime_transition_delta_cliffiness.png`. Section: Regime Transition Analysis.

## 5. Discussion

The evidence supports a three-act structure.

**Act I: Morphology.** Accessibility behavior can be represented in a breadth/elevation morphology space. Persistence is largely a morphology-position property. Classical allocators occupy the same broad morphology support as earlier neural and synthetic experiments.

**Act II: Regimes.** Cliffiness is not explained by one global linear, scalar, boundary, or frontier variable. It is organized by macro-regimes. Native HDBSCAN/UMAP replicates the macro-regime result while warning against point-level cluster claims.

**Act III: Transitions.** Interventions induce structured movement through regime space. Some interventions move allocators between regimes; others change cliffiness within regime. Regime transition information improves prediction of cliffiness changes beyond continuous morphology deltas alone.

This hierarchy clarifies the main theory. The global equation hypothesis is only partially supported. The frontier-distance hypothesis is rejected by grouped validation. The regime hypothesis is currently the strongest explanation. The concise claim is:

```text
cliffiness ~= regime + local morphology
```

This does not mean regimes are causal mechanisms. They are explanatory states induced by mechanisms such as objective weighting, density, separability, fragmentation, tree discretization, boosting behavior, and ensembling. The operational value is that regimes give practitioners a language for threshold-path risk. A broad-flat high-cliff allocator may require different governance than a quantized-floor allocator or an elevated plateau allocator.

## 6. Limitations

The first limitation is external validity. Most evidence comes from controlled synthetic severe-skew experiments and classical allocator sweeps. Real or semi-real imbalanced benchmarks are needed before a broad submission.

The second limitation is clustering stability. Native HDBSCAN/UMAP replicates macro-regimes, not exact point-level cluster identity. The DBSCAN-vs-HDBSCAN ARI is 0.1705, so the manuscript should avoid pointwise cluster robustness claims.

The third limitation is local-law instability. Fallback regimes showed partial within-regime ridge improvement, but native HDBSCAN did not replicate this improvement cleanly. Local laws should be treated as a hypothesis, not as a settled main result.

The fourth limitation is transition labeling. Most transition-analysis rows are classifier-projected into the native regime vocabulary. This is useful for exploratory dynamics but weaker than native clustering for every pooled experiment family.

The fifth limitation is coordinate dependence. Breadth and elevation are meaningful operational coordinates, but the paper does not prove they are uniquely optimal. Topology-aware, calibration-aware, or density-aware coordinates may improve the regime map.

The sixth limitation is interpretation. Regimes are explanatory compression, not necessarily optimal predictors. Confounding analysis shows substantial R2 gains from regime information, but bucket-accuracy gains beyond family and skew are modest.

## 7. Future Work

Future work should validate regimes on real or semi-real imbalanced datasets, repeat atlas construction for related survival-shape metrics, and test topology-augmented regimes. Neural accessibility dynamics deserve a more focused study because training trajectories show frequent regime movement. Causal intervention studies are also needed: the current transition graph is observational, even though the sweeps are controlled. Finally, regime-aware optimization is a natural goal. If broad high-cliff basins are operationally dangerous, training procedures could penalize movement into them or encourage elevated plateau behavior.

## 8. Conclusion

Accessibility in rare-event learning is not only a ranking problem. It is a threshold-path morphology problem. Breadth and elevation provide a useful state space for minority score allocation. Persistence is largely explained by morphology position. Cliffiness is not.

The main result is that cliffiness is regime-structured. Macro-regimes survive native HDBSCAN/UMAP replication, regime information contributes beyond family and skew in regression checks, and intervention paths show structured movement through regime space. The appropriate scientific claim is conservative but strong: accessibility behavior is organized into morphology regimes, and those regimes provide a useful abstraction for understanding threshold brittleness in rare-event learning.

## Review Notes for v0.3

### Figures Needing Regeneration

| Figure | Issue | Recommendation |
| --- | --- | --- |
| Figure 1 | Existing draft figure may not have final typography or labels. | Regenerate with colorblind-safe palette and larger labels. |
| Figure 2 | Native UMAP figure is useful but likely needs publication styling. | Regenerate with clearer panel labels and consistent colors. |
| Figure 3 | Dendrogram should show selected four-regime cut. | Add cut line and macro-regime annotations. |
| Figure 4 | Compression curve should highlight k=4 and cluster baseline. | Regenerate for final paper. |
| Figures 7-9 | Transition figures are exploratory and may be visually dense. | Decide main text vs appendix after advisor review. |

### Tables With TODO Values

No central table uses invented values. If submission requires native-regime confounding rather than fallback-regime confounding, Table 4 should be recomputed.

### Claims Needing Advisor Feedback

1. Whether transition analysis should remain in main Results or move to Appendix.
2. Whether the title `Accessibility Regimes in Rare-Event Learning` is broad enough and accurate.
3. Whether local laws should appear in the core thesis or only in Future Work.
4. Whether the paper should target a workshop first or wait for real-data validation.
5. Whether regime names should be polished before advisor circulation.

### Caveat Checklist

1. Native HDBSCAN/UMAP replicates macro-regimes, not point-level cluster identity.
2. Local-law improvement did not replicate cleanly under native HDBSCAN.
3. Transition analysis is exploratory and relies partly on classifier-projected regime labels.
4. Most evidence remains controlled/synthetic; real-data validation remains future or pre-submission work.
5. Regimes are framed as explanatory compression, not optimal classifiers.
