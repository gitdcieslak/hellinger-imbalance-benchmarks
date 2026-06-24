# Morphology Regimes of Accessibility

## Abstract

Extreme class imbalance creates an operational threshold problem that is not fully described by AUROC, average precision, calibration, or accuracy. A classifier may rank rare positives adequately while still providing no stable threshold region in which minority support remains accessible. Building on the accessibility framework introduced in Paper 1, this manuscript studies accessibility as morphology: the shape of minority-positive score allocation over the threshold path. We represent allocator behavior using two primary coordinates, breadth and elevation. Breadth measures how broadly minority-positive score mass is distributed across accessible score regions; elevation measures how much minority-positive mass is lifted into high-score regions. Across neural objectives, dropout, density, separability, fragmentation, training trajectories, and a 480-run classical allocator sweep, morphology position strongly explains accessibility persistence. Cliffiness, the abruptness of minority survival loss, behaves differently. Linear morphology, scalar manifold coordinates, topology-only descriptors, boundary distance, and frontier distance fail to explain cliffiness under grouped validation, while nonlinear morphology models remain predictive. We therefore construct a morphology atlas over classical allocator behavior using UMAP visualization, HDBSCAN clustering, and hierarchical macro-regime consolidation. The atlas supports stable macro-regimes: a native HDBSCAN/UMAP replication selects four macro-regimes, preserves the broad high-cliff basin and quantized floor, yields cliffiness-bucket accuracy of 0.7152, and gives a between/within cliffiness variance ratio of 7.9470. Regime information contributes beyond allocator family and skew in regression checks, though bucket-classification gains are modest. Transition analysis further shows that controlled interventions induce structured movement through regime space, and regime transitions improve prediction of cliffiness changes beyond continuous morphology deltas alone. The resulting thesis is that accessibility morphology is not governed by one global equation or frontier distance. Instead, allocator behavior is best understood as regime-structured: cliffiness is approximately regime plus local morphology.

## 1. Introduction

Rare-event learning is usually framed as a problem of prediction under skew. A model should rank rare positives above negatives, produce calibrated probabilities, maintain useful precision and recall, or optimize a cost-sensitive loss. These goals matter, but they do not fully describe how the model behaves when it must be deployed at a threshold. In severe class imbalance, the practical question is often whether there exists a usable threshold region in which rare positives remain accessible. A model may achieve acceptable ranking metrics while concentrating minority examples into a narrow score band. Such a model can look useful by AUROC but brittle under threshold selection.

Paper 1 introduced accessibility as a threshold-path property. Instead of evaluating one threshold or one ranking summary, accessibility asks how minority support survives as the threshold tightens. The threshold path can be persistent, gradual, quantized, or cliff-like. A high-survival path gives the practitioner room to choose a threshold without immediately losing the minority class. A cliff-like path creates operational risk: small changes in threshold can remove many positives at once.

The present paper asks whether accessibility behavior can be organized geometrically. The starting hypothesis was simple: perhaps accessibility level and accessibility dynamics are both functions of position in a morphology space. We define this space through minority score-allocation coordinates, especially breadth and elevation. Breadth measures the spread of minority-positive score mass over accessible score regions. Elevation measures how much minority-positive score mass reaches high-score regions. Together, these coordinates describe the morphology of allocation rather than only its ranking performance.

The empirical answer is mixed in an informative way. Accessibility persistence is largely explained by morphology position. Cliffiness is not. Linear breadth and elevation, scalar manifold coordinates, topology-only descriptors, simple decision boundaries, and morphology frontier distances do not explain cliffiness well. Nonlinear morphology models do, but attempts to compress that nonlinear signal into a single global equation or frontier-distance rule only partially succeed.

This motivates the central question of the paper:

```text
Can allocator accessibility behavior be organized into identifiable regimes?
```

The answer developed here is yes. Allocators occupy a shared morphology space, but cliffiness is structured by macro-regimes inside that space. These regimes are not merely model-family labels. They summarize operational states such as quantized floors, broad low-elevation high-cliff basins, and elevated broad plateaus. Regime membership predicts accessibility behavior, contributes beyond family and skew in regression checks, and provides a vocabulary for describing how interventions move allocators through morphology space.

The paper makes six contributions.

1. It presents accessibility morphology as a score-allocation view of rare-event threshold usability.
2. It shows that accessibility persistence is largely explained by breadth/elevation morphology position.
3. It falsifies several simpler explanations of cliffiness, including linear morphology, scalar manifold coordinates, frontier distance, and simple boundary distance.
4. It constructs a morphology atlas and consolidates it into interpretable macro-regimes.
5. It validates the regime result with native HDBSCAN/UMAP, confounding checks, and transition analysis.
6. It argues that cliffiness is best understood as regime plus local morphology, not as one global equation.

[FIGURE 1 HERE: Accessibility morphology framework schematic]

## 2. Background

The accessibility framework treats a classifier as an allocator of minority-positive examples across score space. Given model scores and a moving threshold, we measure the fraction of minority positives that remain above threshold. This produces a minority survival curve. The curve is the object of interest: it describes threshold-mediated access to the rare class.

The first major quantity is accessibility persistence. In this manuscript, persistence is measured primarily by minority survival AUC. A high survival AUC means that minority support remains available over more of the threshold path. Persistence is an accessibility-level measure.

The second major quantity is accessibility cliffiness. Cliffiness measures whether survival loss is smooth or abrupt. High cliffiness means that survival is lost in sharp drops. This is operationally important because threshold selection becomes brittle when small threshold changes produce large minority-support losses.

Morphology describes the allocation shape that gives rise to these threshold-path outcomes. The most important morphology coordinates are breadth and elevation. Breadth is derived from the spread or entropy of minority-positive scores. High breadth means positives are distributed over a wider score range. Elevation is derived from the concentration of positives in high-score regions. High elevation means positives are lifted into score regions that survive strict thresholds.

Breadth and elevation are not conventional performance metrics. A broad allocator can be good or bad depending on elevation. Low-elevation breadth may indicate a broad but inaccessible score distribution. High elevation with low breadth may indicate strong but quantized access. This is why the morphology view is useful: it distinguishes score-allocation states that may have similar AUROC or AP but different threshold usability.

The broader morphology dataset combines several controlled experiments. These include training trajectories, weighted dropout sweeps, density threshold sweeps, density/separability factorial sweeps, fragmentation sweeps, MLP objective topology experiments, and a classical allocator morphology sweep. The classical sweep contains 480 successful runs over CART, HDDT, Bagged HDDT, Random Forest, XGBoost, and LightGBM. All six classical allocator families fall inside the prior morphology support observed in neural and synthetic experiments, indicating that the morphology space is not only a neural-model artifact.

[TABLE 1 HERE: Accessibility morphology variables and operational meanings]

## 3. Morphology Atlas Construction

The atlas was constructed to test whether allocator morphologies form discrete or semi-discrete regions. The primary atlas dataset was the 480-row classical allocator morphology sweep. Each run contributed breadth, elevation, survival AUC, cliffiness, and related score-allocation summaries. Frontier-derived features were also attached, including vertical slack, normalized position between envelopes, local frontier width, nearest frontier distance, and frontier curvature. The atlas feature set therefore combined base morphology coordinates with local frontier descriptors.

Features were robust-scaled. PCA retaining 95 percent variance was used for clustering diagnostics and for the initial fallback atlas. The initial implementation used PCA(2) visualization and DBSCAN fallback labels because UMAP and HDBSCAN were unavailable. That fallback atlas was useful for discovery, but the current manuscript treats native HDBSCAN/UMAP as the robustness reference.

The native replication used UMAP with `n_neighbors=15`, `min_dist=0.1`, and `random_state=42` for visualization. HDBSCAN was run with `min_cluster_size=15` and `min_samples=5` for the primary native atlas. Sensitivity runs varied `min_cluster_size` over 10, 15, 20, and 30. At 15, HDBSCAN produced 9 clusters with noise fraction 0.3271 and silhouette 0.4692. At 20 and 30, the cluster count compressed to 4 and 3 respectively, while morphology regions remained visible.

The native and fallback point-level cluster assignments are not identical. DBSCAN fallback versus native HDBSCAN had ARI 0.1705 and NMI 0.5147. This is a caution against claiming pointwise cluster stability. The appropriate robustness claim is macro-regime stability: after hierarchical consolidation, the major operational regions survive.

Macro-regime consolidation was performed on cluster centroids. Each centroid included mean breadth, mean elevation, mean cliffiness, mean survival AUC, mean vertical slack, and mean frontier position. Agglomerative clustering was applied to the centroid table for k from 2 to 8. Compression curves measured how much cliffiness-bucket predictability was retained as the number of regimes decreased. The preferred representation is a four-regime vocabulary because it balances interpretability, variance separation, and predictive retention. The native HDBSCAN rerun also selected four macro-regimes.

[FIGURE 2 HERE: Morphology phase diagram]

[FIGURE 3 HERE: UMAP morphology space colored by family, cliffiness, and regime]

## 4. Results

### 4.1 Morphology Atlas

Allocator morphologies form non-random regions in breadth/elevation space. In the fallback atlas, cluster ID predicted cliffiness buckets with 0.7710 cross-validated accuracy against a 0.3333 baseline. Between-cluster cliffiness variance was about 4.94 times mean within-cluster variance. The native HDBSCAN replication strengthened the macro-regime separation result: after consolidation, the between/within cliffiness variance ratio was 7.9470, and the broad high-cliff basin and quantized floor remained recognizable.

The atlas also clarifies why morphology position is useful but insufficient. Broadness alone is not the answer. Some broad allocators are elevated and moderately persistent; others are broad but low-elevation and highly cliff-like. Low-breadth regions include quantized floor behavior, especially for single-tree-like allocators. Cliffiness overlays show that high-cliff behavior concentrates in particular regions rather than increasing monotonically with one coordinate.

[FIGURE 4 HERE: Cliffiness overlay in morphology space]

[TABLE 2 HERE: HDBSCAN sensitivity and atlas variance decomposition]

### 4.2 Regime Consolidation

The fallback atlas discovered 14 clusters, which is too many for a useful operational theory. Hierarchical consolidation compressed these clusters into four macro-regimes. The fallback four-regime representation preserved most of the cluster-level cliffiness information: cluster ID predicted cliffiness buckets with accuracy 0.7710, while four regimes predicted them with accuracy 0.7328. The predictive loss was only 0.0382. Four regimes explained 0.6861 of cliffiness variance.

The native HDBSCAN rerun also selected four macro-regimes. Native regime ID predicted cliffiness buckets with accuracy 0.7152. This is lower than the fallback cluster result but still far above the 0.3344 baseline. The native result is therefore an acceptable-success replication of the regime theory, not a replication of exact clusters.

The working regime names are:

1. Quantized floor: low breadth, low elevation, low survival, and typically low cliffiness because positives are concentrated near a floor-like score state.
2. Broad-flat high-cliff basin: broad but low-elevation morphology with high cliffiness and low survival.
3. Elevated broad plateau: broad morphology with higher elevation and improved survival, but still nontrivial cliffiness.
4. Mixed transition: intermediate or elevated states that connect the floor and broad regimes.

These labels are intentionally operational rather than algorithmic. The regime number is not the scientific object; the morphology state is. The most important point is that high cliffiness is not a smooth global function of breadth and elevation. It is concentrated in recognizable morphology regions.

[FIGURE 5 HERE: Regime consolidation dendrogram]

[FIGURE 6 HERE: Regime compression curve]

### 4.3 Confounding Analysis

A natural objection is that regimes might only proxy allocator family or skew. This objection is partly expected: allocator families should occupy different operational regimes. However, if regime information adds nothing beyond family and skew, then the regime vocabulary is scientifically weak.

The confounding check compared held-out cliffiness prediction using family and skew baselines against models that also included regime ID. For cliffiness buckets, family plus skew achieved 0.8015 accuracy. Adding regime increased this to 0.8117, a modest gain of 0.0102. For continuous cliffiness, family plus skew achieved CV R2 of 0.6361. Adding regime increased CV R2 to 0.7317, a gain of 0.0956. When breadth and elevation were included with family and skew, adding regime increased CV R2 from 0.6354 to 0.7772 and bucket accuracy from 0.8142 to 0.8346.

The conclusion is mixed but useful. Regime ID is not merely redundant with family and skew in regression. It adds substantial held-out R2. However, bucket-classification gains are modest because family, skew, and coordinates already classify cliffiness buckets well. The manuscript should therefore frame regimes as an explanatory and compression vocabulary, not necessarily the best possible classifier.

[FIGURE 7 HERE: Confounding comparison, family+skew vs family+skew+regime]

### 4.4 Equation Discovery

Equation discovery was a deliberate attempt to avoid overusing clustering. If cliffiness could be explained by a compact global equation, then regimes might be unnecessary.

The best compact equation selected by stepwise regression was:

```text
log_breadth + breadth + breadth^2 * elevation
```

This equation reached grouped CV R2 of 0.5964 over the 4785-row pooled dataset. Lasso and ElasticNet sparse expansions reached 0.4858 and 0.4600, respectively. A random forest on breadth and elevation reached 0.6830. RBF morphology models in the robustness analysis reached still higher values, with best experiment-family grouped CV R2 of 0.7422.

This is a partial success. The compact equation shows that cliffiness is not arbitrary; nonlinear breadth effects and breadth/elevation interactions matter. But it is not a universal law. It underperforms nonlinear local models and does not remove the need for regime structure. The appropriate interpretation is that local equations exist, but no stable global law has emerged.

[TABLE 3 HERE: Equation discovery model comparison]

### 4.5 Frontier Geometry

Frontier geometry was another falsification attempt. The hypothesis was that cliffiness might be explained by distance from the feasible morphology frontier. We fitted upper and lower envelopes in breadth/elevation space and computed vertical slack, normalized slack, nearest frontier distance, frontier curvature, and related quantities.

The frontier-only hypothesis failed under grouped validation. Under experiment-family grouped CV, frontier-only features had R2 of -0.2265, worse than linear morphology. Frontier plus morphology improved to 0.0553, but remained far below the compact equation, random forest, and RBF references. The RBF reference reached 0.7343 in the frontier comparison report.

This negative result is important. Cliffiness is not primarily a distance-to-frontier phenomenon, at least under the tested envelope construction. Frontier features may contain local signal, but the stronger explanation is region-specific nonlinear morphology.

[FIGURE 8 HERE: Frontier geometry model comparison]

### 4.6 Regime Transition Analysis

The static atlas supports regime structure. The next question is whether interventions move allocators through regime space in coherent ways. We analyzed pooled transition paths from allocation trajectories, weighted dropout sweeps, density threshold sweeps, density/separability factorial sweeps, fragmentation sweeps, MLP objective topology, and classical allocator morphology. Native HDBSCAN labels were reconstructed on the classical atlas and consolidated to the four-regime vocabulary. Exact matched classical rows were marked `native_cluster`; all other pooled rows were assigned by a RandomForest classifier projection from breadth/elevation into the native regime vocabulary.

The transition analysis found structured movement. The most common directed edges were from one mixed/elevated regime into the transitional broad ridge under density increase, and the reverse edge under separability increase. Objective interventions often moved allocators from mixed morphology into the transitional broad ridge. Allocator family changes frequently moved between the quantized floor and broader regimes.

At the intervention level, transition probabilities were high for allocator family change, objective intervention, training epoch, and density increase. Allocator family change, objective intervention, and training epoch had transition probabilities near or equal to 1.0. Density increase had transition probability 0.7500. Separability increase was more moderate at 0.3786. Dropout increase and fragmentation increase did not usually change projected regime in this analysis, although they changed cliffiness within regime.

The strongest quantitative transition result is that regime transitions improve prediction of path-level cliffiness change. Continuous morphology deltas alone predicted delta cliffiness with CV R2 of 0.1138. Regime transition features alone reached 0.0672. Combining morphology deltas with start/end regime raised CV R2 to 0.2776. This meets the predefined exceptional-success criterion for the transition analysis: regime transitions explain changes in cliffiness better than continuous morphology deltas alone.

This result should still be framed carefully. The paths are observational sweeps, not randomized causal interventions over all possible allocators. Many regime labels outside the classical atlas are classifier projections. The best placement is an exploratory results subsection or appendix unless additional causal intervention experiments are added.

[FIGURE 9 HERE: Regime transition graph]

[FIGURE 10 HERE: Regime transition summary or intervention heatmap]

## 5. Discussion

The evidence supports a hierarchy of explanations.

The first explanation is the global equation hypothesis: cliffiness is a single function of breadth and elevation. This hypothesis is attractive because it would yield a compact theory. It is only partially supported. The best compact equation reaches CV R2 of 0.5964, which is useful but below nonlinear references. The equation also does not capture the instability of local-law results under native HDBSCAN. A global equation is an approximation, not the organizing principle.

The second explanation is the frontier-distance hypothesis: cliffiness is determined by distance from morphology boundaries or feasible envelopes. This is not supported by the current tests. Boundary distances and frontier-only features perform poorly under grouped validation. Frontier features can supplement morphology, but they do not explain cliffiness by themselves.

The third explanation is the regime hypothesis. This is the best-supported explanation. Allocator behavior occupies stable macro-regions. Those regions predict cliffiness, survive native HDBSCAN/UMAP replication, contribute information beyond family and skew in regression, and organize transition paths. The regime hypothesis also explains why a single global law is difficult: different regions have different local geometry and different operational meanings.

The resulting theory is:

```text
Accessibility level ~= morphology position
Accessibility cliffiness ~= regime + local morphology
```

This does not mean that regimes are causal mechanisms. The causal mechanisms may include objective functions, regularization, density, separability, fragmentation, tree discretization, boosting behavior, and ensemble smoothing. Regimes are an intermediate representation: they describe the morphology states that these mechanisms induce.

Operationally, the regime view changes model evaluation. Practitioners should not only ask which model maximizes AUROC or AP. They should ask what accessibility regime the allocator occupies. A broad-flat high-cliff allocator may require careful threshold governance even if ranking metrics are acceptable. A quantized-floor allocator may require calibration, resampling, or model redesign to create usable threshold states. An elevated broad plateau may offer more threshold flexibility but still require monitoring for cliffs.

For allocator design, the transition analysis suggests a route to regime-aware optimization. Training objectives and data perturbations can be viewed as movements through morphology space. Future procedures could penalize high-cliff broad basins, encourage elevated plateau behavior, or constrain transitions that produce brittle threshold paths. This is a design hypothesis, not yet a validated control method.

## 6. Limitations

The first limitation is external validity. Most evidence comes from synthetic severe-skew generators and controlled allocator sweeps. Synthetic data make density, separability, fragmentation, and skew controllable, but real datasets may contain morphology states not represented here. At least one real or semi-real imbalanced benchmark is needed before a broad venue submission.

The second limitation is regime assignment. Native HDBSCAN/UMAP supports macro-regime structure, but point-level labels changed substantially relative to the DBSCAN fallback. ARI was 0.1705 and NMI was 0.5147. The robust claim is macro-regime behavior, not exact cluster identity.

The third limitation is classifier projection. Transition analysis assigns most non-classical pooled rows to regimes using a classifier trained on the native classical atlas. These projected labels are useful for exploratory dynamics, but they are not native HDBSCAN assignments. The transition results should therefore be interpreted as projected regime dynamics.

The fourth limitation is local-law instability. Fallback regimes showed partial within-regime ridge improvement, but the native HDBSCAN rerun did not replicate that improvement. Under native HDBSCAN, global ridge CV R2 was 0.5254, while non-constant within-regime ridge scores were lower. Local laws should remain a hypothesis rather than a main replicated claim.

The fifth limitation is coordinate dependence. Breadth and elevation are operationally meaningful, but they are engineered morphology coordinates. The current results do not prove that these are the only or optimal coordinates. Topology-augmented, calibration-aware, or density-aware coordinates may improve regime definitions.

The sixth limitation is extrapolation. Kernel morphology robustness is strong under experiment-family and model-grouped validation, but source-family holdout is poor. The theory should be scoped to observed severe-skew morphology support, not claimed as universal extrapolation.

The seventh limitation is causality. Transition paths arise from controlled sweeps, but they are not randomized causal interventions over all allocator designs. The transition graph is evidence of structured movement, not proof that a given intervention will always cause a specific regime shift.

## 7. Future Work

The first direction is real-data validation. A small number of real or semi-real imbalanced datasets would test whether the same morphology regimes appear outside synthetic severe-skew generators.

The second direction is topology-augmented regimes. Topology does not directly explain cliffiness as a low-dimensional predictor, but it may generate transitions through morphology space. Combining local topology with breadth/elevation may improve regime assignment and transition prediction.

The third direction is neural accessibility dynamics. Training trajectories already show regime movement. A deeper study could track how objectives, dropout, weighting, oversampling, and calibration move neural allocators through morphology space over epochs.

The fourth direction is causal intervention studies. The current transition graph is observational. A stronger study would deliberately perturb objective terms to move allocators between regimes and measure resulting changes in survival and cliffiness.

The fifth direction is regime-aware optimization. If high-cliff basins are operationally dangerous, training objectives could penalize morphology states associated with brittle threshold paths. Conversely, objectives could encourage elevated plateau behavior or constrain movement near regime boundaries.

The sixth direction is metric-dependence testing. The atlas should be repeated for related dynamic metrics such as maximum survival drop and effective drop count to verify that regimes are not artifacts of one cliffiness definition.

## 8. Conclusion

Accessibility under extreme class imbalance is not only a ranking problem. It is a threshold-path morphology problem. Breadth and elevation provide a useful morphology space for minority score allocation. Persistence is largely explained by position in that space. Cliffiness is not.

The main finding of this manuscript is that cliffiness is regime-structured. Global equations, scalar coordinates, boundary distances, and frontier distances do not fully explain it. Morphology regimes do. Native HDBSCAN/UMAP replication preserves macro-regime structure, confounding checks show that regimes add regression signal beyond allocator family and skew, and transition analysis shows that interventions induce structured movement through regime space.

The strongest current statement is therefore:

```text
cliffiness ~= regime + local morphology
```

This regime abstraction is useful because it connects model choice, data perturbation, and threshold usability. It gives practitioners and researchers a language for discussing not only whether a rare-event classifier ranks positives well, but what kind of accessibility state it creates and how interventions move that state.

## Figure Plan

| Figure | Draft Artifact | Purpose |
| --- | --- | --- |
| Figure 1 | New schematic needed | Accessibility morphology framework |
| Figure 2 | `reports/topology/regime_phase_diagram.png` or native replacement | Morphology phase diagram |
| Figure 3 | `reports/topology/morphology_umap_space.png` | HDBSCAN/UMAP robustness visualization |
| Figure 4 | `reports/topology/regime_cliffiness_overlay.png` | Cliffiness overlay |
| Figure 5 | `reports/topology/cluster_dendrogram.png` | Regime consolidation dendrogram |
| Figure 6 | `reports/topology/regime_compression_curve.png` | Compression curve |
| Figure 7 | New or table-based figure from `regime_confounding_report.md` | Confounding comparison |
| Figure 8 | `reports/topology/frontier_model_comparison.png` | Frontier falsification |
| Figure 9 | `reports/topology/regime_transition_graph.png` | Directed transition graph |
| Figure 10 | `reports/topology/regime_stability_by_intervention.png` | Transition stability by intervention |
