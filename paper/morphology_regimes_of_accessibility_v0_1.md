# Morphology Regimes of Accessibility

## Abstract

Extreme class imbalance creates a threshold-selection problem that is not fully described by ranking metrics, calibration metrics, or aggregate classifier accuracy. A model may rank minority examples adequately while still offering no stable operating threshold at which minority support persists. Conversely, a model may improve minority access over part of the threshold path while introducing sharp drops that make operational control brittle. This paper introduces an accessibility morphology view of imbalanced classification. We represent allocator behavior using morphology coordinates derived from minority score allocation, primarily accessibility breadth and accessibility elevation, and study two threshold-path outcomes: accessibility persistence, measured by minority survival AUC, and accessibility cliffiness, measured by the abruptness of minority survival loss.

Across synthetic severe-skew experiments, neural objectives, density and fragmentation sweeps, dropout interventions, trajectory studies, and a 480-run classical allocator sweep over CART, HDDT, Bagged HDDT, Random Forest, XGBoost, and LightGBM, we find that accessibility level is largely explained by morphology position. In pooled pre-classical morphology analyses, linear breadth/elevation predicted minority survival AUC with grouped CV R2 of 0.8580. With classical allocators included, linear morphology remained strong for survival AUC in robustness analysis, with grouped CV R2 of 0.8745 under experiment-family grouping. Cliffiness behaves differently. Linear morphology failed for cliffiness, with grouped CV R2 around -0.1786 after adding classical allocators. Scalar manifold coordinates, topology-only descriptors, simple boundary distance, and morphology frontier distance also failed to explain cliffiness.

Nonlinear morphology models robustly explain cliffiness. In the full pooled dataset including classical allocators, 18 nonlinear models beat linear morphology under experiment-family grouped CV, and all nine tested RBF bandwidths exceeded linear morphology by at least 0.25 CV R2. The best grouped RBF model reached cliffiness CV R2 of 0.7422. However, subsequent equation and frontier analyses showed that this nonlinear signal is not reducible to a single global equation or simple frontier distance. A compact stepwise equation, `log_breadth + breadth + breadth^2 * elevation`, reached CV R2 of 0.5964, but frontier-only geometry performed poorly.

We therefore construct a morphology atlas over classical allocator behavior. Unsupervised clustering revealed stable regime structure: cluster assignment predicted cliffiness buckets with 0.7710 cross-validated accuracy against a 0.3333 baseline, and between-cluster cliffiness variance was about 4.94 times within-cluster variance. Hierarchical consolidation compressed 14 discovered clusters into 4 macro-regimes while preserving most predictive power: cliffiness-bucket accuracy fell only from 0.7710 to 0.7328, and the four-regime representation explained 0.6861 of cliffiness variance. A native HDBSCAN/UMAP robustness rerun preserved the main regime result, selecting 4 macro-regimes with cliffiness-bucket accuracy 0.7152 and between/within cliffiness variance ratio 7.9470. Local-law evidence is weaker: fallback regimes showed partial within-regime improvement, but this improvement did not replicate under native HDBSCAN. These findings suggest that accessibility morphology is not merely a neural-network phenomenon, but a cross-allocator structure of operational threshold behavior under extreme skew.

## 1. Introduction

Class imbalance is usually discussed as a problem of ranking, calibration, loss weighting, resampling, or decision threshold selection. These are all important. They are also incomplete. In severe skew, the deployment question is not only whether positives are ranked above negatives on average. The practitioner also needs to know whether there is a usable threshold region where minority examples remain accessible and where small threshold changes do not create catastrophic loss of minority support.

This distinction matters because ranking quality and operational threshold usability can diverge. A model can have an acceptable AUROC while assigning minority examples to a narrow or quantized score region. Such a model may look useful under ranking metrics but behave poorly when an operating threshold must be chosen. Similarly, a training intervention may improve minority survival over thresholds while increasing cliffiness, producing a more accessible but less controllable threshold path. This paper calls that operational property accessibility.

Accessibility is the availability of minority examples along the threshold path. It is not a single threshold metric. It is a shape property of the score allocation induced by a learner. If a model lifts many minority examples into high-score regions, accessibility may persist over more threshold values. If it distributes minority score mass broadly, the threshold path may offer more operational flexibility. If it compresses minority scores into discrete values or creates sharp separations, the path may contain cliffs: abrupt drops in minority support that make threshold tuning brittle.

The central idea of this paper is that accessibility behavior can be studied as morphology. We define accessibility morphology through score-allocation coordinates. The most important coordinates are accessibility breadth and accessibility elevation. Breadth measures how broadly minority-positive score mass is distributed across accessible score regions. Elevation measures how much minority-positive mass is lifted into a high-score, high-accessibility region. These coordinates form a morphology state space. We then study two outcomes on top of that space. Accessibility persistence, measured by minority survival AUC, describes how much minority support survives as the threshold tightens. Accessibility cliffiness describes how abruptly that support is lost.

The paper began from a simple hypothesis: perhaps accessibility level and dynamics are both functions of morphology position. The completed experiments support only half of that hypothesis. Accessibility level behaves largely like a morphology-position property. Cliffiness does not. Linear breadth/elevation, scalar manifold coordinates, topology descriptors, simple equations, SVM/logistic boundaries, and morphology frontier distances all fail to fully explain cliffiness. Nonlinear morphology models do explain cliffiness robustly, but they do so by exploiting local structure. The question is what structure those nonlinear models are using.

The answer developed here is regime structure. Instead of treating cliffiness as one global function over morphology coordinates, we construct a morphology atlas over allocator behavior. The atlas reveals discrete or semi-discrete phases of allocator behavior. These phases can be consolidated into a small set of interpretable operational regimes. The resulting theory is not that cliffiness equals a single frontier distance or one universal equation. The replicated claim is:

```text
global cliffiness = morphology regime selection + residual local structure
```

This shift matters. A single global equation is attractive but brittle. A regime-based theory is more consistent with the evidence: broad low-elevation allocators, quantized floor allocators, elevated broad plateau allocators, and transitional allocators occupy different cliffiness regimes. Whether those regimes also support stable simple local equations remains an open refinement.

The contributions of this draft are as follows.

1. We define accessibility morphology as an operational representation of minority score allocation under extreme skew.
2. We show that accessibility level, measured by minority survival AUC, is largely predicted by morphology position.
3. We show that accessibility cliffiness is not explained by linear morphology, scalar manifold coordinates, topology-only descriptors, or frontier distance.
4. We show that nonlinear morphology models robustly explain cliffiness across bandwidths, validation schemes, and smoother families.
5. We construct a morphology atlas and consolidate it into four macro-regimes that preserve most of the cluster-level cliffiness information, with native HDBSCAN/UMAP replication supporting the regime result.
6. We provide partial evidence for local accessibility laws, while noting that within-regime equation improvement does not replicate under native HDBSCAN.

[FIGURE 1 HERE: Accessibility Morphology Framework schematic]

## 2. Related Work

Research on imbalanced classification has developed many strategies for improving minority-class performance. These include resampling, class weighting, cost-sensitive learning, specialized splitting criteria, threshold moving, calibration, and ranking-based evaluation. Hellinger-distance decision trees and related skew-aware tree methods address the known weakness of impurity criteria under class imbalance. Boosted trees and random forests are often strong tabular baselines. Neural models can be adjusted through weighting, oversampling, focal objectives, logit adjustment, and regularization.

Most of this literature focuses on the relationship between learning procedure and predictive performance. Common performance views include AUROC, average precision, recall at a chosen threshold, calibration error, and Brier score. These metrics answer important questions, but they do not directly describe the shape of threshold accessibility. AUROC summarizes pairwise ranking. Average precision emphasizes precision-recall behavior over ranked lists. Calibration measures probability fidelity. None of these directly asks whether the minority class remains accessible over a stable threshold interval.

Threshold analysis is closer to our question, but threshold analysis often evaluates a small set of operating points or optimizes a threshold according to an external utility. Accessibility morphology instead treats the threshold path itself as an object of study. A model induces a minority survival curve as the threshold moves. The shape of that curve can be persistent, smooth, brittle, or cliff-like. This is the operational object we study.

The paper also connects to manifold and phase-space views of model behavior. A classifier can be represented not only by its parameters or predictions but by the geometry of its score distribution. If many models and interventions can be embedded in a shared morphology space, then model behavior can be compared through trajectories, regimes, and transitions. This is common in dynamical systems language, but less common in imbalanced classification evaluation.

Finally, this paper relates to interpretable model diagnostics. Global equations, frontier distances, clustering, and local laws are all attempts to make nonlinear model behavior understandable. We use nonlinear models first as diagnostic evidence, then ask what interpretable structure they exploit. The resulting regime theory is not intended to replace conventional metrics. It is intended to supplement them with an operational description of threshold usability.

## 3. Accessibility Morphology Framework

Accessibility morphology begins with the minority-positive score distribution. Given a classifier and a test set, we compute scores for the positive class. We then examine how positive examples are allocated across score bins and thresholds.

Accessibility breadth measures the spread of minority-positive score mass. In the implemented analyses, breadth is represented by positive-score histogram entropy and related effective-bin measures. A high-breadth allocator spreads minority scores across a wider score range. A low-breadth allocator concentrates minority scores into fewer bins. Breadth is not automatically good or bad. Broad allocation may provide threshold flexibility, but if scores are broadly spread at low elevation, accessibility may still be poor.

Accessibility elevation measures how much minority-positive mass reaches high-score regions. In the implemented analyses, elevation is represented by top-bin mass and related peak-concentration quantities. High elevation means many minority examples are lifted into high-accessibility score regions. Low elevation means minority examples remain near the floor or are not strongly separated into high-score regions.

Accessibility persistence describes the survival of minority support along the threshold path. The primary metric is minority survival AUC. For a sequence of thresholds, we compute the fraction of positive examples whose score remains above threshold and integrate the survival curve. A high survival AUC means minority support persists as the decision threshold tightens.

Accessibility dynamics describe how survival changes. The primary dynamic metric in this paper is minority survival cliffiness. Cliffiness captures whether survival loss is distributed across thresholds or concentrated in sharp drops. High cliffiness means the threshold path contains abrupt transitions. This is operationally important because abrupt transitions make threshold selection brittle.

The first empirical question is whether these quantities form a coherent morphology state space. The answer is yes. In the morphology manifold analysis before classical allocator integration, the pooled dataset contained 4305 rows across six experiment families and 23 models. PCA on standardized breadth/elevation showed that PC1 explained 0.9301 of variance, with PC2 explaining 0.0699. Breadth and elevation were strongly negatively correlated at -0.8602. This indicates a dominant morphology axis, but not a complete one-dimensional explanation.

When classical allocators were added, the pooled data increased to 4785 rows and seven experiment families. The manifold remained stable: PC1 explained variance changed from 0.9301 to 0.8809, and the manifold shift distance was 0.2050. All six classical allocator families fell inside the prior morphology support. This is important because it means the morphology space is not a neural-only artifact. CART, HDDT, Bagged HDDT, Random Forest, XGBoost, and LightGBM occupy the same breadth/elevation support as the earlier neural and synthetic morphology experiments.

[FIGURE 2 HERE: Shared Accessibility Morphology Space with Classical Allocators]

[TABLE PLACEHOLDER: Morphology state variables and operational meanings]

## 4. Accessibility Level as Morphology Position

The second empirical question is whether accessibility level is predictable from morphology position. The evidence is strong. In the pre-classical manifold experiments, linear breadth/elevation predicted minority survival AUC with experiment-family grouped CV R2 of 0.8580. PC1 alone predicted survival AUC with CV R2 of 0.8008. PC1 plus PC2 returned to 0.8580. Kernel morphology improved survival AUC to about 0.9485, but the improvement was modest compared with cliffiness because linear morphology was already strong.

Classical allocator integration made this test stricter. In the classical-inclusive robustness report, linear morphology predicted survival AUC with experiment-family grouped CV R2 of 0.8745 and model-grouped CV R2 of 0.8912. These results support the claim that accessibility level is largely a morphology-position variable. The position does not need a complex model to be useful.

The scalar coordinate tests were more nuanced. In the classical-inclusive report, arc-coordinate survival CV R2 was 0.6442. This is meaningful, but weaker than full breadth/elevation linear morphology. This suggests that survival AUC is morphology-position based, but not fully scalar once classical allocators are included. The earlier stronger scalar story should therefore be softened. The appropriate claim is not that one universal scalar coordinate explains level under all conditions. The stronger supported claim is that morphology position, especially breadth/elevation, substantially explains accessibility level.

This distinction matters for manuscript framing. A reviewer could object that scalar survival drops after adding classical allocators. That objection is valid. The response is that the framework does not require a single scalar coordinate. It requires that accessibility level is represented in morphology space. Linear breadth/elevation remains strong for survival AUC even after adding classical allocators.

[TABLE PLACEHOLDER: Survival AUC prediction across linear, scalar, kernel, and validation schemes]

## 5. Competing Explanations of Cliffiness

The central puzzle is cliffiness. If persistence is largely morphology-position based, perhaps cliffiness is too. The evidence says no. Cliffiness is the outcome that forces a regime theory.

### 5.1 Linear Morphology Fails

Linear breadth/elevation is a poor model of cliffiness. In the pre-classical manifold report, linear morphology cliffiness CV R2 was -0.1839. With classical allocators included, linear morphology cliffiness CV R2 was -0.1786 under experiment-family grouped CV and -0.1439 under model-grouped CV. These are not merely low positive values. They are negative out-of-sample R2 values, meaning the linear model generalizes worse than a mean predictor.

The competing explanation tested here is simple: perhaps cliffiness is just another linear function of breadth and elevation. This explanation is rejected because linear morphology fails under grouped validation while succeeding for survival AUC.

### 5.2 Scalar Coordinates Fail

The manifold coordinate experiments tested whether cliffiness could be explained by PC1, PC1 plus PC2, global arc position, family-local arc position, or a scalar kernel coordinate. These scalar coordinate models failed. Before classical integration, PC1-only cliffiness CV R2 was -0.1717. PC1 plus PC2 was -0.1839. The best scalar coordinate predicted survival AUC well but cliffiness poorly, with scalar cliffiness CV R2 around -0.1521.

After classical integration, scalar-coordinate cliffiness remained weak. Arc-coordinate cliffiness CV R2 was -0.1232. This rejects the hypothesis that cliffiness is primarily a one-dimensional manifold coordinate.

### 5.3 Topology-Only Descriptors Fail as Direct Predictors

Topology studies measured component structure, fragmentation, and minority support geometry. These are useful for mechanism hypotheses. However, topology-only and topology-augmented models did not directly explain cliffiness. In the topology-augmented morphology report, topology-only cliffiness CV R2 was -0.1398. Morphology plus topology improved over morphology by only 0.0016 in the cliffiness improvement test.

This does not mean topology is irrelevant. Fragmentation studies showed that fragmentation can increase density-adjusted cliffiness, and density/separability sweeps move allocators through morphology space. The correct interpretation is that topology may be a generator of morphology transitions, not the final low-dimensional explanatory coordinate for cliffiness.

### 5.4 Boundary and Frontier Distance Fail

The boundary-distance hypothesis was attractive: perhaps cliffiness is high near a morphology transition boundary. SVM and logistic boundary distances did not support this. Boundary-distance CV R2 values were negative. Frontier geometry was a more morphology-specific version of the same idea. We fitted upper and lower envelopes in breadth/elevation space, computed vertical slack, normalized slack, distance to lower envelope, normalized position between envelopes, nearest frontier distance, and frontier curvature.

The frontier result was also weak. Under experiment-family grouped CV, frontier-only features had CV R2 of -0.2265, worse than linear morphology. Frontier plus morphology improved to 0.0553, but remained far below the compact equation, random forest, and RBF references. The frontier hypothesis is therefore only weakly or partially supported: frontier features contain some signal when combined with morphology, but cliffiness is not reducible to distance from the feasible morphology envelope.

[FIGURE 3 HERE: Competing Explanations of Cliffiness]

[TABLE PLACEHOLDER: Linear, scalar, topology, boundary, frontier, compact equation, nonlinear model comparison]

## 6. Nonlinear Morphology Is Robust

If linear, scalar, topology-only, and frontier-distance explanations fail, the next question is whether nonlinear morphology is a tuning artifact. We tested this directly through kernel morphology robustness.

The robustness study used the full pooled dataset including classical allocators: 4785 rows, seven experiment families, and 29 models. It compared linear regression, ridge, polynomial morphology of degree 2 and 3, RBF kernel approximations over nine bandwidths, kNN regressors, random forests, and gradient boosting. It evaluated random K-fold CV, experiment-family GroupKFold, model-ID GroupKFold, and source-family GroupKFold.

The result was robust under the main manuscript validation schemes. For cliffiness under experiment-family grouped CV, linear regression had R2 of -0.1786. The RBF sweep achieved positive R2 across all nine gamma values. At gamma 0.01, RBF cliffiness CV R2 was 0.2187. It increased through 0.3710, 0.5245, 0.6175, 0.6335, 0.6927, 0.7078, 0.7343, and 0.7422 as gamma increased from 0.01 to 100. All nine bandwidths beat linear morphology by at least 0.25. The bandwidth curve did not show a single narrow spike. It showed a stable region of nonlinear improvement.

Other nonlinear models also beat linear. Under experiment-family grouped CV, polynomial degree 3 reached 0.4852, random forest reached 0.6826, gradient boosting reached 0.5824, kNN variants ranged from 0.6312 to 0.6988, and the best RBF reached 0.7422. Eighteen nonlinear models beat linear morphology under grouped CV. Under model-grouped CV, the best RBF reached 0.8013.

Random CV was more optimistic, as expected. The best random-CV cliffiness R2 was 0.8864. The important point is that grouped CV remained meaningfully positive. This supports the claim that nonlinear morphology is not merely interpolating random folds.

There is one important limitation. Source-family holdout was poor. When the validation scheme held out the broad source family, performance collapsed for cliffiness. This is an extrapolation warning. The nonlinear structure is robust across experiment-family and model-family splits inside the observed morphology support, but not guaranteed to extrapolate from neural/synthetic families to classical families or vice versa when an entire source family is held out.

[FIGURE 4 HERE: Kernel Robustness Bandwidth Sweep]

[FIGURE 5 HERE: Nonlinear Model Comparison]

## 7. Morphology Equation Discovery

Nonlinear models explain cliffiness, but that does not yet provide theory. The next question is what structure they exploit.

We first tried symbolic feature expansion. Candidate features included breadth, elevation, squared and cubic terms, breadth/elevation interactions, square roots, log breadth, distance from origin, distance from broad allocator corner, distance from quantized line, and distance from manifold centroid. Lasso and ElasticNet produced sparse equations. Lasso reached CV R2 of 0.4858 with six nonzero terms. ElasticNet reached 0.4600 with seven nonzero terms.

Stepwise regression found a smaller and more useful equation. The first selected term was `log_breadth`, which alone reached CV R2 of 0.3399. Adding `breadth` raised CV R2 to 0.5918. Adding `breadth^2 * elevation` raised CV R2 slightly to 0.5964. This is close to the weak-success threshold for an interpretable equation but still below nonlinear references. The equation suggests that cliffiness depends strongly on nonlinear breadth effects, with elevation entering through an interaction term.

This is a meaningful partial result. It tells us that nonlinear models are not exploiting arbitrary noise. A compact equation captures substantial structure. But it is not the full theory. Random forest on breadth/elevation reached CV R2 of 0.6830 in the equation discovery report, and RBF models in the robustness report reached higher values. The compact equation is a useful approximation, not a complete explanation.

Partial dependence plots and fallback feature-importance plots further support the importance of breadth and interactions. SHAP was not installed during this run, so SHAP-named plots were generated using model-importance and dependence fallbacks. These figures should not be presented as SHAP evidence unless SHAP is installed and rerun.

[FIGURE 6 HERE: Equation Discovery PDPs or compact equation comparison]

[TABLE PLACEHOLDER: Sparse equation terms and grouped CV performance]

## 8. Morphology Atlas Construction

The failure of global explanations motivates a different approach: perhaps cliffiness is governed by morphology regimes. Instead of fitting one global function over breadth/elevation, we construct an atlas of allocator behavior space.

The primary atlas dataset was the 480-row classical allocator morphology sweep. It contains six model families, each with 80 runs across skew ratios and seeds: CART, HDDT, Bagged HDDT, Random Forest, XGBoost, and LightGBM. For each run, the atlas used breadth, elevation, accessibility breadth, accessibility elevation, density-separability, and frontier-derived features such as vertical slack, normalized position between envelopes, local frontier width, nearest frontier distance, and frontier curvature. Features were robust-scaled. PCA retained 95 percent variance for clustering diagnostics. The initial atlas used PCA(2) and DBSCAN fallbacks because UMAP and HDBSCAN were unavailable. A native robustness rerun later used UMAP with `n_neighbors=15`, `min_dist=0.1`, `random_state=42`, and HDBSCAN with `min_cluster_size=15`, `min_samples=5`.

Despite that caveat, the atlas showed substantial structure. Agreement across clustering methods was moderate. DBSCAN fallback versus KMeans had ARI 0.6189. KMeans versus Spectral had ARI 0.6389. Bootstrap ARI to the primary clustering had mean 0.5922. Silhouette values were also meaningful, with the primary fallback clustering silhouette at 0.4913.

The clusters carried outcome information. Cluster ID predicted cliffiness buckets with 0.7710 cross-validated accuracy against a 0.3333 baseline. It predicted survival AUC buckets with 0.6921 accuracy. Between-cluster cliffiness variance was 0.0519, while mean within-cluster variance was 0.0105, producing a between-to-within ratio of 4.9351. Cluster assignment also explained residual structure from compact equations.

These results answer the atlas question. Regimes do appear to exist in the classical allocator morphology space. They are not merely projection artifacts: the native HDBSCAN/UMAP rerun changed point-level assignments but preserved the major operational regions, including the broad high-cliff basin and the quantized floor.

[FIGURE 7 HERE: Morphology Phase Diagram]

[FIGURE 8 HERE: Cliffiness Overlay]

[TABLE PLACEHOLDER: Cluster stability and outcome separability]

## 9. Morphology Regime Consolidation

Fourteen clusters are too many for operational theory. We therefore consolidated the atlas clusters into macro-regimes. Cluster-level centroids were built from mean breadth, mean elevation, mean cliffiness, mean survival AUC, mean vertical slack, and mean frontier position. We computed Euclidean, Mahalanobis, and cosine distances, then ran agglomerative clustering on the centroids for k from 2 to 8.

The best operational choice was four macro-regimes. At k=4, silhouette was 0.4505, Davies-Bouldin was 0.6355, and cliffiness variance explained was 0.6861. Increasing to k=5 barely changed variance explained, and increasing to k=6 improved variance explained to 0.7102 but reduced parsimony. The compression curve supported k=4. Cluster ID predicted cliffiness buckets with accuracy 0.7710. Four regimes predicted cliffiness buckets with accuracy 0.7328, a loss of only 0.0382. Three regimes dropped further to 0.6616, and two regimes dropped to 0.4351. Native HDBSCAN also selected four macro-regimes, with regime cliffiness-bucket accuracy 0.7152 and a stronger between/within cliffiness variance ratio of 7.9470.

The four regimes were:

1. Quantized floor. This regime had low breadth, low elevation, low survival AUC, and low cliffiness. It was dominated by CART and LightGBM runs near the score floor. Its mean breadth was 0.2231, mean elevation 0.0659, mean cliffiness 0.0459, and mean survival AUC 0.0661.
2. Broad-flat high-cliff basin. This regime had broad, low-elevation morphology and high cliffiness. It was dominated by XGBoost, Bagged HDDT, and Random Forest. Its mean breadth was 1.1362, mean elevation 0.0236, mean cliffiness 0.5980, and mean survival AUC 0.0994.
3. Elevated broad plateau. This regime had high breadth and higher elevation, with moderate cliffiness and higher survival. It was dominated by HDDT, Random Forest, and XGBoost. Its mean breadth was 1.5900, mean elevation 0.2377, mean cliffiness 0.4472, and mean survival AUC 0.4049.
4. Mixed or elevated transition. This smaller regime included CART and LightGBM runs with higher elevation and zero observed cliffiness. Its mean breadth was 0.5944, mean elevation 0.2893, mean cliffiness 0.0000, and mean survival AUC 0.2893.

The regime labels need final naming polish, but the structure is useful. The core insight is that high cliffiness is not simply high breadth or low elevation. It is associated with particular morphology phases, especially broad low-elevation basins and transitional broad structures.

[FIGURE 9 HERE: Cluster Distance Heatmap]

[FIGURE 10 HERE: Cluster Dendrogram with k=4 cut]

[FIGURE 11 HERE: Regime Compression Curve]

[TABLE PLACEHOLDER: Four-regime characterization]

## 10. Local Accessibility Laws

The final explanatory step is local law discovery. If global cliffiness is regime selection plus local equation, then equations fitted within regimes should outperform a global equation.

The consolidation report supports this. A global ridge model using breadth and elevation had cross-validated R2 of 0.4461. Within-regime ridge models improved over the global model in regimes 0 and 1, with cross-validated R2 of 0.6066 and 0.5193. Regime 2 was slightly below the global ridge baseline but remained positive, with cross-validated R2 of 0.4382. Regime 3 is small and near-degenerate, with 28 rows and zero cliffiness, so cross-validated R2 is undefined and the in-sample score should not be interpreted.

This partially supports the local-law hypothesis. The strongest fallback-atlas evidence is that held-out local ridge equations remain predictive in the three non-degenerate regimes and improve over the global ridge baseline in two of them. The native HDBSCAN rerun weakens this claim: global ridge CV R2 was 0.5254, while the two non-constant native regimes had lower within-regime ridge CV R2 of 0.3778 and 0.2925, and two low-cliffiness regimes had undefined R2. The draft should therefore present local laws as a hypothesis motivated by the fallback atlas, not as a replicated main result.

The theory now has a clear form:

```text
allocator behavior -> morphology regime -> local accessibility law -> cliffiness
```

This is stronger than the earlier kernel-only statement, but should be read as a working model rather than a fully replicated law. The kernel result told us that nonlinear geometry matters. The atlas and consolidation results tell us what kind of nonlinear geometry matters most robustly: regime-structured behavior.

[FIGURE 12 HERE: Global vs Within-Regime Local Law Scores]

[TABLE PLACEHOLDER: Cross-validated within-regime equation scores]

## 11. Operational Consequences

The regime view has practical implications. It changes how allocator behavior should be interpreted under severe skew.

CART is consistently quantized. In the classical allocator integration report, CART was classified as a quantized allocator. In the full atlas, CART dominated low-breadth, low-elevation clusters. This matches operational intuition: tree leaves can create discrete score states, and under severe skew those states can place positives on a limited score staircase.

Bagged HDDT, HDDT, Random Forest, and XGBoost often occupy broad low-elevation or elevated broad regimes. These allocators can produce broad score distributions without necessarily lifting many positives into high-elevation regions. In the broad-flat high-cliff basin, mean cliffiness was high and mean survival AUC low. This is a dangerous operational regime: scores are broad, but accessibility remains low and brittle.

LightGBM occupied mixed regions under the severe-skew-safe settings used here. Some LightGBM runs appeared near the low-elevation floor; others appeared in transitional regions. This should not be overgeneralized as a claim about LightGBM universally. It is a placement result under the current synthetic severe-skew design and registry settings.

For model selection, the regime view suggests that a practitioner should not only ask which model has best AUROC or AP. They should ask which morphology regime the allocator occupies. A model in a broad-flat high-cliff basin may require different threshold governance than one in an elevated plateau. A quantized floor allocator may need calibration, resampling, or architecture changes to create usable threshold states. A transitional allocator may be sensitive to small data or objective perturbations.

This is also useful for intervention design. Weighting, oversampling, dropout, density, and fragmentation experiments can be interpreted as movements through morphology space. Future training procedures could target regimes directly, for example by penalizing broad low-elevation high-cliff morphology or encouraging elevated plateau behavior.

## 12. Limitations

The first limitation is synthetic data. Most experiments use synthetic severe-skew generators. The advantage is control over skew, density, separability, fragmentation, and model objective. The limitation is external validity. Real datasets may contain morphology patterns not represented here. A real or semi-real validation dataset is strongly recommended before submission.

The second limitation is clustering sensitivity. Native HDBSCAN/UMAP has now been run and supports the main regime conclusion, but point-level labels changed substantially relative to the DBSCAN fallback: ARI was 0.1705 and NMI was 0.5147. This means the paper should not claim exact cluster identity robustness. The robust claim is macro-regime structure and recognizable operational regions, not stable pointwise cluster assignments.

The third limitation is model-family confounding. Regimes are associated with allocator families. That is partly the point: allocators occupy operational regimes. A direct confounding check now suggests that regime ID is not only a family/skew proxy, but the evidence is mixed. Adding regime ID to allocator family and skew improved cross-validated cliffiness R2 from 0.6361 to 0.7317, while bucket accuracy improved only from 0.8015 to 0.8117. Adding regime ID on top of family, skew, breadth, and elevation improved R2 from 0.6354 to 0.7772 and bucket accuracy from 0.8142 to 0.8346. This supports incremental regime signal, but the classification gain is modest.

The fourth limitation is within-regime validation. Cross-validated local equation scores are now available and are positive in the three non-degenerate regimes, but the evidence is uneven: two regimes improve over the global ridge baseline, one approximately matches it, and the small zero-cliffiness regime cannot be evaluated with R2. Local laws should be framed as partially validated until regime definitions are rerun with native clustering and small-regime handling is improved.

The fifth limitation is extrapolation. Source-family holdout in kernel robustness was poor. This means the morphology model is reliable inside observed support under experiment-family and model-family grouping, but it should not be presented as guaranteed to extrapolate from one broad source family to another.

The sixth limitation is metric dependence. Cliffiness is one dynamic metric. Other survival-shape metrics, such as max drop and effective drop count, should be checked to ensure regimes are not artifacts of one metric definition.

## 13. Future Work

The immediate future work is robustness and validation. The family/skew confounding check should be repeated on the native HDBSCAN atlas. Within-regime validation also needs refinement because native HDBSCAN preserved regime structure but did not replicate the local-law improvement; small or near-constant regimes require explicit handling.

The next empirical extension is real-data validation. The paper does not need dozens of new datasets for a first submission, but at least one real or semi-real imbalanced benchmark would materially strengthen the argument. The goal would not be to reproduce every synthetic result, but to test whether accessibility morphology and regime structure appear outside the synthetic generator.

The next theoretical extension is transition dynamics. The current transition graph shows adjacency between macro-regimes. This could become a flow-field model of accessibility: interventions move allocators between regimes, and regime transitions determine changes in persistence and cliffiness. Such a model would connect morphology theory to training control.

The long-term goal is accessibility optimization. If regimes are operationally meaningful, training objectives can target them. Instead of optimizing only AUROC, AP, or loss, one could optimize for elevated plateau regimes, penalize high-cliff broad basins, or constrain morphology transitions under skew.

## 14. Conclusion

This draft argues that accessibility under extreme class imbalance is a morphology-regime phenomenon. Accessibility is distinct from ranking quality because it describes threshold-path usability. Breadth and elevation provide a morphology state space for minority score allocation. Persistence is largely explained by morphology position. Cliffiness is not.

The negative results are central. Cliffiness is not explained by linear morphology, scalar manifold coordinates, topology-only descriptors, simple boundary distances, or frontier slack. Nonlinear models explain cliffiness robustly, but equation discovery and frontier analysis show that the nonlinear signal is not captured by one global theory variable. The morphology atlas resolves this tension. Allocator behavior organizes into regimes. Four macro-regimes preserve most of the cliffiness predictability of the fallback and native atlases. Within-regime equations remain a promising but non-replicated local-law hypothesis rather than a settled result.

The strongest current theory is therefore:

```text
Accessibility level = morphology position
Accessibility dynamics = morphology regime selection + residual local structure
```

This is a stronger and more nuanced claim than the initial kernel result. It says that accessibility morphology is not a neural-network phenomenon. Across trees, ensembles, boosted learners, Hellinger-based allocators, and neural objectives, allocators occupy a shared morphology space. But the operational dynamics of that space are regime structured. This gives Paper 2 a central contribution: a morphology-regime framework for understanding and controlling threshold accessibility under extreme class imbalance.

### Remaining Risks Before Submission

| Risk | Status | Classification | Required Action |
| --- | --- | --- | --- |
| HDBSCAN/UMAP rerun status | Completed. Native UMAP/HDBSCAN produced acceptable-success replication: 4 macro-regimes, cliffiness-bucket accuracy 0.7152, variance ratio 7.9470, broad high-cliff basin and quantized floor preserved. | Completed | Use macro-regime robustness language; do not claim point-level cluster identity robustness. |
| Family-confounding check status | Partially complete. Regime ID adds held-out R2 beyond family/skew and beyond family/skew/coordinates, but bucket-accuracy gains are modest. | Recommended | Repeat after native atlas rerun and report both regression and bucket-classification deltas. |
| Within-regime CV status | Partially complete but weakened by native replication. Fallback local ridge improves over global in two non-degenerate regimes, but native HDBSCAN local ridge does not outperform global. | Recommended | Treat local laws as hypothesis/future work unless improved small-regime handling or alternative local models replicate the effect. |
| Real-data validation status | Not complete. Current evidence is mostly synthetic severe-skew plus synthetic classical allocator sweep. | Recommended | Add at least one real or semi-real imbalanced dataset if targeting a broad ML venue. |
| Publication-quality figures | Draft figures exist, but several need regeneration and consistent styling. | Required | Regenerate main figures with consistent typography, color palette, labels, and manuscript numbering. |
| Metric-dependence check | Not complete for atlas/regime analysis. | Recommended | Repeat regime analysis or summary checks using max drop and effective drop count. |
| Exact SHAP analysis | Not complete. SHAP package was unavailable and fallback feature-importance plots were generated. | Future Work | Install SHAP or remove SHAP claims from main text. |
| Causal intervention evidence | Not complete. Current evidence is descriptive and predictive. | Future Work | Design interventions that deliberately move allocators between regimes. |
