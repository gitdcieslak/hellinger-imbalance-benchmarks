# Terminology Stabilization (Provisional)

## operational accessibility

**Working definition:** Practical ability of minority examples to remain selectable/actionable under deployment thresholds.

**Use when:** Discussing threshold-mediated deployment behavior.

**Avoid using when:** Referring only to ranking quality independent of threshold.

**Related terms:** reachability, threshold-mediated deployment, occupancy persistence.

**Example from current experiments:** `mlp_bce` has low accessibility at `t=0.50` despite non-trivial AUROC.

**Caveats:** Provisional construct; operational meaning depends on deployment policy.

## reachability

**Working definition:** `R(t)=P(score>=t | y=1)`; minority survival/accessibility as threshold varies.

**Use when:** Analyzing accessibility trajectories across thresholds.

**Avoid using when:** Reporting a single-threshold metric only.

**Related terms:** recall, threshold elasticity, accessibility morphology.

**Example from current experiments:** `mlp_bce` vs `mlp_oversampled` reachability derivative differences in `results/geometry_transition_analysis/reachability_derivatives.csv`.

**Caveats:** Reachability contains recall as a special case at fixed `t`.

## allocation geometry

**Working definition:** Distributional and structural properties of predicted scores under imbalance (support, concentration, occupancy, threshold behavior).

**Use when:** Discussing model-level score allocation structure.

**Avoid using when:** Claiming formal geometric theory.

**Related terms:** accessibility morphology, occupancy, support breadth.

**Example from current experiments:** Regime summaries in `reports/*allocation_regime_summary.csv`.

**Caveats:** Empirical abstraction, not strict mathematical geometry.

## accessibility morphology

**Working definition:** Shape class of operational accessibility behavior under threshold evolution (cliff/smooth/quantized/broad/conservative).

**Use when:** Interpreting threshold trajectory patterns.

**Avoid using when:** Referring only to static score histograms.

**Related terms:** threshold morphology, regime labels.

**Example from current experiments:** `mlp_bce` cliff vs `mlp_oversampled` smooth transition.

**Caveats:** Label taxonomy is provisional.

## threshold morphology

**Working definition:** Local/global structure of metric change as threshold varies (where jumps occur, how concentrated they are).

**Use when:** Discussing elasticity localization and collapse intervals.

**Avoid using when:** Discussing calibration curves alone.

**Related terms:** threshold elasticity, max recall jump, operational smoothness.

**Example from current experiments:** `0.50->0.25` dominant jump for `mlp_bce`.

**Caveats:** Sensitive to chosen threshold grid.

## operational smoothness

**Working definition:** Degree to which accessibility changes gradually rather than abruptly across thresholds.

**Use when:** Comparing controllability/robustness of threshold tuning.

**Avoid using when:** Equating with good calibration.

**Related terms:** max recall jump, elasticity concentration.

**Example from current experiments:** `mlp_weighted` smoothness `0.6080` vs `mlp_bce` `0.4178`.

**Caveats:** One aggregate index; inspect interval-level dynamics too.

## threshold elasticity

**Working definition:** Sensitivity of recall/precision/F1 to threshold movement over intervals.

**Use when:** Quantifying threshold sensitivity concentration.

**Avoid using when:** Using only endpoint differences.

**Related terms:** operational smoothness, max recall jump.

**Example from current experiments:** `results/geometry_transition_analysis/elasticity_concentration.csv`.

**Caveats:** Interval granularity affects values.

## occupancy

**Working definition:** How score mass fills probability space across all examples and by class.

**Use when:** Discussing entropy/sparsity/persistence/compression.

**Avoid using when:** Using occupancy as synonym for calibration.

**Related terms:** occupancy persistence, support breadth.

**Example from current experiments:** `reports/*prediction_space_occupancy_summary.csv`.

**Caveats:** Needs class-conditional context.

## occupancy persistence

**Working definition:** Stability of minority occupancy/accessibility under threshold relaxation/tightening.

**Use when:** Assessing operational robustness.

**Avoid using when:** Interpreting only single-threshold occupancy.

**Related terms:** threshold occupancy persistence, reachability.

**Example from current experiments:** `mlp_weighted` persistence above `mlp_bce` in transition means.

**Caveats:** Can improve without broad support expansion.

## support breadth

**Working definition:** Effective width/diversity of posterior score support (entropy/effective support size).

**Use when:** Describing spread of score mass.

**Avoid using when:** Claiming smoothness from breadth alone.

**Related terms:** occupancy entropy, effective support size.

**Example from current experiments:** `mlp_weighted` broader support than `mlp_bce`.

**Caveats:** Breadth is not sufficient for smooth morphology.

## posterior compression

**Working definition:** Concentration of minority-relevant probability mass into narrow or low-score regions.

**Use when:** Interpreting conservative or collapse-prone behavior.

**Avoid using when:** Assuming all concentration is harmful.

**Related terms:** conservative allocator, low-score mass, occupancy compression.

**Example from current experiments:** High low-score mass on severe datasets.

**Caveats:** Compression metrics can be unstable when denominator widths are near zero.

## conservative allocator

**Working definition:** Model class with high low-score concentration and limited minority accessibility at standard thresholds.

**Use when:** Regime synthesis indicates persistent low accessibility.

**Avoid using when:** Equating with poor ranking globally.

**Related terms:** cliff allocator, posterior compression.

**Example from current experiments:** LightGBM frequently categorized conservative in baseline analyses.

**Caveats:** Regime can shift with calibration or objective perturbation.

## cliff allocator

**Working definition:** Accessibility transition concentrated in narrow threshold intervals; abrupt recall jumps under relaxation.

**Use when:** Max jump/elasticity concentration is high.

**Avoid using when:** Jumps are distributed and smoothness high.

**Related terms:** threshold morphology, elasticity concentration.

**Example from current experiments:** `mlp_bce` cliff with `max_recall_jump=0.4061`.

**Caveats:** Classification depends on heuristic summary pipeline.

## smooth allocator

**Working definition:** Accessibility change is distributed over thresholds with lower peak jumps and lower concentrated elasticity.

**Use when:** Smoothness increases and jump concentration declines.

**Avoid using when:** Only calibration improved, but operational jump profile worsened.

**Related terms:** operational smoothness, reachability persistence.

**Example from current experiments:** `mlp_oversampled`, `mlp_weighted` raw regime labels.

**Caveats:** Smooth does not imply broad support or best ranking.

## broad allocator

**Working definition:** Regime with relatively broad support and persistent accessibility across threshold bands.

**Use when:** Support and persistence jointly elevated.

**Avoid using when:** Behavior is smooth but still highly concentrated in low-score mass.

**Related terms:** support breadth, occupancy persistence.

**Example from current experiments:** Bagged HDDT/HDDT references in regime summaries.

**Caveats:** Broad and smooth are related but not identical.

## quantized allocator

**Working definition:** Score support concentrated on discrete mass points causing threshold-insensitive plateaus.

**Use when:** Threshold sweeps show minimal change over many cutoffs.

**Avoid using when:** Continuous support with gradual transitions.

**Related terms:** CART-like discrete probabilities.

**Example from current experiments:** CART threshold-invariant behavior.

**Caveats:** May depend on tree depth/probability estimation details.

## calibration-morphology interaction

**Working definition:** Joint behavior where calibration improves reliability metrics yet can alter or worsen operational morphology.

**Use when:** Comparing raw vs calibrated regime/smoothness outcomes.

**Avoid using when:** Reporting ECE/Brier alone.

**Related terms:** calibration vs smoothness non-equivalence.

**Example from current experiments:** `mlp_oversampled` and `mlp_weighted` raw smooth -> calibrated cliff.

**Caveats:** Method-dependent (Platt vs isotonic) and split-size sensitive.

## morphology transition

**Working definition:** Regime movement under controlled perturbation (e.g., cliff -> smooth) with architecture fixed.

**Use when:** Interpreting causal hints from objective/sampling changes.

**Avoid using when:** Architecture changes simultaneously.

**Related terms:** objective perturbation, reachability evolution.

**Example from current experiments:** `mlp_bce` to `mlp_oversampled`/`mlp_weighted`.

**Caveats:** Indicates sensitivity, not full causality proof.

## operational persistence

**Working definition:** Ability to retain minority accessibility across stricter threshold settings.

**Use when:** Discussing deployment robustness to threshold policy changes.

**Avoid using when:** Confusing with static performance at one threshold.

**Related terms:** reachability persistence, occupancy persistence.

**Example from current experiments:** Improved persistence in smooth MLP variants.

**Caveats:** Dataset geometry can dominate persistence limits.

## threshold-mediated deployment

**Working definition:** Operational setting where actionability depends on score-threshold rules.

**Use when:** Motivating why trajectory metrics matter.

**Avoid using when:** Pure ranking-only offline use cases.

**Related terms:** operational accessibility, queue sensitivity.

**Example from current experiments:** fixed threshold set `0.50, 0.25, 0.10, 0.05, 0.01`.

**Caveats:** Real systems may use adaptive or cost-based thresholds.

## empirical + conceptual framing paper

**Working definition:** Paper type that combines reproducible observations with a structured conceptual vocabulary/framework, without claiming complete theory.

**Use when:** Positioning this manuscript.

**Avoid using when:** Presenting algorithmic novelty or formal theorem claims.

**Related terms:** framing paper, provisional taxonomy.

**Example from current experiments:** Regime taxonomy + transition evidence + calibration interaction.

**Caveats:** Must keep claims calibrated and explicitly bounded.

---

## Explicit Distinctions (for reviewer clarity)

- **reachability vs recall:** recall is a point on `R(t)`; reachability is a threshold trajectory.
- **calibration vs operational smoothness:** ECE/Brier can improve while jump intensity worsens.
- **support breadth vs occupancy persistence:** broader support can help but does not guarantee persistence.
- **ranking quality vs operational accessibility:** AUROC/AP do not guarantee usable default-threshold access.
- **broad allocation vs smooth allocation:** broad concerns support/persistence extent; smooth concerns transition morphology.
- **allocation geometry vs accessibility morphology:** geometry is full score-structure umbrella; morphology is threshold-evolution shape layer.
