# Related Work (Final)

## 2.1 Severe Class Imbalance Learning

Severe class imbalance has long been studied through cost-sensitive learning, sampling/reweighting strategies, and imbalance-aware tree criteria. Foundational references include cost-sensitive classification (Elkan, 2001), synthetic minority over-sampling (Chawla et al., 2002), and broad imbalance surveys (He and Garcia, 2009; Krawczyk, 2016). Class weighting and reweighting are now standard tools in both classical and neural pipelines (Buda et al., 2018; Johnson and Khoshgoftaar, 2019).

Our setup is explicitly within this tradition: we use weighting, oversampling, and fixed-threshold evaluation under severe skew. For Hellinger-based trees, the HDDT line (Cieslak and Chawla, 2008) motivates inclusion of HDDT-family baselines. We do not propose a new learner or a new HDDT variant. The contribution is evaluative: how minority accessibility changes as threshold policy moves.

## 2.2 Evaluation Under Severe Imbalance

Imbalance evaluation is commonly anchored by ROC/AUROC and PR/AP analyses (Davis and Goadrich, 2006; Saito and Rehmsmeier, 2015), often accompanied by threshold-specific confusion-matrix metrics. Operating-point analysis is also established in decision-oriented and cost-curve traditions (Fawcett, 2006; Drummond and Holte, 2006).

This paper does not argue against these tools. The narrower claim is that, in threshold-mediated severe-imbalance deployment, ranking summaries can underdescribe minority accessibility trajectories. In our evidence, near-neighbor ranking models can still differ materially in collapse/recovery behavior, jump intensity, and smoothness.

## 2.3 Selective Prediction, Coverage, and Threshold Analysis

This is the closest overlap area. Reject-option and selective classification study confidence-thresholded action sets and risk-coverage behavior (Chow, 1970; El-Yaniv and Wiener, 2010; Geifman and El-Yaniv, 2017).

Reachability in this manuscript is

\[
R(t) = P(\hat p(x) \ge t \mid y=1).
\]

Relation to class-conditional coverage is direct: reachability is a minority-conditioned class-conditional coverage trajectory over thresholds. We therefore do not claim a new mathematical object. The manuscript-specific difference is operational focus under severe imbalance: we pair this trajectory with persistence, smoothness, and elasticity-localization diagnostics to assess threshold-policy controllability.

In short, overlap is acknowledged at the object level, while contribution is an operational reinterpretation and empirical synthesis in severe-imbalance deployment conditions.

## 2.4 Calibration and Decision-Focused Probability Estimation

Calibration methods and diagnostics are well established, including Platt scaling (Platt, 1999), isotonic calibration (Zadrozny and Elkan, 2002), and reliability-oriented metrics such as ECE/Brier (Niculescu-Mizil and Caruana, 2005; Guo et al., 2017).

This paper does not challenge calibration and does not propose a calibration method. It studies calibration-accessibility interaction under severe imbalance. The observed pattern is bounded and empirical: reliability improvements can coexist with less smooth accessibility trajectories in this setting.

## 2.5 Decision-Theoretic and Operational ML Perspectives

Decision-theoretic classification emphasizes threshold choice under asymmetric cost and utility assumptions (Elkan, 2001; Fawcett, 2006). Our viewpoint is complementary: instead of optimizing one operating point only, we analyze accessibility behavior as threshold policy moves.

This aligns with operational ML concerns in triage and review systems, where threshold changes are frequent due to workload and policy constraints. In that context, trajectory shape (e.g., cliff-like vs smoother behavior) is a deployment-relevant controllability signal, not merely a post-hoc visualization.

## 2.6 Positioning and Boundaries

The manuscript is best read as an empirical and conceptual framing contribution at the intersection of severe-imbalance evaluation, selective-prediction-adjacent trajectory analysis, calibration-aware interpretation, and deployment behavior.

It is not a new learner, not a complete operational morphology theory, and not a universal taxonomy claim. The strongest novelty emphasis is on the empirical synthesis: ranking/reliability summaries and threshold-mediated accessibility can be partially non-equivalent in severe-imbalance settings, with stable directional evidence in key family-level contrasts under current resampling.
