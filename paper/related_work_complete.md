# Related Work (Complete Draft)

## 2.1 Severe Class Imbalance Learning

Severe class imbalance has long been studied through cost-sensitive learning, sampling/reweighting strategies, and imbalance-aware tree criteria. Foundational work includes cost-sensitive classification formulations (Elkan, 2001), synthetic minority over-sampling (Chawla et al., 2002), and broad imbalance surveys (He and Garcia, 2009; Krawczyk, 2016). Class-weighting and reweighting are now standard components in both classical and neural pipelines (Buda et al., 2018; Johnson and Khoshgoftaar, 2019).

Our setup is directly in this tradition: we use weighting, oversampling, and fixed-threshold evaluation under severe skew. We do not introduce a new learner. Our contribution is in *evaluation emphasis*: how minority accessibility evolves under threshold movement rather than endpoint ranking alone.

For Hellinger-based trees, the original Hellinger Distance Decision Tree line (Cieslak and Chawla, 2008) and follow-on imbalance studies motivate inclusion of HDDT-style families as strong rare-event baselines. In our manuscript, these families are used to expose family-level operational trajectory differences under a shared protocol, not to claim a new HDDT variant.

## 2.2 Evaluation Under Imbalance

Imbalance evaluation is commonly anchored by ROC/AUROC and PR/AP analyses (Davis and Goadrich, 2006; Saito and Rehmsmeier, 2015), often supplemented by threshold-specific confusion-matrix metrics. Operating-point analysis is well established in decision-oriented classification and cost-curve traditions (Drummond and Holte, 2006; Fawcett, 2006).

We agree with this literature that AUROC/AP are valuable discrimination summaries. Our claim is narrower: in threshold-mediated severe-imbalance deployment, these summaries can underdescribe minority accessibility behavior. Concretely, models with similar ranking quality can exhibit materially different default-threshold accessibility collapse/recovery and different jump/smoothness profiles under threshold policy changes.

## 2.3 Selective Prediction, Coverage, and Threshold Analysis

This is our closest overlap area. Reject-option and selective classification traditions study confidence-thresholded action sets and risk-coverage tradeoffs (Chow, 1970; El-Yaniv and Wiener, 2010; Geifman and El-Yaniv, 2017). Risk-coverage style evaluation is now standard in selective prediction.

Our reachability definition,

\[
R(t) = P(\hat p(x) \ge t \mid y=1),
\]

is closely related to class-conditional coverage trajectories. We therefore do **not** claim a wholly new mathematical object. The manuscript’s position is:

1. **Object-level overlap:** yes, reachability is adjacent to class-conditional coverage.
2. **Specialization:** minority-conditioned under severe imbalance.
3. **Operational reinterpretation:** emphasis on threshold-policy controllability via persistence, smoothness, and elasticity localization.

Submission-safe positioning sentence:

"Reachability is best interpreted as an operationally targeted, minority-conditioned reinterpretation of class-conditional threshold coverage, integrated here with threshold-sensitivity diagnostics for severe-imbalance deployment analysis."

## 2.4 Calibration and Decision-Focused Probability Estimation

Calibration methods and diagnostics are well established: Platt scaling (Platt, 1999), isotonic calibration (Zadrozny and Elkan, 2002), and reliability metrics such as ECE/Brier (Niculescu-Mizil and Caruana, 2005; Guo et al., 2017). We use these methods directly and maintain the standard view that calibration is valuable for probability reliability.

Our contribution is not methodological calibration novelty. Instead, we analyze calibration-accessibility interaction under severe imbalance and observe a bounded tension: reliability improvements can coexist with less smooth threshold-accessibility geometry. This is framed as a setting-specific empirical interaction, not an anti-calibration general claim.

## 2.5 Decision-Theoretic and Operational ML Perspectives

Decision-theoretic classification emphasizes threshold choice under asymmetric costs and utility models (Elkan, 2001; Fawcett, 2006). Operational ML and human-in-the-loop systems emphasize policy updates, workload constraints, and intervention pipelines in fraud, triage, and monitoring workflows.

Our manuscript is complementary: decision theory asks which threshold is optimal under a utility model; we emphasize how minority accessibility behaves as threshold policy moves in practice. This makes the strongest positioning in deployment-behavior terms: threshold as a control knob, and trajectory morphology as a controllability/stability diagnostic.

## 2.6 Novelty Boundary (Conservative)

What appears primarily novel/underemphasized in this combination:

- Family-level empirical demonstration that similar ranking summaries can mask materially different threshold-mediated accessibility behavior.
- Fixed-architecture MLP morphology transition evidence under imbalance-pressure perturbation.
- Bounded calibration-reliability vs accessibility-smoothness interaction framing in this severe-imbalance protocol.

What appears primarily reframing/integration:

- Reachability as central analytical lens (adjacent to class-conditional coverage).
- Recurring morphology language as provisional empirical shorthand.
- Operational accessibility synthesis across ranking, threshold trajectories, occupancy, and calibration interaction.

## 2.7 Reviewer-Facing Positioning Summary

- We build on imbalance learning and selective prediction-adjacent trajectory objects.
- We do not claim algorithmic novelty, complete theory, or universal taxonomy.
- We contribute a deployment-oriented empirical synthesis showing that ranking, reliability, and threshold-mediated accessibility can be partially non-equivalent under severe imbalance.

---

## References to Include in Bibliography (minimum set)

- Chow, C. K. (1970). On optimum recognition error and reject tradeoff.
- Elkan, C. (2001). The foundations of cost-sensitive learning.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer, W. P. (2002). SMOTE.
- Davis, J. and Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves.
- Fawcett, T. (2006). An introduction to ROC analysis.
- Drummond, C. and Holte, R. (2006). Cost curves.
- Cieslak, D. A. and Chawla, N. V. (2008). Learning decision trees for unbalanced data (HDDT).
- He, H. and Garcia, E. (2009). Learning from imbalanced data.
- El-Yaniv, R. and Wiener, Y. (2010). Selective classification.
- Saito, T. and Rehmsmeier, M. (2015). PR plots for imbalanced datasets.
- Krawczyk, B. (2016). Learning from imbalanced data: open challenges and future directions.
- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks.
- Buda, M., Maki, A., and Mazurowski, M. A. (2018). Class imbalance in CNNs.
- Johnson, J. M. and Khoshgoftaar, T. M. (2019). Survey on deep learning with class imbalance.
- Geifman, Y. and El-Yaniv, R. (2017). Selective classification for deep neural networks.
- Niculescu-Mizil, A. and Caruana, R. (2005). Predicting good probabilities.
- Platt, J. (1999). Probabilistic outputs for SVMs.
- Zadrozny, B. and Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates.
