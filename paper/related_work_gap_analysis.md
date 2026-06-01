# Related Work Gap Analysis

## Purpose

This note clarifies overlap and difference between the manuscript and adjacent literatures so reviewers can quickly see scope boundaries.

| Area | Overlap | Difference |
| --- | --- | --- |
| Severe imbalance learning | Uses standard imbalance tools: cost-sensitive framing, oversampling, weighting, and HDDT-family baselines. | Does not propose a new learner; focuses on threshold-mediated accessibility trajectories and policy sensitivity under severe skew. |
| Selective prediction / class-conditional coverage | Reachability uses the same mathematical family as class-conditional threshold coverage (`R(t)=P(\hat p(x)\ge t \mid y=1)`). | Reinterprets this object for minority accessibility control in severe imbalance and bundles it with persistence/smoothness/jump diagnostics for deployment interpretation. |
| Calibration literature | Adopts standard calibration methods and reliability metrics (Platt, isotonic, ECE/Brier). | Does not challenge calibration value; examines when reliability gains and accessibility smoothness diverge in this specific severe-imbalance protocol. |
| Operational ML | Shares concern with threshold policy, triage/review constraints, and deployment behavior under workload pressure. | Adds a trajectory-centric evaluation lens to characterize controllability as thresholds move, rather than reporting only fixed operating points. |
| Decision theory / operating-point analysis | Builds on cost/utility-aware threshold reasoning and ROC/operating-point traditions. | Extends emphasis from selecting one threshold to evaluating stability of accessibility geometry across threshold movement. |

## Practical Reading Guide

- What this paper is: an empirical + conceptual synthesis for threshold-mediated severe-imbalance deployment behavior.
- What this paper is not: new algorithm, formal universal theory, or replacement for calibration/ranking metrics.
- Why it belongs: it addresses a deployment-relevant gap where standard summaries can miss accessibility controllability differences between otherwise similar-ranking models.
