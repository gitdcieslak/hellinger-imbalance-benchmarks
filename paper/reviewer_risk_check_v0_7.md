# Reviewer Risk Check (v0.7)

| Risk | Status | Rationale |
| --- | --- | --- |
| Selective prediction overlap | Medium | Overlap is explicitly acknowledged and reachability is framed as a minority-conditioned reinterpretation, but some reviewers may still view object-level novelty as limited. |
| Calibration overlap | Low | Text now clearly states calibration remains valuable and that the contribution is interaction analysis, not anti-calibration or method novelty. |
| Operational ML positioning | Low | Previously unresolved operational placeholders are now backed by concrete references (technical debt, production readiness rubric, human-AI workflow guidance). |
| HDDT justification | Low | HDDT lineage is now explicit (2008 introduction + 2012 skew-insensitivity/robustness), and model-family inclusion rationale is directly stated in methods framing. |
| Mechanism overclaim | Medium | Main manuscript language is cautious, but Bagged HDDT effects are strong and can invite causal over-reading; guardrail wording should be preserved in rebuttal and oral discussion. |
| External validity | Medium | Stability under current resampling is now clearly supported, but dataset scope remains small (`n=5` datasets) and the manuscript appropriately calls for broader replication. |

## Summary

- Highest residual risks are **scope/novelty interpretation**, not missing evidence links.
- Positioning is now substantially hardened for external circulation: citation closure complete, HDDT lineage clarified, and mechanism claims bounded.
