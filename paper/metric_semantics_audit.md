# Metric Semantics Audit

## Purpose

This audit checks whether the operational metrics used in the manuscript behave intuitively, match their names, and support the paper narrative under likely reviewer interpretation.

Primary finding: most metrics are valid, but one naming/semantic mismatch is currently high-risk:

- **"Accessibility Persistence" in the family table is currently populated with `mean_fraction_below_0_01`**, which is a low-score-mass metric, not a threshold-survival persistence metric.

---

## 1) Reachability

### Formal Definition

- Manuscript definition: `R(t)=P(\hat p(x) >= t | y=1)`.
- Implemented in occupancy/reachability routines as minority survival above threshold.

### Intended Interpretation

- Fraction of minority instances operationally accessible at threshold `t`.

### Observed Behavior

- Behaves as expected in trajectory plots; monotone non-increasing with threshold tightening.

### Potential Reader Interpretation

- Usually interpreted correctly as class-conditional coverage/recall trajectory.

### Naming Risks

- Moderate overlap risk with selective prediction terminology; low semantic risk.

### Recommended Action

- **Keep** (with existing overlap guardrail language).

---

## 2) Accessibility Persistence

### Formal Definition

- **Current manuscript table usage:** `mean_fraction_below_0_01` from `allocation_regime_summary.csv`.
- This equals proportion of predicted scores below 0.01 (score-mass concentration), not persistence across thresholds.

### Intended Interpretation

- Implied manuscript meaning: how well minority accessibility persists as threshold changes.

### Observed Behavior

- Produces counterintuitive values for persistence claims:
  - CART appears highly "persistent" (`0.9429`) due to extreme low-score mass pattern.
  - Bagged HDDT appears weakly "persistent" (`0.1252`) despite broad low-threshold recall.

### Potential Reader Interpretation

- Readers will infer this is survival-like persistence; current values contradict that intuition.

### Naming Risks

- **High risk**: semantic mismatch likely to trigger reviewer confusion or criticism.

### Recommended Action

- **Redefine** primary persistence metric to `threshold_occupancy_persistence` (already computed in occupancy artifacts).
- If `mean_fraction_below_0_01` is retained, **rename** to `Low-Score Mass (<0.01)` and treat as secondary support/compression signal.

---

## 3) Operational Smoothness

### Formal Definition

- From `elasticity.py`: `operational_smoothness_index = 1 / (1 + mean_abs_recall_elasticity + mean_abs_precision_elasticity)`.

### Intended Interpretation

- Higher values indicate less abrupt threshold-response changes.

### Observed Behavior

- Tracks cliff-to-smooth transitions sensibly in MLP perturbation and family comparisons.

### Potential Reader Interpretation

- Interpreted as composite stability proxy; generally intuitive.

### Naming Risks

- Low-to-medium (composite index may seem opaque without formula).

### Recommended Action

- **Keep** (show formula once; remind readers it is an index, not physical smoothness).

---

## 4) Max Recall Jump

### Formal Definition

- Maximum interval recall delta across threshold steps (`max(recall_end - recall_start)`).

### Intended Interpretation

- Magnitude of worst abrupt accessibility step under threshold relaxation.

### Observed Behavior

- Strong discriminator for cliff-like behavior; aligns with trajectory visuals.

### Potential Reader Interpretation

- Intuitive "largest cliff" measure.

### Naming Risks

- Low.

### Recommended Action

- **Keep**.

---

## 5) Recovery

### Formal Definition

- `Recall@0.01 - Recall@0.50`.

### Intended Interpretation

- Net accessibility regained by threshold relaxation.

### Observed Behavior

- Useful collapse/recovery anchor; does not encode path shape (can hide abruptness).

### Potential Reader Interpretation

- Straightforward and easy to audit.

### Naming Risks

- Low.

### Recommended Action

- **Keep** as primary scalar anchor, paired with trajectory/jump metrics.

---

## 6) Support Breadth

### Formal Definition

- Usually represented by `effective_support_size = exp(histogram_entropy)` over score histogram bins.

### Intended Interpretation

- Breadth/diversity of occupied score support.

### Observed Behavior

- Informative but not sufficient; manuscript already notes smoothness is not reducible to support breadth alone.

### Potential Reader Interpretation

- May be mistaken for direct controllability/stability metric.

### Naming Risks

- Medium (could be overinterpreted as causal driver).

### Recommended Action

- **Keep as secondary**; avoid headline emphasis.

---

# Special Focus: Why CART Looks Highly Persistent While Bagged HDDT Looks Weakly Persistent

## Diagnosis

This is primarily a **semantic mismatch**, not necessarily a computation bug.

- The current "Accessibility Persistence" column uses `mean_fraction_below_0_01`.
- That metric measures low-score mass concentration, not minority survival persistence across thresholds.

So the surprising pattern is expected under this definition:

- CART can score very high on low-score mass metric.
- Bagged HDDT can score low on low-score mass metric while still showing broad low-threshold accessibility.

## Conclusion

- Metric computation appears internally consistent.
- Interpretation under the label "Accessibility Persistence" is misleading.
- The name should change or the metric should be replaced for primary persistence claims.

---

# Recommended Primary Metrics

1. Reachability trajectory `R(t)` (figure-first).
2. Recovery (`Recall@0.01 - Recall@0.50`).
3. Operational Smoothness Index.
4. Max Recall Jump.
5. **Threshold Occupancy Persistence** (replace current mislabeled persistence field when persistence language is used).

# Recommended Secondary Metrics

1. Effective Support Size (support breadth).
2. Histogram entropy.
3. Low-score mass (`fraction_scores_below_0_01`) as compression indicator.

# Recommended Terminology Changes

1. Replace current table label `Accessibility Persistence` with:
   - `Low-Score Mass (<0.01)` if value remains `mean_fraction_below_0_01`, **or**
   - `Threshold Occupancy Persistence` if swapped to true persistence metric.
2. Use `Operational Smoothness Index` consistently (include formula once).
3. Keep `Reachability` with explicit note: closely related to class-conditional coverage trajectory.

# Metrics To Avoid Highlighting

1. `mean_fraction_below_0_01` under persistence framing (high confusion risk).
2. Compression ratio as standalone evidence (known instability in edge cases).
3. Support breadth as primary driver claim.

---

# Actionable Manuscript Impact

Minimal high-value correction:

1. Update `paper/manuscript_tables/table_allocator_family_summary.md` semantics:
   - either relabel current persistence column to `Low-Score Mass (<0.01)`;
   - or replace it with true `threshold_occupancy_persistence` values.
2. In narrative text, reserve "persistence" for threshold-survival metrics, not score-mass concentration metrics.

This change will substantially improve reviewer trust in metric semantics without altering core claims.
