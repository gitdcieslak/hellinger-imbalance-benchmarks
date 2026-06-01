# Related Work Insertion Plan

## Goal

Replace placeholder-heavy Related Work text in `paper/manuscript_v0_5.md` Section 2 with a citation-complete, overlap-explicit, claim-safe version from `paper/related_work_complete.md`.

## Scope of Edits

Target section in manuscript:

- `paper/manuscript_v0_5.md` lines under `## 2. Related Work` through `### 2.6 Summary Positioning`.

No changes required to methods/results claims in this pass.

## Section-by-Section Replacement

### 2.1 Severe Class Imbalance Learning

Replace placeholders:

- `[CITATION: severe imbalance survey]`
- `[CITATION: cost-sensitive learning foundation]`
- `[CITATION: oversampling foundation]`
- `[CITATION: HDDT original paper]`

With citation cluster:

- Elkan (2001), Chawla et al. (2002), He & Garcia (2009), Krawczyk (2016), Cieslak & Chawla (2008), plus optional modern imbalance survey entries.

### 2.2 Evaluation Under Severe Imbalance

Replace placeholders:

- `[CITATION: PR vs ROC under imbalance]`
- `[CITATION: threshold evaluation under class imbalance]`

With citation cluster:

- Davis & Goadrich (2006), Saito & Rehmsmeier (2015), Fawcett (2006), Drummond & Holte (2006).

### 2.3 Selective Prediction, Coverage, and Threshold Analysis

Replace placeholders:

- `[CITATION: reject-option foundation]`
- `[CITATION: selective prediction foundation]`
- `[CITATION: risk-coverage analysis]`

With citation cluster:

- Chow (1970), El-Yaniv & Wiener (2010), Geifman & El-Yaniv (2017), and one risk-coverage reference used in your bibliography style.

Mandatory wording guardrail to keep:

- "Reachability is closely related to class-conditional coverage trajectories".
- "Operational reinterpretation/specialization under severe imbalance".

### 2.4 Calibration and Decision-Focused Probability Estimation

Replace placeholders:

- `[CITATION: Platt scaling]`
- `[CITATION: isotonic calibration]`
- `[CITATION: calibration foundations]`
- `[CITATION: modern calibration analysis]`

With citation cluster:

- Platt (1999), Zadrozny & Elkan (2002), Niculescu-Mizil & Caruana (2005), Guo et al. (2017).

Mandatory guardrail:

- Keep explicit "calibration remains valuable" sentence.

### 2.5 Decision-Theoretic and Operational Perspectives

Replace placeholders:

- `[CITATION: cost-sensitive threshold theory]`
- `[CITATION: decision-theoretic classification]`
- `[CITATION: operational ML monitoring]`
- `[CITATION: human-in-the-loop triage]`
- `[CITATION: queue-aware ML deployment]`

With citation cluster:

- Elkan (2001), Fawcett (2006), and selected operational-ML/triage/deployment references already planned in your bib.

### 2.6 Summary Positioning

Keep concise and claim-safe:

- emphasize complementarity,
- no "first/unique" claims,
- repeat non-claims briefly.

## Novelty Boundary Integration

Use `paper/novelty_boundary_table.md` to ensure section wording stays conservative:

- Reachability: framed as reframing/integration.
- Stronger novelty emphasis: empirical interaction findings (family-level divergence, MLP transition, calibration interaction).

## Reviewer-Risk Checks Before Finalizing

1. **Selective prediction overlap check:**
   - Does the final text explicitly state overlap and difference in one paragraph?
2. **Calibration misread check:**
   - Is anti-calibration language absent?
3. **Overclaim check:**
   - Any use of "first," "novel object," "unprecedented" should be removed.
4. **Consistency check:**
   - Terms match v0.5 semantics (especially persistence definition).

## Minimal Citation Set to Unblock Submission-Ready Related Work

Required minimum:

- Chow (1970)
- Elkan (2001)
- Chawla et al. (2002)
- Davis & Goadrich (2006)
- Fawcett (2006)
- Cieslak & Chawla (2008)
- He & Garcia (2009)
- El-Yaniv & Wiener (2010)
- Saito & Rehmsmeier (2015)
- Krawczyk (2016)
- Geifman & El-Yaniv (2017)
- Guo et al. (2017)
- Niculescu-Mizil & Caruana (2005)
- Platt (1999)
- Zadrozny & Elkan (2002)

## Final Acceptance Test

A reviewer should be able to answer clearly:

1. "How is this different from selective prediction?"
   - Answer: object overlap acknowledged; contribution is severe-imbalance minority-conditioned operational reinterpretation plus integrated trajectory/jump/smoothness evidence.

2. "What is actually new here?"
   - Answer: empirical synthesis and interaction findings, not a new learner or foundational coverage theory.
