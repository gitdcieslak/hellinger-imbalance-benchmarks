# Strategic Narrative — HDDT Imbalance Study

## Original Goal

The original intent of this project was relatively straightforward:

- reimplement Hellinger Distance Decision Trees (HDDT),
- modernize the original benchmarking environment,
- compare HDDT against modern ensemble methods such as:
  - Random Forest,
  - XGBoost,
  - LightGBM,
- evaluate whether the original HDDT imbalance claims still hold under modern baselines.

The initial framing was essentially:

> “Does HDDT still matter?”

---

# Evolution of the Project

As the benchmark framework matured, the project evolved significantly.

The work expanded from:
- simple classifier comparison,
- toward studying the operational behavior of probabilistic classifiers under severe class imbalance.

Several infrastructure additions enabled this transition:

- repeated 5x2 stratified evaluation,
- threshold sweep analysis,
- score distribution analysis,
- separation metrics,
- score visualization,
- synthetic imbalance generation,
- legacy HDDT dataset ingestion,
- curated benchmark registry,
- operational reporting.

The project increasingly shifted away from:
- “who wins AUROC?”,
and toward:
- “how do classifiers allocate posterior probability mass under imbalance?”

---

# Current Core Hypothesis

A major emerging hypothesis is:

> Ranking quality and operational probability allocation behavior are meaningfully different properties under severe class imbalance.

In particular:
- modern ensemble methods often achieve strong AUROC and Average Precision,
- while simultaneously allocating extremely conservative posterior probabilities to minority examples,
- leading to operational collapse at default deployment thresholds (e.g. 0.50).

This phenomenon appears repeatedly across:
- synthetic imbalance experiments,
- and real legacy HDDT datasets.

---

# Emerging Observations

## 1. Threshold Collapse

On several severely imbalanced datasets:
- `boundary`
- `cam`
- `compustat`
- `oil`

models such as:
- Random Forest,
- XGBoost,
- LightGBM

often produce:
- high AUROC,
- strong ranking behavior,
- but near-zero recall at threshold 0.50.

However:
- lowering thresholds to 0.01 or 0.05 frequently recovers substantial recall.

This suggests:
- the models know where positives are,
- but allocate insufficient posterior mass at operational thresholds.

This behavior has been referred to internally as:

> “threshold collapse”

---

## 2. CART Threshold Invariance

CART models behave differently.

Observed behavior:
- threshold sweeps often produce nearly identical metrics across thresholds,
- suggesting highly discrete posterior allocation,
- with leaf probabilities behaving almost like hard partitions.

This contrasts sharply with:
- RF,
- XGB,
- LightGBM.

Emerging interpretation:
- CART may represent a fundamentally different score geometry regime.

---

## 3. HDDT Operational Distinctiveness

HDDT often exhibits:
- lower AUROC than modern boosted ensembles,
- but materially stronger recall/F1 at default thresholds under severe imbalance.

This suggests HDDT may:
- allocate minority posterior mass less conservatively,
- resulting in operationally different deployment behavior.

Bagged HDDT ensembles have proven surprisingly competitive on several datasets.

---

## 4. Three Emerging Allocation Regimes

An emerging conceptual framing is:

| Regime | Characteristics | Example Models |
|---|---|---|
| Conservative posterior allocators | Excellent ranking, low default recall under imbalance | RF, XGB, LightGBM |
| Moderate allocators | Better minority allocation at default thresholds | HDDT |
| Discrete allocators | Threshold-invariant leaf behavior | CART |

This framing is still preliminary but increasingly supported empirically.

---

# Important Datasets

The following datasets appear especially informative:

| Dataset | Importance |
|---|---|
| boundary | Extreme threshold collapse |
| cam | Severe imbalance operational behavior |
| compustat | Strong divergence between AUROC and recall |
| oil | Strong synthetic-to-real continuity |
| satimage | “healthy imbalance” comparison case |
| sick | Easier operational regime |

---

# Important Distinction

The project now appears less concerned with:

> “Which classifier is best?”

and more concerned with:

> “How do classifiers allocate probability mass under severe imbalance, and what operational consequences follow?”

This distinction increasingly appears central.

---

# Immediate Next Directions

## Near-term

- combined cross-dataset reporting,
- threshold-response visualization,
- precision/recall vs threshold curves,
- cross-model operational comparisons,
- benchmark snapshot preservation,
- paper outline development.

## Possible later directions

- calibration analysis,
- prediction landscape analysis,
- OpenML expansion,
- multiclass imbalance,
- neural tabular models,
- probability calibration correction methods.

---

# Current Strategic Recommendation

Avoid turning the project into:
- a generic benchmark zoo,
- or a hyperparameter sweep framework.

The strongest contribution currently appears to be:

> the distinction between ranking quality and posterior allocation behavior under severe class imbalance.

That conceptual thread should remain central.