# Bagged HDDT Experiment C

## Q1 — Is high recovery stable under bootstrapping?
- Yes, directionally robust under current resampling. `hddt_forest-hddt` recovery delta mean 0.3417 (95% CI [0.2210, 0.4480]).
- Versus LightGBM, delta mean 0.3553 (95% CI [0.1260, 0.6034]).

## Q2 — Is high max-jump behavior stable?
- Yes, stable under current resampling. `hddt_forest-hddt` jump delta mean 0.3418 (95% CI [0.2252, 0.5116]).
- Versus LightGBM, jump delta mean 0.3894 (95% CI [0.2689, 0.4948]).

## Q3 — Is low smoothness stable?
- Yes versus HDDT and LightGBM: smoothness deltas are negative (`hddt_forest-hddt`: -0.0615, `hddt_forest-lightgbm`: -0.0888) under bootstrap means.
- This should be described as directionally robust, with small-sample caution (5 datasets).

## Q4 — Is persistence meaningfully higher than HDDT or LightGBM?
- Versus LightGBM: yes directionally (`0.0985` mean delta).
- Versus HDDT: weak-to-moderate uplift (`0.0388` mean delta), treated as qualified rather than headline evidence.

## Q5 — Is the phenomenon dominated by one dataset?
- Leave-one-dataset-out deltas preserve the key signs for recovery and max jump in primary comparisons, indicating no single-dataset domination for those effects.
- Persistence and smoothness magnitudes vary across drops; these are dataset-sensitive in effect size even when direction is often stable.

## Q6 — What can be safely claimed in the manuscript?
- Safe Claim: Bagged HDDT shows elevated recovery and larger recall jumps than HDDT and LightGBM, stable under current dataset resampling.
- Qualified Claim: Bagged HDDT exhibits lower smoothness and somewhat higher threshold-survival persistence; magnitude is dataset-sensitive and should be framed conservatively.
- Unsafe Claim: Strong mechanistic assertions about vote-level causes without per-instance ensemble decomposition.
