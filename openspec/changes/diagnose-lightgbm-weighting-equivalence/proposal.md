# Diagnose LightGBM Weighting Equivalence

## Summary

Add diagnostics to determine whether similar behavior between `lightgbm_unbalanced` and `lightgbm_weighted` is expected or due to parameter resolution bugs.

## Motivation

Near-identical benchmark summaries for both variants can be legitimate under some splits, but need explicit confirmation that parameter plumbing is correct.

## Scope

- Add diagnostics utility and CLI.
- Capture resolved parameters and split-level ratio values.
- Compare probability outputs across LightGBM variants.
- Emit markdown and optional JSON diagnostic artifacts.
- Add tests for parameter resolution and comparison logic.

## Non-Goals

- Do not remove either LightGBM variant.
- Do not alter behavior unless diagnostics reveal a bug.
