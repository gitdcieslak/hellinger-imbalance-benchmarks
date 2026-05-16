# Add Threshold Sweep Analysis

## Summary

Add threshold sweep evaluation and reporting for synthetic benchmark outputs.

## Motivation

Some models can produce useful ranking metrics such as AUROC and average precision while predicting no positives under a default classification threshold. Threshold sweep analysis shows how precision, recall, F1, and balanced accuracy change across probability cutoffs.

## Scope

- Add threshold-aware metric utilities.
- Add synthetic threshold sweep runner output.
- Add JSONL threshold records.
- Add CSV and markdown threshold summaries.
- Add `scripts/run_threshold_sweep.py`.
- Add tests for metric correctness, sweep output, and summary aggregation.

## Non-Goals

- Do not add neural models.
- Do not add real datasets.
- Do not add plots.
