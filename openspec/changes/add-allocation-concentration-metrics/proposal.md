# Add Allocation Concentration Metrics

## Summary

Add compact score-allocation concentration metrics to quantify how models allocate positive-class probability mass across dataset/model/split pairs.

## Motivation

Threshold-response and trajectory analyses suggest that model families differ in operational allocation geometry under severe imbalance. Concentration metrics provide concise, comparable descriptors of that behavior.

## Scope

- Add allocation metrics module.
- Add runner integration for legacy and synthetic sources.
- Add reporting summaries (CSV + Markdown).
- Add CLI for execution.
- Add tests for metric definitions and record/summary behavior.

## Non-Goals

- No calibration.
- No threshold tuning.
- No model additions or benchmark-metric changes.
