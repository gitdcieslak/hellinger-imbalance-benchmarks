# Add Prediction-Space Occupancy Analysis

## Summary

Add prediction-space occupancy metrics and visualizations to characterize posterior allocation geometry under severe imbalance.

## Motivation

The paper framing now centers on operational allocation geometry. Occupancy analysis provides direct evidence of score compression, sparsity, quantization, and threshold occupancy behavior.

## Scope

- add occupancy metrics module
- add legacy occupancy runner integration
- add occupancy reporting summaries
- add histogram/ECDF/threshold-occupancy/quantization visualizations
- add CLI and tests

## Non-Goals

- no new model families
- no benchmark split or metric replacement
- no representation-learning or embedding analysis
