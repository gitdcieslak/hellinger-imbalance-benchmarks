# hellinger-imbalance-benchmarks

## Purpose

This repository contains reproducible benchmark infrastructure for studying class imbalance behavior across Hellinger Distance Decision Trees, modern tree ensembles, gradient boosting systems, and selected neural baselines.

The repo depends on the separate `hellinger-tree` package for the HDDT implementation.

## Research Goal

Evaluate whether the skew-insensitive behavior originally demonstrated for Hellinger Distance Decision Trees remains relevant when compared against modern methods such as CART, RandomForest, BalancedRandomForest, XGBoost, LightGBM, and neural tabular models.

## Scope

This project manages:

- dataset manifests
- synthetic skew experiments
- model baseline configuration
- experiment execution
- metrics computation
- result aggregation
- reproducibility metadata
- report-ready output tables and figures

This project does not implement the HDDT estimator itself.

## Principles

- Separate method implementation from benchmark evidence.
- Store configs and manifests, not large datasets.
- Make every run reproducible from code, config, package versions, and seeds.
- Prefer small smoke tests before large benchmark campaigns.
- Treat benchmark results as generated artifacts.
