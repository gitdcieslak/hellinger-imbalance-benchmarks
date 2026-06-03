"""Mediation-style objective/allocation/topology regression report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


TARGETS = ["minority_survival_auc", "minority_survival_cliffiness"]

ALLOCATION_FEATURES = [
    "positive_histogram_entropy",
    "positive_effective_score_bins",
    "positive_max_bin_mass",
    "positive_top_bin_mass",
    "positive_score_gini_or_concentration_index",
]

TOPOLOGY_METRICS = [
    "n_components",
    "giant_component_fraction",
    "component_entropy",
    "isolated_positive_fraction",
]

FAMILIES = [
    "objective_only",
    "allocation_only",
    "topology_only",
    "objective_allocation",
    "objective_topology",
    "objective_allocation_topology",
]


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(format(value, floatfmt) if pd.notna(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_run_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    successful = df[df["fit_failed"].astype(str).str.lower() != "true"].copy()
    keys = ["objective", "seed", "model_name"]
    run_level = successful.drop_duplicates(keys).copy()

    topology_aggs = successful.groupby(keys)[TOPOLOGY_METRICS].agg(["mean", "median", "min", "max"])
    topology_aggs.columns = [f"topology_{metric}_{agg}" for metric, agg in topology_aggs.columns]
    topology_aggs = topology_aggs.reset_index()

    run_level = run_level.merge(topology_aggs, on=keys, how="left")
    return run_level.reset_index(drop=True)


def feature_columns(run_level: pd.DataFrame) -> dict[str, list[str]]:
    objective_cols = sorted(
        col for col in pd.get_dummies(run_level["objective"], prefix="objective").columns
    )
    topology_cols = sorted(col for col in run_level.columns if col.startswith("topology_"))
    return {
        "objective_only": objective_cols,
        "allocation_only": ALLOCATION_FEATURES,
        "topology_only": topology_cols,
        "objective_allocation": objective_cols + ALLOCATION_FEATURES,
        "objective_topology": objective_cols + topology_cols,
        "objective_allocation_topology": objective_cols + ALLOCATION_FEATURES + topology_cols,
    }


def _design_matrix(run_level: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    objective = pd.get_dummies(run_level["objective"], prefix="objective", dtype=float)
    numeric = run_level.drop(columns=[col for col in ["objective"] if col in run_level])
    features = pd.concat([objective, numeric], axis=1)
    for column in columns:
        if column not in features:
            features[column] = 0.0
    return features[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _cv_predictions(X: pd.DataFrame, y: pd.Series, seeds: pd.Series) -> np.ndarray:
    predictions = np.full(y.shape[0], np.nan, dtype=float)
    unique_seeds = sorted(pd.unique(seeds))
    if len(unique_seeds) >= 2:
        splits = [(seeds != seed, seeds == seed) for seed in unique_seeds]
    else:
        n_splits = min(5, len(y))
        if n_splits < 2:
            return np.full(y.shape[0], float(np.mean(y)), dtype=float)
        splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X))

    for train_idx, test_idx in splits:
        if isinstance(train_idx, pd.Series):
            train_mask = train_idx.to_numpy(dtype=bool)
            test_mask = test_idx.to_numpy(dtype=bool)
        else:
            train_mask = np.zeros(len(y), dtype=bool)
            test_mask = np.zeros(len(y), dtype=bool)
            train_mask[train_idx] = True
            test_mask[test_idx] = True
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X.loc[train_mask], y.loc[train_mask])
        predictions[test_mask] = model.predict(X.loc[test_mask])
    return predictions


def evaluate_regression_families(run_level: pd.DataFrame) -> pd.DataFrame:
    columns_by_family = feature_columns(run_level)
    rows = []
    for target in TARGETS:
        y = pd.to_numeric(run_level[target], errors="coerce")
        for family in FAMILIES:
            X = _design_matrix(run_level, columns_by_family[family])
            valid = y.notna()
            X_valid = X.loc[valid].reset_index(drop=True)
            y_valid = y.loc[valid].reset_index(drop=True)
            seeds_valid = run_level.loc[valid, "seed"].reset_index(drop=True)
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(X_valid, y_valid)
            in_sample_pred = model.predict(X_valid)
            cv_pred = _cv_predictions(X_valid, y_valid, seeds_valid)
            rows.append(
                {
                    "target": target,
                    "family": family,
                    "n_features": int(X_valid.shape[1]),
                    "cv_r2": float(r2_score(y_valid, cv_pred)),
                    "in_sample_r2": float(r2_score(y_valid, in_sample_pred)),
                    "cv_mse": float(mean_squared_error(y_valid, cv_pred)),
                    "in_sample_mse": float(mean_squared_error(y_valid, in_sample_pred)),
                }
            )
    return pd.DataFrame(rows)


def mediation_conclusions(results: pd.DataFrame) -> list[str]:
    lines = []
    for target, target_df in results.groupby("target"):
        by_family = target_df.set_index("family")
        objective_r2 = float(by_family.loc["objective_only", "cv_r2"])
        allocation_r2 = float(by_family.loc["allocation_only", "cv_r2"])
        topology_r2 = float(by_family.loc["topology_only", "cv_r2"])
        objective_allocation_r2 = float(by_family.loc["objective_allocation", "cv_r2"])
        objective_topology_r2 = float(by_family.loc["objective_topology", "cv_r2"])
        full_r2 = float(by_family.loc["objective_allocation_topology", "cv_r2"])
        objective_mse = float(by_family.loc["objective_only", "cv_mse"])
        objective_allocation_mse = float(by_family.loc["objective_allocation", "cv_mse"])
        allocation_mse_reduction = objective_mse - objective_allocation_mse
        topology_after_allocation = full_r2 - objective_allocation_r2

        lines.append(f"### {target}")
        lines.append(
            f"- Does objective explain survival? Objective-only CV R2 is {objective_r2:.4f}."
        )
        lines.append(
            f"- Does allocation explain survival? Allocation-only CV R2 is {allocation_r2:.4f}, compared with topology-only CV R2 {topology_r2:.4f}."
        )
        lines.append(
            f"- Does allocation explain away much of objective? Adding allocation to objective changes CV R2 from {objective_r2:.4f} to {objective_allocation_r2:.4f} and reduces CV MSE by {allocation_mse_reduction:.6f}."
        )
        lines.append(
            f"- Does topology add anything after allocation? Full model CV R2 is {full_r2:.4f}; incremental topology-after-allocation CV R2 is {topology_after_allocation:.4f}. Objective+topology CV R2 is {objective_topology_r2:.4f}."
        )
        if allocation_r2 > topology_r2 and topology_after_allocation <= 0.02:
            lines.append("- Interpretation: allocation is the stronger mediator-like explanation; topology adds little after allocation.")
        elif topology_after_allocation > 0.02:
            lines.append("- Interpretation: topology contributes additional explanatory power after allocation.")
        else:
            lines.append("- Interpretation: evidence is mixed or weak under cross-validation.")
        lines.append("")
    return lines


def build_allocation_mediation_report(df: pd.DataFrame) -> str:
    run_level = build_run_level_dataset(df)
    results = evaluate_regression_families(run_level)
    lines = [
        "# Allocation Mediation-Style Regression Summary",
        "",
        f"Input rows: {len(df)}",
        f"Run-level rows: {len(run_level)}",
        "",
        "## Regression Family Performance",
        _markdown_table(results),
        "",
        "## Mediation-Style Answers",
        *mediation_conclusions(results),
    ]
    return "\n".join(lines) + "\n"


def write_report(input_path: Path, output_path: Path) -> Path:
    df = pd.read_csv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_allocation_mediation_report(df), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "topology" / "mlp_objectives_topology.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "topology" / "allocation_mediation_summary.md",
    )
    args = parser.parse_args()
    output = write_report(args.input, args.output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
