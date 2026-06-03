"""Summarize topology and score-allocation explanations for survival shape."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SURVIVAL_METRICS = [
    "minority_survival_auc",
    "persistence",
    "minority_survival_cliffiness",
    "minority_survival_max_drop",
    "minority_survival_total_variation",
    "minority_survival_effective_drop_count",
]

ALLOCATION_METRICS = [
    "positive_unique_score_ratio",
    "positive_quantization_score",
    "positive_histogram_entropy",
    "positive_effective_score_bins",
    "positive_max_bin_mass",
    "positive_top_bin_mass",
    "positive_score_iqr",
    "positive_score_q10_q90_width",
    "positive_score_gini_or_concentration_index",
]

TOPOLOGY_METRICS = [
    "n_components",
    "giant_component_fraction",
    "mean_component_size",
    "median_component_size",
    "component_entropy",
    "isolated_positive_fraction",
]

CONTRAST_METRICS = [
    "minority_survival_auc",
    "minority_survival_cliffiness",
    "positive_histogram_entropy",
    "positive_effective_score_bins",
    "positive_max_bin_mass",
    "positive_top_bin_mass",
]

OBJECTIVE_PAIRS = [
    ("weighted_bce", "oversampled_bce"),
    ("weighted_bce", "bce"),
    ("oversampled_bce", "bce"),
]


def _corr_table(df: pd.DataFrame, predictors: list[str], target: str) -> pd.DataFrame:
    rows = []
    for metric in predictors:
        values = df[[metric, target]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(values) < 2 or values[metric].nunique() < 2 or values[target].nunique() < 2:
            corr = np.nan
        else:
            corr = float(values[metric].corr(values[target]))
        rows.append({"metric": metric, "target": target, "correlation": corr, "abs_correlation": abs(corr) if pd.notna(corr) else np.nan})
    return pd.DataFrame(rows)


def _allocation_correlation_table(run_level: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            _corr_table(run_level, ALLOCATION_METRICS, "minority_survival_cliffiness"),
            _corr_table(run_level, ALLOCATION_METRICS, "minority_survival_auc"),
        ],
        ignore_index=True,
    )


def _topology_correlation_table(successful: pd.DataFrame) -> pd.DataFrame:
    topology_tables = []
    for mode, mode_df in successful.groupby("topology_mode"):
        group_col = "k" if mode == "knn" else "radius_quantile"
        for group_value, group_df in mode_df.groupby(group_col, dropna=False):
            for target in ("minority_survival_cliffiness", "minority_survival_auc"):
                table = _corr_table(group_df, TOPOLOGY_METRICS, target)
                table.insert(0, "topology_mode", mode)
                table.insert(1, group_col, group_value)
                topology_tables.append(table)
    return pd.concat(topology_tables, ignore_index=True) if topology_tables else pd.DataFrame()


def _max_abs_by_target(corrs: pd.DataFrame, family: str) -> pd.DataFrame:
    best = corrs.groupby("target", as_index=False)["abs_correlation"].max()
    return best.rename(columns={"abs_correlation": f"max_abs_{family}_correlation"})


def bootstrap_family_correlation_ci(
    run_level: pd.DataFrame,
    successful: pd.DataFrame,
    n_bootstrap: int = 100,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap max absolute allocation/topology correlations over objective/seed runs."""

    run_ids = run_level[["objective", "seed", "model_name"]].drop_duplicates().reset_index(drop=True)
    if run_ids.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    records = []
    for _ in range(int(n_bootstrap)):
        sampled = run_ids.iloc[rng.integers(0, len(run_ids), size=len(run_ids))].copy()
        sampled["bootstrap_id"] = np.arange(len(sampled))

        sample_run_level = sampled.merge(run_level, on=["objective", "seed", "model_name"], how="left")
        sample_topology = sampled.merge(successful, on=["objective", "seed", "model_name"], how="left")

        allocation_best = _max_abs_by_target(
            _allocation_correlation_table(sample_run_level),
            "allocation",
        )
        topology_best = _max_abs_by_target(
            _topology_correlation_table(sample_topology),
            "topology",
        )
        merged = allocation_best.merge(topology_best, on="target", how="outer")
        records.extend(merged.to_dict("records"))

    boot = pd.DataFrame(records)
    rows = []
    for target, target_df in boot.groupby("target"):
        row = {"target": target}
        for family in ("allocation", "topology"):
            col = f"max_abs_{family}_correlation"
            values = pd.to_numeric(target_df[col], errors="coerce").dropna()
            if values.empty:
                row[f"{family}_ci_low"] = np.nan
                row[f"{family}_ci_high"] = np.nan
            else:
                row[f"{family}_ci_low"] = float(np.quantile(values, 0.025))
                row[f"{family}_ci_high"] = float(np.quantile(values, 0.975))
        rows.append(row)
    ci = pd.DataFrame(rows)
    ci["stability_interpretation"] = ci.apply(_stability_label, axis=1)
    return ci


def _stability_label(row: pd.Series) -> str:
    if row["allocation_ci_low"] > row["topology_ci_high"]:
        return "stable_allocation_stronger"
    if row["topology_ci_low"] > row["allocation_ci_high"]:
        return "stable_topology_stronger"
    return "exploratory_overlapping_ci"


def objective_pair_contrasts(run_level: pd.DataFrame) -> pd.DataFrame:
    means = run_level.groupby("objective", as_index=True)[CONTRAST_METRICS].mean()
    rows = []
    for left, right in OBJECTIVE_PAIRS:
        if left not in means.index or right not in means.index:
            continue
        for metric in CONTRAST_METRICS:
            rows.append(
                {
                    "contrast": f"{left} - {right}",
                    "metric": metric,
                    "mean_difference": float(means.loc[left, metric] - means.loc[right, metric]),
                }
            )
    return pd.DataFrame(rows)


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


def build_topology_vs_allocation_report(df: pd.DataFrame, n_bootstrap: int = 100) -> str:
    successful = df[df["fit_failed"].astype(str).str.lower() != "true"].copy()
    run_level = successful.drop_duplicates(["objective", "seed", "model_name"])

    survival_means = run_level.groupby("objective", as_index=False)[SURVIVAL_METRICS].mean()
    allocation_means = run_level.groupby("objective", as_index=False)[ALLOCATION_METRICS].mean()

    allocation_corrs = _allocation_correlation_table(run_level)
    topology_corrs = _topology_correlation_table(successful)

    allocation_best = allocation_corrs.groupby("target", as_index=False)["abs_correlation"].max()
    topology_best = topology_corrs.groupby("target", as_index=False)["abs_correlation"].max()
    comparison = allocation_best.merge(topology_best, on="target", suffixes=("_allocation", "_topology"))
    comparison["stronger_family"] = np.where(
        comparison["abs_correlation_allocation"] >= comparison["abs_correlation_topology"],
        "allocation",
        "topology",
    )

    top_allocation = allocation_corrs.sort_values("abs_correlation", ascending=False).head(12)
    top_topology = topology_corrs.sort_values("abs_correlation", ascending=False).head(16)
    bootstrap_ci = bootstrap_family_correlation_ci(run_level, successful, n_bootstrap=n_bootstrap)
    contrasts = objective_pair_contrasts(run_level)

    lines = [
        "# Topology vs Score Allocation Summary",
        "",
        f"Input rows: {len(df)}",
        f"Successful rows: {len(successful)}",
        f"Run-level rows for allocation/survival summaries: {len(run_level)}",
        "",
        "## Objective-Level Survival Means",
        _markdown_table(survival_means),
        "",
        "## Objective-Level Score Allocation Means",
        _markdown_table(allocation_means),
        "",
        "## Stronger Explanatory Family",
        _markdown_table(comparison),
        "",
        "## Bootstrap Confidence Intervals",
        _markdown_table(bootstrap_ci),
        "",
        "## Objective-Pair Contrasts",
        _markdown_table(contrasts),
        "",
        "## Run-Level Allocation Correlations With Survival Shape",
        _markdown_table(allocation_corrs.sort_values(["target", "abs_correlation"], ascending=[True, False])),
        "",
        "## Run-Level Topology Correlations By Mode And Scale",
        _markdown_table(topology_corrs.sort_values(["target", "topology_mode", "abs_correlation"], ascending=[True, True, False])),
        "",
        "## Allocation Correlations With Survival Shape",
        _markdown_table(top_allocation),
        "",
        "## Topology Correlations With Survival Shape",
        _markdown_table(top_topology),
        "",
        "## Interpretation",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"- For `{row.target}`, the stronger observed family is `{row.stronger_family}` "
            f"(allocation max |r|={row.abs_correlation_allocation:.4f}, "
            f"topology max |r|={row.abs_correlation_topology:.4f})."
        )
    for row in bootstrap_ci.itertuples(index=False):
        if row.stability_interpretation == "stable_allocation_stronger":
            lines.append(
                f"- For `{row.target}`, allocation is stable: its bootstrap CI "
                f"[{row.allocation_ci_low:.4f}, {row.allocation_ci_high:.4f}] is mostly above topology "
                f"[{row.topology_ci_low:.4f}, {row.topology_ci_high:.4f}]."
            )
        elif row.stability_interpretation == "stable_topology_stronger":
            lines.append(
                f"- For `{row.target}`, topology is stable: its bootstrap CI "
                f"[{row.topology_ci_low:.4f}, {row.topology_ci_high:.4f}] is mostly above allocation "
                f"[{row.allocation_ci_low:.4f}, {row.allocation_ci_high:.4f}]."
            )
        else:
            lines.append(
                f"- For `{row.target}`, the explanation is exploratory: allocation CI "
                f"[{row.allocation_ci_low:.4f}, {row.allocation_ci_high:.4f}] and topology CI "
                f"[{row.topology_ci_low:.4f}, {row.topology_ci_high:.4f}] overlap heavily."
            )
    return "\n".join(lines) + "\n"


def write_report(input_path: Path, output_path: Path, n_bootstrap: int = 100) -> Path:
    df = pd.read_csv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_topology_vs_allocation_report(df, n_bootstrap=n_bootstrap),
        encoding="utf-8",
    )
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
        default=ROOT / "reports" / "topology" / "topology_vs_allocation_summary.md",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    args = parser.parse_args()
    output = write_report(args.input, args.output, n_bootstrap=args.bootstrap_samples)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
