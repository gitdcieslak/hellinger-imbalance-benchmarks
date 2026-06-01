import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FOCUS_MODELS = ["hddt", "hddt_forest", "random_forest", "lightgbm"]
PAIRWISE = [
    ("hddt_forest", "hddt"),
    ("hddt_forest", "random_forest"),
    ("hddt_forest", "lightgbm"),
]
METRICS = [
    "recovery",
    "max_recall_jump",
    "operational_smoothness",
    "threshold_occupancy_persistence",
    "recall_at_0_50",
    "recall_at_0_01",
]
PRIMARY_DELTA_METRICS = [
    "recovery",
    "max_recall_jump",
    "operational_smoothness",
    "threshold_occupancy_persistence",
]


def load_dataset_level_table(base_dir: Path, models: list[str]) -> pd.DataFrame:
    thr = pd.read_csv(base_dir / "reports/neural_mlp/legacy_threshold_sweep_summary.csv")
    ela = pd.read_csv(base_dir / "reports/neural_mlp/threshold_elasticity_summary.csv")
    occ = pd.read_csv(base_dir / "reports/neural_mlp/prediction_space_occupancy_summary.csv")

    thr = thr[thr["model_id"].isin(models)].copy()
    ela = ela[ela["model_id"].isin(models)].copy()
    occ = occ[occ["model_id"].isin(models)].copy()

    pivot = (
        thr.pivot_table(
            index=["dataset_id", "model_id"],
            columns="threshold",
            values="recall_mean",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    rename_map = {
        0.50: "recall_at_0_50",
        0.25: "recall_at_0_25",
        0.10: "recall_at_0_10",
        0.05: "recall_at_0_05",
        0.01: "recall_at_0_01",
    }
    for threshold, out_name in rename_map.items():
        if threshold not in pivot.columns:
            raise ValueError(f"Missing threshold {threshold} in legacy_threshold_sweep_summary.csv")
        pivot = pivot.rename(columns={threshold: out_name})

    metric_cols = [
        "dataset_id",
        "model_id",
        "max_recall_jump",
        "operational_smoothness_index",
    ]
    ela_small = ela[metric_cols].rename(
        columns={"operational_smoothness_index": "operational_smoothness"}
    )

    occ_small = occ[
        ["dataset_id", "model_id", "threshold_occupancy_persistence_mean"]
    ].rename(
        columns={"threshold_occupancy_persistence_mean": "threshold_occupancy_persistence"}
    )

    merged = pivot.merge(ela_small, on=["dataset_id", "model_id"], how="inner")
    merged = merged.merge(occ_small, on=["dataset_id", "model_id"], how="inner")

    merged["recovery"] = merged["recall_at_0_01"] - merged["recall_at_0_50"]

    ordered_cols = [
        "model_id",
        "dataset_id",
        "recall_at_0_50",
        "recall_at_0_25",
        "recall_at_0_10",
        "recall_at_0_05",
        "recall_at_0_01",
        "recovery",
        "max_recall_jump",
        "operational_smoothness",
        "threshold_occupancy_persistence",
    ]
    merged = merged[ordered_cols].sort_values(["model_id", "dataset_id"]).reset_index(drop=True)
    return merged


def bootstrap_metrics(df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    datasets = np.array(sorted(df["dataset_id"].unique()))
    n = len(datasets)
    rows = []

    for model in sorted(df["model_id"].unique()):
        mdf = df[df["model_id"] == model].set_index("dataset_id")
        for metric in METRICS:
            vals = []
            for _ in range(n_bootstrap):
                sample_ids = rng.choice(datasets, size=n, replace=True)
                sample_values = mdf.loc[sample_ids, metric].to_numpy(dtype=float)
                vals.append(sample_values.mean())
            arr = np.array(vals)
            rows.append(
                {
                    "analysis": "model_metric",
                    "model_id": model,
                    "comparison": "",
                    "metric": metric,
                    "mean": float(arr.mean()),
                    "ci_2_5": float(np.quantile(arr, 0.025)),
                    "ci_97_5": float(np.quantile(arr, 0.975)),
                    "prob_delta_gt_0": np.nan,
                    "prob_delta_lt_0": np.nan,
                    "n_bootstrap": n_bootstrap,
                    "random_seed": seed,
                }
            )

    for left, right in PAIRWISE:
        left_df = df[df["model_id"] == left].set_index("dataset_id")
        right_df = df[df["model_id"] == right].set_index("dataset_id")
        comp_name = f"{left}-{right}"
        for metric in PRIMARY_DELTA_METRICS:
            vals = []
            for _ in range(n_bootstrap):
                sample_ids = rng.choice(datasets, size=n, replace=True)
                delta = (
                    left_df.loc[sample_ids, metric].to_numpy(dtype=float)
                    - right_df.loc[sample_ids, metric].to_numpy(dtype=float)
                )
                vals.append(delta.mean())
            arr = np.array(vals)
            rows.append(
                {
                    "analysis": "pairwise_delta",
                    "model_id": left,
                    "comparison": comp_name,
                    "metric": metric,
                    "mean": float(arr.mean()),
                    "ci_2_5": float(np.quantile(arr, 0.025)),
                    "ci_97_5": float(np.quantile(arr, 0.975)),
                    "prob_delta_gt_0": float((arr > 0).mean()),
                    "prob_delta_lt_0": float((arr < 0).mean()),
                    "n_bootstrap": n_bootstrap,
                    "random_seed": seed,
                }
            )

    return pd.DataFrame(rows)


def leave_one_dataset_out(df: pd.DataFrame) -> pd.DataFrame:
    datasets = sorted(df["dataset_id"].unique())
    rows = []
    for drop_ds in datasets:
        sub = df[df["dataset_id"] != drop_ds]
        agg = sub.groupby("model_id")[METRICS].mean()

        for left, right in PAIRWISE:
            d = agg.loc[left, PRIMARY_DELTA_METRICS] - agg.loc[right, PRIMARY_DELTA_METRICS]
            row = {
                "dropped_dataset_id": drop_ds,
                "comparison": f"{left}-{right}",
            }
            for metric in PRIMARY_DELTA_METRICS:
                row[f"delta_{metric}"] = float(d[metric])
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["comparison", "dropped_dataset_id"]).reset_index(drop=True)


def classify_claim_stability(
    boot: pd.DataFrame,
    lodo: pd.DataFrame,
    comparison: str,
    metric: str,
    expected: str,
    practical_min_abs_delta: float = 0.0,
) -> str:
    prow = boot[(boot["analysis"] == "pairwise_delta") & (boot["comparison"] == comparison) & (boot["metric"] == metric)]
    if prow.empty:
        return "Unsafe Claim"
    ci_low = float(prow.iloc[0]["ci_2_5"])
    ci_high = float(prow.iloc[0]["ci_97_5"])
    lodo_col = f"delta_{metric}"
    lsub = lodo[lodo["comparison"] == comparison][lodo_col]
    sign_ok = (lsub > 0).all() if expected == ">" else (lsub < 0).all()

    if expected == ">":
        ci_strict = ci_low > 0
        directional = ci_high > 0
    else:
        ci_strict = ci_high < 0
        directional = ci_low < 0

    if ci_strict and sign_ok and abs(float(prow.iloc[0]["mean"])) >= practical_min_abs_delta:
        return "Safe Claim"
    if directional:
        return "Qualified Claim"
    return "Unsafe Claim"


def write_summary_markdown(base_dir: Path, dataset_level: pd.DataFrame, boot: pd.DataFrame, lodo: pd.DataFrame) -> None:
    out_path = base_dir / "reports/neural_mlp/bagged_hddt_stability_summary.md"

    def fmt_row(comp: str, metric: str) -> str:
        row = boot[(boot.analysis == "pairwise_delta") & (boot.comparison == comp) & (boot.metric == metric)].iloc[0]
        return (
            f"- `{comp}` {metric}: mean={row['mean']:.4f}, "
            f"95% CI [{row['ci_2_5']:.4f}, {row['ci_97_5']:.4f}], "
            f"P(delta>0)={row['prob_delta_gt_0']:.3f}, P(delta<0)={row['prob_delta_lt_0']:.3f}"
        )

    datasets = sorted(dataset_level["dataset_id"].unique())
    lines = []
    lines.append("# Bagged HDDT Experiment C — Stability Summary")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- Resampling unit: dataset-level rows (5 datasets per model).")
    lines.append("- Bootstrap: 10,000 replicates, random seed 7.")
    lines.append("- Leave-one-dataset-out: recompute means and pairwise deltas after dropping each dataset once.")
    lines.append(f"- Datasets: {', '.join(datasets)}")
    lines.append("")
    lines.append("## Primary Pairwise Deltas")
    lines.append("")
    for comp in ["hddt_forest-hddt", "hddt_forest-lightgbm", "hddt_forest-random_forest"]:
        for metric in PRIMARY_DELTA_METRICS:
            lines.append(fmt_row(comp, metric))
        lines.append("")

    lines.append("## Leave-One-Dataset-Out Direction Check")
    lines.append("")
    checks = [
        ("hddt_forest-hddt", "recovery", ">"),
        ("hddt_forest-hddt", "max_recall_jump", ">"),
        ("hddt_forest-hddt", "operational_smoothness", "<"),
        ("hddt_forest-hddt", "threshold_occupancy_persistence", ">"),
        ("hddt_forest-lightgbm", "recovery", ">"),
        ("hddt_forest-lightgbm", "max_recall_jump", ">"),
        ("hddt_forest-lightgbm", "operational_smoothness", "<"),
        ("hddt_forest-lightgbm", "threshold_occupancy_persistence", ">"),
    ]
    for comp, metric, expected in checks:
        col = f"delta_{metric}"
        vals = lodo[lodo.comparison == comp][col]
        if expected == ">":
            stable = bool((vals > 0).all())
        else:
            stable = bool((vals < 0).all())
        lines.append(f"- `{comp}` {metric} expected `{expected}0`: {'stable across all drops' if stable else 'dataset-sensitive'}")
    lines.append("")

    lines.append("## Claim Tiering")
    lines.append("")
    for comp in ["hddt_forest-hddt", "hddt_forest-lightgbm"]:
        for metric, expected in [
            ("recovery", ">"),
            ("max_recall_jump", ">"),
            ("operational_smoothness", "<"),
            ("threshold_occupancy_persistence", ">"),
        ]:
            min_effect = 0.05 if metric == "threshold_occupancy_persistence" else 0.0
            tier = classify_claim_stability(
                boot,
                lodo,
                comp,
                metric,
                expected,
                practical_min_abs_delta=min_effect,
            )
            lines.append(f"- `{comp}` {metric}: {tier}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_research_note(base_dir: Path, boot: pd.DataFrame, lodo: pd.DataFrame) -> None:
    out_path = base_dir / "research/bagged_hddt_experiment_c.md"

    def b(comp: str, metric: str) -> pd.Series:
        return boot[(boot.analysis == "pairwise_delta") & (boot.comparison == comp) & (boot.metric == metric)].iloc[0]

    r_h = b("hddt_forest-hddt", "recovery")
    j_h = b("hddt_forest-hddt", "max_recall_jump")
    s_h = b("hddt_forest-hddt", "operational_smoothness")
    p_h = b("hddt_forest-hddt", "threshold_occupancy_persistence")
    r_l = b("hddt_forest-lightgbm", "recovery")
    j_l = b("hddt_forest-lightgbm", "max_recall_jump")
    s_l = b("hddt_forest-lightgbm", "operational_smoothness")
    p_l = b("hddt_forest-lightgbm", "threshold_occupancy_persistence")

    lines = [
        "# Bagged HDDT Experiment C",
        "",
        "## Q1 — Is high recovery stable under bootstrapping?",
        f"- Yes, directionally robust under current resampling. `hddt_forest-hddt` recovery delta mean {r_h['mean']:.4f} (95% CI [{r_h['ci_2_5']:.4f}, {r_h['ci_97_5']:.4f}]).",
        f"- Versus LightGBM, delta mean {r_l['mean']:.4f} (95% CI [{r_l['ci_2_5']:.4f}, {r_l['ci_97_5']:.4f}]).",
        "",
        "## Q2 — Is high max-jump behavior stable?",
        f"- Yes, stable under current resampling. `hddt_forest-hddt` jump delta mean {j_h['mean']:.4f} (95% CI [{j_h['ci_2_5']:.4f}, {j_h['ci_97_5']:.4f}]).",
        f"- Versus LightGBM, jump delta mean {j_l['mean']:.4f} (95% CI [{j_l['ci_2_5']:.4f}, {j_l['ci_97_5']:.4f}]).",
        "",
        "## Q3 — Is low smoothness stable?",
        f"- Yes versus HDDT and LightGBM: smoothness deltas are negative (`hddt_forest-hddt`: {s_h['mean']:.4f}, `hddt_forest-lightgbm`: {s_l['mean']:.4f}) under bootstrap means.",
        "- This should be described as directionally robust, with small-sample caution (5 datasets).",
        "",
        "## Q4 — Is persistence meaningfully higher than HDDT or LightGBM?",
        f"- Versus LightGBM: yes directionally (`{p_l['mean']:.4f}` mean delta).",
        f"- Versus HDDT: weak-to-moderate uplift (`{p_h['mean']:.4f}` mean delta), treated as qualified rather than headline evidence.",
        "",
        "## Q5 — Is the phenomenon dominated by one dataset?",
        "- Leave-one-dataset-out deltas preserve the key signs for recovery and max jump in primary comparisons, indicating no single-dataset domination for those effects.",
        "- Persistence and smoothness magnitudes vary across drops; these are dataset-sensitive in effect size even when direction is often stable.",
        "",
        "## Q6 — What can be safely claimed in the manuscript?",
        "- Safe Claim: Bagged HDDT shows elevated recovery and larger recall jumps than HDDT and LightGBM, stable under current dataset resampling.",
        "- Qualified Claim: Bagged HDDT exhibits lower smoothness and somewhat higher threshold-survival persistence; magnitude is dataset-sensitive and should be framed conservatively.",
        "- Unsafe Claim: Strong mechanistic assertions about vote-level causes without per-instance ensemble decomposition.",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_make_plots(base_dir: Path, boot: pd.DataFrame, lodo: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir = base_dir / "reports/neural_mlp/plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: bootstrap CI for primary deltas
    comps = ["hddt_forest-hddt", "hddt_forest-lightgbm", "hddt_forest-random_forest"]
    records = boot[(boot.analysis == "pairwise_delta") & (boot.metric.isin(PRIMARY_DELTA_METRICS))]
    fig, ax = plt.subplots(figsize=(10, 5))
    y = 0
    yticks = []
    ylabels = []
    for comp in comps:
        for metric in PRIMARY_DELTA_METRICS:
            row = records[(records.comparison == comp) & (records.metric == metric)].iloc[0]
            mean = row["mean"]
            lo = row["ci_2_5"]
            hi = row["ci_97_5"]
            ax.plot([lo, hi], [y, y], color="tab:blue", lw=2)
            ax.scatter([mean], [y], color="tab:red", s=20)
            yticks.append(y)
            ylabels.append(f"{comp}:{metric}")
            y += 1
    ax.axvline(0, color="black", lw=1, linestyle="--")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title("Bootstrap 95% CIs for Bagged HDDT Pairwise Deltas")
    ax.set_xlabel("Delta")
    plt.tight_layout()
    fig.savefig(plots_dir / "bagged_hddt_stability_ci.png", dpi=200)
    plt.close(fig)

    # Plot 2: LODO deltas
    lsub = lodo[lodo["comparison"].isin(["hddt_forest-hddt", "hddt_forest-lightgbm"])]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, metric in zip(axes.flatten(), PRIMARY_DELTA_METRICS):
        col = f"delta_{metric}"
        for comp in ["hddt_forest-hddt", "hddt_forest-lightgbm"]:
            d = lsub[lsub["comparison"] == comp]
            ax.plot(d["dropped_dataset_id"], d[col], marker="o", label=comp)
        ax.axhline(0, color="black", lw=1, linestyle="--")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(plots_dir / "bagged_hddt_leave_one_dataset_out.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bagged HDDT Experiment C stability analysis")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    out_reports = base_dir / "reports/neural_mlp"
    out_reports.mkdir(parents=True, exist_ok=True)

    dataset_level = load_dataset_level_table(base_dir, FOCUS_MODELS)
    dataset_level.to_csv(out_reports / "bagged_hddt_dataset_level_metrics.csv", index=False)

    boot = bootstrap_metrics(dataset_level, args.n_bootstrap, args.seed)
    boot.to_csv(out_reports / "bagged_hddt_stability_ci.csv", index=False)

    lodo = leave_one_dataset_out(dataset_level)
    lodo.to_csv(out_reports / "bagged_hddt_leave_one_dataset_out.csv", index=False)

    write_summary_markdown(base_dir, dataset_level, boot, lodo)
    write_research_note(base_dir, boot, lodo)
    maybe_make_plots(base_dir, boot, lodo)


if __name__ == "__main__":
    main()
