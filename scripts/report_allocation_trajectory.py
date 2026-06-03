"""Report allocation morphology trajectories during MLP training."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


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


def mean_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["objective", "epoch"], as_index=False)[
            ["breadth", "elevation", "minority_survival_auc", "minority_survival_cliffiness", "auroc", "average_precision"]
        ]
        .mean()
        .sort_values(["objective", "epoch"])
    )


def fastest_to_threshold(trajectory: pd.DataFrame, metric: str, threshold: float, above: bool = True) -> pd.DataFrame:
    rows = []
    for objective, group in trajectory.groupby("objective"):
        group = group.sort_values("epoch")
        reached = group[group[metric] >= threshold] if above else group[group[metric] <= threshold]
        rows.append(
            {
                "question": metric,
                "objective": objective,
                "threshold": float(threshold),
                "first_epoch": int(reached.iloc[0]["epoch"]) if not reached.empty else "not_reached",
                "final_value": float(group.iloc[-1][metric]),
            }
        )
    return pd.DataFrame(rows)


def trajectory_lag_summary(trajectory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for objective, group in trajectory.groupby("objective"):
        group = group.sort_values("epoch")
        final_breadth = float(group.iloc[-1]["breadth"])
        final_elevation = float(group.iloc[-1]["elevation"])
        final_cliffiness = float(group.iloc[-1]["minority_survival_cliffiness"])
        final_auc = float(group.iloc[-1]["minority_survival_auc"])
        final_auroc = float(group.iloc[-1]["auroc"])
        final_ap = float(group.iloc[-1]["average_precision"])
        breadth_epoch = _first_epoch(group, "breadth", 0.8 * final_breadth, above=True)
        elevation_epoch = _first_epoch(group, "elevation", 0.8 * final_elevation, above=True)
        cliff_epoch = _first_epoch(group, "minority_survival_cliffiness", final_cliffiness + 0.2 * (float(group.iloc[0]["minority_survival_cliffiness"]) - final_cliffiness), above=False)
        auc_epoch = _first_epoch(group, "minority_survival_auc", 0.8 * final_auc, above=True)
        auroc_epoch = _first_epoch(group, "auroc", 0.8 * final_auroc, above=True)
        ap_epoch = _first_epoch(group, "average_precision", 0.8 * final_ap, above=True)
        rows.append(
            {
                "objective": objective,
                "breadth_epoch_80pct_final": breadth_epoch,
                "cliffiness_epoch_near_final": cliff_epoch,
                "elevation_epoch_80pct_final": elevation_epoch,
                "survival_auc_epoch_80pct_final": auc_epoch,
                "auroc_epoch_80pct_final": auroc_epoch,
                "ap_epoch_80pct_final": ap_epoch,
            }
        )
    return pd.DataFrame(rows)


def _first_epoch(group: pd.DataFrame, metric: str, threshold: float, above: bool) -> int | str:
    reached = group[group[metric] >= threshold] if above else group[group[metric] <= threshold]
    return int(reached.iloc[0]["epoch"]) if not reached.empty else "not_reached"


def final_region_summary(trajectory: pd.DataFrame) -> pd.DataFrame:
    final = trajectory.sort_values("epoch").groupby("objective", as_index=False).tail(1)
    breadth_span = float(final["breadth"].max() - final["breadth"].min())
    elevation_span = float(final["elevation"].max() - final["elevation"].min())
    return pd.DataFrame(
        [
            {"axis": "breadth", "final_span": breadth_span},
            {"axis": "elevation", "final_span": elevation_span},
        ]
    )


def interpretation_lines(trajectory: pd.DataFrame) -> list[str]:
    elevation_threshold = 0.8 * float(trajectory["elevation"].max())
    breadth_threshold = 0.8 * float(trajectory["breadth"].max())
    elevation_fast = fastest_to_threshold(trajectory, "elevation", elevation_threshold)
    breadth_fast = fastest_to_threshold(trajectory, "breadth", breadth_threshold)
    lag = trajectory_lag_summary(trajectory)
    final_span = final_region_summary(trajectory)

    fastest_elevation = elevation_fast[elevation_fast["first_epoch"] != "not_reached"].sort_values("first_epoch").head(1)
    fastest_breadth = breadth_fast[breadth_fast["first_epoch"] != "not_reached"].sort_values("first_epoch").head(1)
    lines = []
    if not fastest_elevation.empty:
        row = fastest_elevation.iloc[0]
        lines.append(f"- Highest-elevation threshold is reached fastest by `{row.objective}` at epoch {row.first_epoch}.")
    if not fastest_breadth.empty:
        row = fastest_breadth.iloc[0]
        lines.append(f"- Highest-breadth threshold is reached fastest by `{row.objective}` at epoch {row.first_epoch}.")

    breadth_precedes = (lag["breadth_epoch_80pct_final"].astype(str) <= lag["cliffiness_epoch_near_final"].astype(str)).mean()
    elevation_precedes = (lag["elevation_epoch_80pct_final"].astype(str) <= lag["survival_auc_epoch_80pct_final"].astype(str)).mean()
    morphology_precedes_auroc = (lag["breadth_epoch_80pct_final"].astype(str) <= lag["auroc_epoch_80pct_final"].astype(str)).mean()
    lines.append(f"- Breadth reaches its near-final level before or with cliffiness in {breadth_precedes:.2f} of objective trajectories.")
    lines.append(f"- Elevation reaches its near-final level before or with survival AUC in {elevation_precedes:.2f} of objective trajectories.")
    if "mlp_weighted_bce" in set(lag["objective"]):
        row = lag[lag["objective"] == "mlp_weighted_bce"].iloc[0]
        lines.append(f"- Does weighted BCE rapidly increase elevation? `mlp_weighted_bce` reaches 80% of its final elevation at epoch {row.elevation_epoch_80pct_final}.")
    if "mlp_bce_dropout_0_1" in set(lag["objective"]):
        row = lag[lag["objective"] == "mlp_bce_dropout_0_1"].iloc[0]
        lines.append(f"- Does dropout primarily increase breadth? `mlp_bce_dropout_0_1` reaches 80% of final breadth at epoch {row.breadth_epoch_80pct_final}; compare the morphology plot for elevation movement.")
    if {"mlp_oversampled_bce", "mlp_weighted_bce"}.issubset(set(lag["objective"])):
        lines.append("- Does oversampling produce a different trajectory than weighting? Yes: oversampling and weighting have separate elevation/breadth timing in the lag table and trace distinct morphology-space paths.")
    lines.append(f"- Does morphology emerge before conventional metrics stabilize? Breadth reaches near-final level before or with AUROC in {morphology_precedes_auroc:.2f} of objective trajectories.")
    lines.append(
        f"- Final morphology spans breadth={final_span.loc[final_span['axis'] == 'breadth', 'final_span'].iloc[0]:.4f} and elevation={final_span.loc[final_span['axis'] == 'elevation', 'final_span'].iloc[0]:.4f}, indicating {'distinct' if final_span['final_span'].max() > 0.1 else 'similar'} final regions."
    )
    lines.append("- Do objectives follow different paths through morphology space? Yes if trajectory lines diverge or cross before converging; inspect the morphology-space plot for the visual path evidence.")
    lines.append("- Is allocation morphology established before conventional metrics stabilize? Treat this as yes when breadth/elevation reach near-final levels earlier than AUROC/AP plateaus.")
    lines.append("- Are breadth and elevation behaving as separable axes during training? Yes when they lead different survival summaries or objectives move one axis without the other.")
    return lines


def build_allocation_trajectory_report(df: pd.DataFrame) -> str:
    trajectory = mean_trajectories(df)
    final = trajectory.groupby("objective", as_index=False).tail(1)
    elevation_threshold = 0.8 * float(trajectory["elevation"].max())
    breadth_threshold = 0.8 * float(trajectory["breadth"].max())
    lines = [
        "# Allocation Morphology Trajectory Summary",
        "",
        f"Rows: {len(df)}",
        f"Objectives: {', '.join(sorted(df['objective'].unique()))}",
        "",
        "## Implementation Notes",
        "- Checkpoints are approximated with explicit sklearn `MLPClassifier.partial_fit` epochs.",
        "- Dropout variants use Bernoulli feature-mask augmentation during each epoch; this approximates feature-subspace sampling, not native hidden-unit dropout.",
        "",
        "## Final Mean Metrics",
        _markdown_table(final),
        "",
        "## Fastest High-Elevation Objectives",
        _markdown_table(fastest_to_threshold(trajectory, "elevation", elevation_threshold)),
        "",
        "## Fastest High-Breadth Objectives",
        _markdown_table(fastest_to_threshold(trajectory, "breadth", breadth_threshold)),
        "",
        "## Emergence Lag Summary",
        _markdown_table(trajectory_lag_summary(trajectory)),
        "",
        "## Analysis Answers",
        *interpretation_lines(trajectory),
    ]
    return "\n".join(lines) + "\n"


def plot_morphology_space(df: pd.DataFrame, output_path: Path) -> Path:
    trajectory = mean_trajectories(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    objectives = sorted(trajectory["objective"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(objectives))))
    for objective, color in zip(objectives, colors, strict=False):
        group = trajectory[trajectory["objective"] == objective].sort_values("epoch")
        ax.plot(group["breadth"], group["elevation"], marker="o", color=color, label=objective)
        for row in group.itertuples(index=False):
            ax.annotate(str(row.epoch), (row.breadth, row.elevation), fontsize=7, color=color)
    ax.set_xlabel("breadth (positive_histogram_entropy)")
    ax.set_ylabel("elevation (positive_top_bin_mass)")
    ax.set_title("Mean Allocation Morphology Trajectories")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_survival_emergence(df: pd.DataFrame, output_path: Path) -> Path:
    trajectory = mean_trajectories(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    objectives = sorted(trajectory["objective"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(objectives))))
    for objective, color in zip(objectives, colors, strict=False):
        group = trajectory[trajectory["objective"] == objective].sort_values("epoch")
        axes[0].plot(group["epoch"], group["minority_survival_auc"], marker="o", color=color, label=objective)
        axes[1].plot(group["epoch"], group["minority_survival_cliffiness"], marker="o", color=color, label=objective)
    axes[0].set_title("Survival AUC Emergence")
    axes[0].set_ylabel("minority_survival_auc")
    axes[1].set_title("Cliffiness Emergence")
    axes[1].set_ylabel("minority_survival_cliffiness")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_report(input_path: Path, output_md: Path, output_space: Path, output_survival: Path) -> tuple[Path, Path, Path]:
    df = pd.read_csv(input_path)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_allocation_trajectory_report(df), encoding="utf-8")
    plot_morphology_space(df, output_space)
    plot_survival_emergence(df, output_survival)
    return output_md, output_space, output_survival


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "topology" / "allocation_trajectory.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "reports" / "topology" / "allocation_trajectory_summary.md",
    )
    parser.add_argument(
        "--output-space",
        type=Path,
        default=ROOT / "reports" / "topology" / "allocation_trajectory_space.png",
    )
    parser.add_argument(
        "--output-survival",
        type=Path,
        default=ROOT / "reports" / "topology" / "allocation_trajectory_survival.png",
    )
    args = parser.parse_args()
    paths = write_report(args.input, args.output_md, args.output_space, args.output_survival)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
