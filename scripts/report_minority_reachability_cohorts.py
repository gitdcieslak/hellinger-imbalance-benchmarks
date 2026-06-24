"""Analyze minority-example cohorts that drive reachability cliffs."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
DEFAULT_POSITIVE_SCORES = ROOT / "results" / "real_validation" / "hddt_positive_scores.csv"
OUT = ROOT / "reports" / "topology"

from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402


def load_positive_scores(path: Path = DEFAULT_POSITIVE_SCORES) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = ["run_id", "dataset_id", "task_id", "model_id", "seed", "example_id", "score", "minority_survival_cliffiness"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing positive-score columns: {missing}")
    df = df.dropna(subset=required).copy()
    df["positive_key"] = df["dataset_id"].astype(str) + "|" + df["task_id"].astype(str) + "|" + df["example_id"].astype(int).astype(str)
    df["score_round"] = df["score"].astype(float).round(6)
    return df


def largest_drop_transitions(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    members = []
    for run_id, group in scores.groupby("run_id", sort=False):
        drop_candidates = group[group["score_round"] < 1.0].copy()
        counts = drop_candidates.groupby("score_round").size().sort_values(ascending=False)
        if counts.empty:
            first = group.iloc[0]
            summaries.append(
                {
                    "run_id": run_id,
                    "dataset_id": first["dataset_id"],
                    "task_id": first["task_id"],
                    "model_id": first["model_id"],
                    "seed": int(first["seed"]),
                    "largest_drop_threshold": np.nan,
                    "drop_size": 0.0,
                    "n_positives_removed": 0,
                    "n_positives_total": int(len(group)),
                    "minority_survival_cliffiness": float(first["minority_survival_cliffiness"]),
                    "minority_survival_auc": float(first.get("minority_survival_auc", np.nan)),
                    "breadth": float(first.get("breadth", np.nan)),
                    "elevation": float(first.get("elevation", np.nan)),
                }
            )
            continue
        drop_score = float(counts.index[0])
        cohort = drop_candidates[drop_candidates["score_round"] == drop_score].copy()
        n_pos = int(len(group))
        drop_n = int(len(cohort))
        first = group.iloc[0]
        summaries.append(
            {
                "run_id": run_id,
                "dataset_id": first["dataset_id"],
                "task_id": first["task_id"],
                "model_id": first["model_id"],
                "seed": int(first["seed"]),
                "largest_drop_threshold": drop_score,
                "drop_size": float(drop_n / max(1, n_pos)),
                "n_positives_removed": drop_n,
                "n_positives_total": n_pos,
                "minority_survival_cliffiness": float(first["minority_survival_cliffiness"]),
                "minority_survival_auc": float(first.get("minority_survival_auc", np.nan)),
                "breadth": float(first.get("breadth", np.nan)),
                "elevation": float(first.get("elevation", np.nan)),
            }
        )
        cohort = cohort.assign(largest_drop_threshold=drop_score, drop_size=float(drop_n / max(1, n_pos)), n_positives_removed=drop_n, n_positives_total=n_pos)
        members.append(cohort)
    return pd.DataFrame(summaries), pd.concat(members, ignore_index=True) if members else pd.DataFrame()


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return float(len(a & b) / max(1, len(a | b)))


def overlap_analysis(members: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    matrix_rows = []
    if members.empty:
        return pd.DataFrame(), pd.DataFrame()
    cohorts = {run_id: set(group["positive_key"].astype(str)) for run_id, group in members.groupby("run_id")}
    meta = members.groupby("run_id").first()[["dataset_id", "task_id", "model_id", "seed"]]
    for run_a, run_b in itertools.combinations(cohorts, 2):
        a = meta.loc[run_a]
        b = meta.loc[run_b]
        if a["task_id"] != b["task_id"]:
            continue
        value = jaccard(cohorts[run_a], cohorts[run_b])
        rows.append(
            {
                "task_id": a["task_id"],
                "dataset_id": a["dataset_id"],
                "model_a": a["model_id"],
                "model_b": b["model_id"],
                "seed_a": int(a["seed"]),
                "seed_b": int(b["seed"]),
                "same_model": bool(a["model_id"] == b["model_id"]),
                "jaccard_overlap": value,
            }
        )
        matrix_rows.append({"run_a": run_a, "run_b": run_b, "jaccard_overlap": value})
    return pd.DataFrame(rows), pd.DataFrame(matrix_rows)


def cohort_persistence(scores: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    total = scores.groupby(["task_id", "model_id", "positive_key"], as_index=False).agg(n_seen=("run_id", "nunique"), mean_score=("score", "mean"))
    if members.empty:
        total["n_cliff_driver"] = 0
    else:
        driver = members.groupby(["task_id", "model_id", "positive_key"], as_index=False).agg(n_cliff_driver=("run_id", "nunique"), mean_driver_threshold=("largest_drop_threshold", "mean"))
        total = total.merge(driver, on=["task_id", "model_id", "positive_key"], how="left")
    total["n_cliff_driver"] = total["n_cliff_driver"].fillna(0).astype(int)
    total["cohort_persistence"] = total["n_cliff_driver"] / total["n_seen"].clip(lower=1)
    return total.sort_values("cohort_persistence", ascending=False).reset_index(drop=True)


def cohort_entropy(persistence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task_id, model_id), group in persistence.groupby(["task_id", "model_id"]):
        weights = group["n_cliff_driver"].to_numpy(dtype=float)
        weights = weights[weights > 0]
        if weights.size == 0:
            ent = 0.0
            eff = 0.0
        else:
            p = weights / weights.sum()
            ent = float(-(p * np.log(p)).sum())
            eff = float(np.exp(ent))
        rows.append({"task_id": task_id, "model_id": model_id, "cohort_entropy": ent, "effective_cliff_driver_count": eff, "n_driver_examples": int((group["n_cliff_driver"] > 0).sum())})
    return pd.DataFrame(rows)


def positive_reachability_matrix(scores: pd.DataFrame, output: Path, max_runs: int = 12) -> pd.DataFrame:
    selected_runs = scores.groupby("run_id")["minority_survival_cliffiness"].first().sort_values(ascending=False).head(max_runs).index
    subset = scores[scores["run_id"].isin(selected_runs)].copy()
    thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for row in subset.itertuples(index=False):
        for threshold in thresholds:
            rows.append({"run_id": row.run_id, "positive_key": row.positive_key, "threshold": float(threshold), "reachable": int(float(row.score) >= float(threshold)), "score": float(row.score)})
    matrix = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output, index=False)
    return matrix


def profile_table(scores: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    all_scores = scores.assign(is_cliff_driver=False)
    if not members.empty:
        driver_keys = set(zip(members["run_id"], members["positive_key"]))
        all_scores["is_cliff_driver"] = [(run_id, key) in driver_keys for run_id, key in zip(all_scores["run_id"], all_scores["positive_key"])]
    return all_scores.groupby("is_cliff_driver", as_index=False).agg(
        n_examples=("positive_key", "size"),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        mean_cliffiness=("minority_survival_cliffiness", "mean"),
        mean_breadth=("breadth", "mean"),
        mean_elevation=("elevation", "mean"),
    )


def plot_positive_heatmap(matrix: pd.DataFrame, output: Path) -> Path:
    if matrix.empty:
        return output
    run = matrix["run_id"].iloc[0]
    pivot = matrix[matrix["run_id"] == run].pivot_table(index="positive_key", columns="threshold", values="reachable", aggfunc="max")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("Positive Reachability Heatmap")
    ax.set_xlabel("threshold index")
    ax.set_ylabel("positive example")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_overlap(overlap: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    if overlap.empty:
        ax.text(0.5, 0.5, "No overlaps", ha="center")
    else:
        summary = overlap.groupby(["model_a", "model_b"], as_index=False)["jaccard_overlap"].mean()
        pivot = summary.pivot(index="model_a", columns="model_b", values="jaccard_overlap").fillna(0)
        im = ax.imshow(pivot.to_numpy(), cmap="magma", vmin=0, vmax=max(0.01, float(pivot.to_numpy().max())))
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        fig.colorbar(im, ax=ax, label="mean Jaccard")
    ax.set_title("Cliff Cohort Overlap Matrix")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_persistence(persistence: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(persistence["cohort_persistence"], bins=20, color="steelblue", alpha=0.85)
    ax.set_xlabel("cohort persistence")
    ax.set_ylabel("positive examples")
    ax.set_title("Cliff-Driver Persistence")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_size_distribution(transitions: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(transitions["drop_size"], bins=25, color="tomato", alpha=0.85)
    ax.set_xlabel("largest-drop fraction")
    ax.set_ylabel("runs")
    ax.set_title("Largest-Drop Cohort Size Distribution")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_topology_overlay(transitions: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(transitions["breadth"], transitions["elevation"], c=transitions["drop_size"], cmap="viridis", s=16, alpha=0.75)
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Positive Cohort Drop Size in Morphology Space")
    fig.colorbar(sc, ax=ax, label="largest-drop fraction")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_reports(output_dir: Path = OUT, positive_scores: Path = DEFAULT_POSITIVE_SCORES) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = load_positive_scores(positive_scores)
    transitions, members = largest_drop_transitions(scores)
    overlap, _ = overlap_analysis(members)
    persistence = cohort_persistence(scores, members)
    ent = cohort_entropy(persistence)
    profiles = profile_table(scores, members)
    matrix_path = output_dir / "positive_reachability_matrix.csv"
    matrix = positive_reachability_matrix(scores, matrix_path)
    transitions_path = output_dir / "cliff_transition_examples.csv"
    members.to_csv(transitions_path, index=False)
    memberships_path = output_dir / "cohort_memberships.csv"
    persistence.to_csv(memberships_path, index=False)
    summary = {
        "n_runs": int(scores["run_id"].nunique()),
        "n_positive_score_rows": int(len(scores)),
        "mean_largest_drop_fraction": float(transitions["drop_size"].mean()),
        "mean_same_model_jaccard": float(overlap[overlap["same_model"]]["jaccard_overlap"].mean()) if not overlap.empty and overlap["same_model"].any() else np.nan,
        "mean_cross_model_jaccard": float(overlap[~overlap["same_model"]]["jaccard_overlap"].mean()) if not overlap.empty and (~overlap["same_model"]).any() else np.nan,
        "mean_cohort_persistence": float(persistence["cohort_persistence"].mean()),
        "high_persistence_examples": int((persistence["cohort_persistence"] >= 0.5).sum()),
    }
    main = output_dir / "minority_reachability_cohorts.md"
    main.write_text("\n".join([
        "# Minority Reachability Cohorts",
        "",
        "## Executive Answer",
        executive_answer(summary),
        "",
        "## Aggregate Summary",
        _markdown_table(pd.DataFrame([summary])),
        "",
        "## Largest-Drop Transitions",
        _markdown_table(transitions.sort_values("drop_size", ascending=False).head(30)),
        "",
        "## Cohort Persistence",
        _markdown_table(persistence.head(30)),
        "",
        "## Cohort Entropy",
        _markdown_table(ent.sort_values("effective_cliff_driver_count", ascending=False).head(30)),
    ]) + "\n", encoding="utf-8")
    overlap_md = output_dir / "cohort_overlap_analysis.md"
    overlap_md.write_text("\n".join(["# Cohort Overlap Analysis", "", _markdown_table(overlap), ""]) , encoding="utf-8")
    profiles_md = output_dir / "cliff_driver_profiles.md"
    profiles_md.write_text("\n".join(["# Cliff Driver Profiles", "", _markdown_table(profiles), "", "Raw feature distributions/topology component membership require saving feature vectors per original example; this run profiles score and run-level morphology only."]) + "\n", encoding="utf-8")
    return (
        main,
        overlap_md,
        profiles_md,
        matrix_path,
        transitions_path,
        memberships_path,
        plot_positive_heatmap(matrix, output_dir / "positive_reachability_heatmap.png"),
        plot_positive_heatmap(matrix, output_dir / "cohort_clustering_heatmap.png"),
        plot_size_distribution(transitions, output_dir / "largest_drop_cohort_examples.png"),
        plot_overlap(overlap, output_dir / "cohort_overlap_matrix.png"),
        plot_persistence(persistence, output_dir / "cohort_persistence_plots.png"),
        plot_topology_overlay(transitions, output_dir / "positive_topology_overlays.png"),
    )


def executive_answer(summary: dict[str, float]) -> str:
    same = summary.get("mean_same_model_jaccard", np.nan)
    persistence = summary.get("mean_cohort_persistence", np.nan)
    if np.isfinite(same) and same > 0.25 and np.isfinite(persistence) and persistence > 0.25:
        return "Large reachability drops show evidence of stable positive-example cohorts within model families."
    return "Large reachability drops can be identified at the example level, but cohort stability is weak or mixed under the captured runs."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-scores", type=Path, default=DEFAULT_POSITIVE_SCORES)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_reports(args.output_dir, args.positive_scores):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
