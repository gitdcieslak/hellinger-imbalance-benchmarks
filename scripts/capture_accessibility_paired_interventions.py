"""Capture paired intervention-ready accessibility coordinate rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "topology"
DEFAULT_COORDINATES = REPORTS / "accessibility_coordinate_scores.csv"
DEFAULT_FEATURES = REPORTS / "regime_feature_matrix.csv"

REQUIRED_COORDS = [f"coord_{i}" for i in range(1, 6)]
OUTCOME_COLS = ["minority_survival_auc", "minority_survival_cliffiness", "persistence", "elevation", "effective_support"]


def _extract_int(pattern: str, value: str) -> int | None:
    match = re.search(pattern, str(value))
    return int(match.group(1)) if match else None


def load_source(coordinates_path: Path, features_path: Path) -> pd.DataFrame:
    coords = pd.read_csv(coordinates_path)
    features = pd.read_csv(features_path)
    keep = [c for c in features.columns if c not in coords.columns or c == "run_id"]
    data = coords.merge(features[keep], on="run_id", how="left")
    data["seed"] = data["run_id"].map(lambda x: _extract_int(r"seed=(\d+)", x))
    data["split_id"] = data["run_id"].map(lambda x: _extract_int(r"split=(\d+)", x))
    data["dataset"] = data["dataset_id"]
    data["persistence"] = data.get("persistence_weighted_mass", np.nan)
    data["effective_support"] = data.get("effective_support_groups", np.nan)
    for i in range(1, 6):
        data[f"coord_{i}"] = data[f"coord_pc{i}"]
    return data


def _base_record(row: pd.Series, pair_id: str, intervention: str, stage: str, strength: float, base_model_id: str, model_family: str) -> dict[str, object]:
    record = {
        "pair_id": pair_id,
        "dataset": row["dataset"],
        "dataset_id": row["dataset_id"],
        "task_id": row["task_id"],
        "split_id": int(row["split_id"]),
        "seed": int(row["seed"]),
        "model_family": model_family,
        "base_model_id": base_model_id,
        "model_id": row["model_id"],
        "run_id": row["run_id"],
        "intervention": intervention,
        "stage": stage,
        "strength": float(strength),
    }
    for col in [*REQUIRED_COORDS, *OUTCOME_COLS]:
        record[col] = row[col]
    metric_cols = [
        "n_score_groups",
        "support_unique_score_ratio",
        "group_mass_entropy",
        "group_mass_gini",
        "group_mass_hhi",
        "top1_group_mass",
        "top3_group_mass",
        "top5_group_mass",
        "effective_support_groups",
        "n_support_islands",
        "largest_island_mass",
        "island_entropy",
        "island_gini",
        "small_island_fraction",
        "largest_to_median_group_ratio",
        "persistence_weighted_mass",
        "fragile_support_mass",
        "stable_support_mass",
        "n_unique_positive_posteriors",
        "effective_posterior_alphabet",
        "posterior_hhi",
        "largest_posterior_mass",
        "posterior_entropy",
        "mean_posterior_gap",
        "max_posterior_gap",
        "mass_at_largest_gap",
        "mass_above_largest_gap",
        "support_redundancy",
        "mass_outside_top3",
        "mass_outside_top5",
        "support_fragility_index",
        "broad_allocator_index",
        "concentrated_allocator_index",
        "quantized_allocator_index",
        "continuous_allocator_index",
        "fragmented_allocator_index",
        "redundant_allocator_index",
        "hierarchical_allocator_index",
        "breadth",
    ]
    for col in metric_cols:
        if col in row.index:
            record[col] = row[col]
    return record


def capture_bagging_pairs(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required = {"cart", "random_forest"}
    for key, group in data.groupby(["dataset_id", "task_id", "split_id", "seed"], dropna=False):
        present = set(group["model_id"].astype(str))
        if not required.issubset(present):
            continue
        cart = group[group["model_id"].astype(str) == "cart"].iloc[0]
        rf = group[group["model_id"].astype(str) == "random_forest"].iloc[0]
        dataset_id, task_id, split_id, seed = key
        pair_id = f"{dataset_id}|task={task_id}|split={split_id}|seed={seed}|base=cart|intervention=bagging"
        rows.append(_base_record(cart, pair_id, "bagging", "before", 0.0, "cart", "tree_ensemble"))
        rows.append(_base_record(rf, pair_id, "bagging", "after", 1.0, "cart", "tree_ensemble"))
    return pd.DataFrame(rows)


def validate_pairs(paired: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    issues = []
    if paired.empty:
        issues.append({"check": "nonempty", "status": "fail", "detail": "No paired rows captured."})
    for col in ["pair_id", "dataset", "split_id", "seed", "model_family", "base_model_id", "intervention", "stage", "strength", *REQUIRED_COORDS, *OUTCOME_COLS]:
        ok = col in paired.columns
        issues.append({"check": f"column:{col}", "status": "pass" if ok else "fail", "detail": "" if ok else "missing"})
    if not paired.empty:
        for pair_id, group in paired.groupby("pair_id"):
            stages = set(group["stage"].astype(str))
            if not ({"before", "after"}.issubset(stages) or group["strength"].nunique() >= 2):
                issues.append({"check": "pair_has_two_stages_or_strengths", "status": "fail", "detail": pair_id})
            for col in ["dataset", "task_id", "split_id", "seed", "base_model_id", "intervention"]:
                if group[col].nunique(dropna=False) != 1:
                    issues.append({"check": f"pair_constant:{col}", "status": "fail", "detail": pair_id})
        numeric_cols = [*REQUIRED_COORDS, *OUTCOME_COLS]
        for col in numeric_cols:
            if col in paired.columns and not pd.api.types.is_numeric_dtype(paired[col]):
                issues.append({"check": f"numeric:{col}", "status": "fail", "detail": "not numeric"})
    issue_df = pd.DataFrame(issues)
    n_pairs = int(paired["pair_id"].nunique()) if "pair_id" in paired else 0
    n_vectors = int(sum({"before", "after"}.issubset(set(g["stage"].astype(str))) for _, g in paired.groupby("pair_id"))) if not paired.empty else 0
    families = sorted(paired["intervention"].dropna().astype(str).unique()) if "intervention" in paired else []
    max_strength_levels = int(paired.groupby("intervention")["strength"].nunique().max()) if not paired.empty else 0
    summary = {
        "n_rows": int(len(paired)),
        "n_pairs": n_pairs,
        "n_paired_vectors": n_vectors,
        "intervention_families": families,
        "n_intervention_families": len(families),
        "max_strength_levels": max_strength_levels,
        "validation_passed": bool((issue_df["status"] == "fail").sum() == 0) if not issue_df.empty else False,
        "minimum_success": n_vectors > 0,
        "strong_success": n_vectors >= 100 and len(families) >= 2 and max_strength_levels >= 3,
        "ideal_success": {"bagging", "dropout", "weighting"}.issubset(set(families)) or {"bagging", "dropout", "objective_shift"}.issubset(set(families)),
    }
    return issue_df, summary


def write_validation(summary: dict[str, object], issues: pd.DataFrame, output: Path) -> Path:
    output.write_text(
        "\n".join(
            [
                "# Accessibility Paired Intervention Capture Validation",
                "",
                f"Rows: {summary['n_rows']}",
                f"Pairs: {summary['n_pairs']}",
                f"Paired vectors: {summary['n_paired_vectors']}",
                f"Intervention families: {', '.join(summary['intervention_families']) if summary['intervention_families'] else 'none'}",
                f"Validation passed: {summary['validation_passed']}",
                f"Minimum success: {summary['minimum_success']}",
                f"Strong success: {summary['strong_success']}",
                f"Ideal success: {summary['ideal_success']}",
                "",
                "## Notes",
                "Captured real matched CART -> random_forest rows as the available bagging/ensemble intervention in the current coordinate artifact.",
                "Dropout, weighting, oversampling, and HDDT -> Bagged HDDT are not present in the current coordinate-score artifact and therefore are not fabricated.",
                "",
                "## Failed Checks",
                issues[issues["status"] == "fail"].to_markdown(index=False) if not issues[issues["status"] == "fail"].empty else "None.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def write_capture(coordinates: Path = DEFAULT_COORDINATES, features: Path = DEFAULT_FEATURES, output_dir: Path = REPORTS) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_source(coordinates, features)
    paired = capture_bagging_pairs(data)
    issues, summary = validate_pairs(paired)
    csv_path = output_dir / "accessibility_paired_interventions.csv"
    summary_path = output_dir / "accessibility_paired_intervention_summary.json"
    validation_path = output_dir / "accessibility_paired_intervention_validation.md"
    paired.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_validation(summary, issues, validation_path)
    return csv_path, summary_path, validation_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, default=DEFAULT_COORDINATES)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=REPORTS)
    args = parser.parse_args()
    for output in write_capture(args.coordinates, args.features, args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
