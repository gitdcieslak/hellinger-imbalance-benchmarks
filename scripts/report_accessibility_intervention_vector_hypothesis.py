"""Test whether interventions induce consistent accessibility-coordinate vectors."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "topology"
DEFAULT_COORDINATES = OUT / "accessibility_coordinate_scores.csv"
TARGETS = ["minority_survival_cliffiness", "persistence_weighted_mass", "minority_survival_auc", "breadth", "elevation"]


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            vals.append(format(value, floatfmt) if isinstance(value, float) and pd.notna(value) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def coordinate_columns(df: pd.DataFrame) -> list[str]:
    pc_cols = [c for c in df.columns if c.startswith("coord_pc") and not c.endswith("explained_variance")]
    if pc_cols:
        return pc_cols
    return [c for c in df.columns if re.match(r"coord_\d+$", c)]


def delta_coordinate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("delta_coord_pc") or re.match(r"delta_coord_\d+$", c)]


def has_pairing_schema(df: pd.DataFrame) -> bool:
    return {"pair_id", "intervention", "stage"}.issubset(df.columns) and {"before", "after"}.issubset(set(df["stage"].astype(str)))


def displacement_vectors(df: pd.DataFrame) -> pd.DataFrame:
    if not has_pairing_schema(df):
        return pd.DataFrame()
    coord_cols = coordinate_columns(df)
    targets = [c for c in TARGETS if c in df.columns]
    rows = []
    for (pair_id, intervention), group in df.groupby(["pair_id", "intervention"], sort=False):
        before = group[group["stage"].astype(str) == "before"]
        after = group[group["stage"].astype(str) == "after"]
        if before.empty or after.empty:
            continue
        b = before.iloc[0]
        a = after.iloc[-1]
        row = {"pair_id": pair_id, "intervention": intervention}
        for meta in ["dataset_id", "task_id", "model_id", "seed", "split_id"]:
            if meta in df.columns:
                row[meta] = b.get(meta)
        for col in coord_cols:
            row[f"delta_{col}"] = float(a[col] - b[col])
        for target in targets:
            row[f"delta_{target}"] = float(a[target] - b[target])
        row["vector_magnitude"] = float(np.linalg.norm([row[f"delta_{c}"] for c in coord_cols]))
        rows.append(row)
    return pd.DataFrame(rows)


def _cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = vectors / np.maximum(norms, 1e-12)
    return safe @ safe.T


def vector_consistency(displacements: pd.DataFrame) -> pd.DataFrame:
    if displacements.empty:
        return pd.DataFrame([{"intervention": "unavailable", "n_vectors": 0, "mean_cosine_similarity": np.nan, "angular_dispersion": np.nan, "magnitude_cv": np.nan, "principal_vector_variance": np.nan, "vector_concentration_score": np.nan}])
    delta_cols = delta_coordinate_columns(displacements)
    rows = []
    rng = np.random.default_rng(181)
    for intervention, group in displacements.groupby("intervention"):
        V = group[delta_cols].to_numpy(dtype=float)
        if len(V) < 2:
            rows.append({"intervention": intervention, "n_vectors": len(V), "mean_cosine_similarity": np.nan, "angular_dispersion": np.nan, "magnitude_cv": np.nan, "principal_vector_variance": np.nan, "vector_concentration_score": np.nan})
            continue
        cos = _cosine_matrix(V)
        tri = cos[np.triu_indices_from(cos, k=1)]
        mags = np.linalg.norm(V, axis=1)
        centered = V - V.mean(axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        total_var = float(np.sum(s**2))
        pc_var = float(s[0] ** 2 / max(1e-12, total_var)) if len(s) else np.nan
        random = rng.normal(size=V.shape)
        random_cos = _cosine_matrix(random)[np.triu_indices(len(random), k=1)].mean()
        rows.append({"intervention": intervention, "n_vectors": len(V), "mean_cosine_similarity": float(np.mean(tri)), "random_mean_cosine_similarity": float(random_cos), "angular_dispersion": float(1 - np.mean(tri)), "magnitude_cv": float(np.std(mags) / max(1e-12, np.mean(mags))), "principal_vector_variance": pc_var, "vector_concentration_score": float(np.mean(tri) / max(1e-12, 1 - random_cos))})
    return pd.DataFrame(rows)


def morphology_prediction(displacements: pd.DataFrame) -> pd.DataFrame:
    if displacements.empty:
        return pd.DataFrame([{"target": "unavailable", "cv_r2": np.nan, "mae": np.nan, "n_pairs": 0}])
    delta_coord = delta_coordinate_columns(displacements)
    targets = [c for c in displacements.columns if c.startswith("delta_") and c not in set(delta_coord)]
    groups = displacements["dataset_id"].astype(str) if "dataset_id" in displacements.columns else pd.Series(np.arange(len(displacements)), index=displacements.index).astype(str)
    rows = []
    for target in targets:
        valid = displacements[delta_coord + [target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 10 or valid[target].nunique() < 2:
            rows.append({"target": target, "cv_r2": np.nan, "mae": np.nan, "n_pairs": int(len(valid))})
            continue
        idx = valid.index
        cv = GroupKFold(n_splits=min(5, groups.loc[idx].nunique())) if groups.loc[idx].nunique() >= 2 else None
        X = valid[delta_coord]
        y = valid[target]
        best_r2, best_mae = -np.inf, np.nan
        for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=183, n_jobs=-1)]:
            pred = cross_val_predict(model, X, y, cv=cv, groups=groups.loc[idx]) if cv is not None else np.repeat(y.mean(), len(y))
            score = float(r2_score(y, pred)) if y.var() > 0 else np.nan
            if score > best_r2:
                best_r2, best_mae = score, float(mean_absolute_error(y, pred))
        rows.append({"target": target, "cv_r2": best_r2, "mae": best_mae, "n_pairs": int(len(valid))})
    return pd.DataFrame(rows)


def trajectory_analysis(df: pd.DataFrame) -> pd.DataFrame:
    required = {"trajectory_id", "intervention", "strength"}
    if not required.issubset(df.columns):
        return pd.DataFrame([{"intervention": "unavailable", "n_trajectories": 0, "mean_path_length": np.nan, "mean_curvature": np.nan, "mean_cliffiness_monotonicity": np.nan, "note": "No trajectory_id/intervention/strength columns available."}])
    coord_cols = coordinate_columns(df)
    rows = []
    for intervention, group in df.groupby("intervention"):
        lengths, curvatures, monotone = [], [], []
        for _, traj in group.sort_values("strength").groupby("trajectory_id"):
            if len(traj) < 2:
                continue
            pts = traj[coord_cols].to_numpy(dtype=float)
            steps = np.diff(pts, axis=0)
            lengths.append(float(np.linalg.norm(steps, axis=1).sum()))
            if len(steps) >= 2:
                cos = np.sum(steps[:-1] * steps[1:], axis=1) / np.maximum(1e-12, np.linalg.norm(steps[:-1], axis=1) * np.linalg.norm(steps[1:], axis=1))
                curvatures.append(float(np.mean(1 - cos)))
            if TARGET in traj.columns:
                diffs = np.diff(traj[TARGET].to_numpy(dtype=float))
                monotone.append(float(max(np.mean(diffs >= 0), np.mean(diffs <= 0))))
        rows.append({"intervention": intervention, "n_trajectories": len(lengths), "mean_path_length": float(np.mean(lengths)) if lengths else np.nan, "mean_curvature": float(np.mean(curvatures)) if curvatures else np.nan, "mean_cliffiness_monotonicity": float(np.mean(monotone)) if monotone else np.nan, "note": ""})
    return pd.DataFrame(rows)


def decision(consistency: pd.DataFrame, prediction: pd.DataFrame) -> tuple[str, str]:
    available = consistency[consistency["n_vectors"] > 0]
    if available.empty:
        return "blocked", "No paired intervention coordinate rows are available; intervention vector dynamics cannot be tested."
    mean_cos = float(available["mean_cosine_similarity"].mean())
    cliff = prediction[prediction["target"] == f"delta_{TARGETS[0]}"]
    cliff_r2 = float(cliff["cv_r2"].iloc[0]) if not cliff.empty and pd.notna(cliff["cv_r2"].iloc[0]) else np.nan
    if mean_cos >= 0.70 and cliff_r2 >= 0.75:
        return "strong_support", "Interventions induce consistent coordinate vectors and coordinate motion predicts accessibility changes."
    if mean_cos >= 0.45 or (pd.notna(cliff_r2) and cliff_r2 >= 0.40):
        return "moderate_support", "Intervention vectors show partial consistency or predictive value, with remaining context dependence."
    return "failure", "Intervention vectors are dispersed or fail to predict morphology changes."


def placeholder_plot(output: Path, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_vectors(displacements: pd.DataFrame, intervention: str, output: Path) -> Path:
    data = displacements[displacements["intervention"].astype(str).str.contains(intervention, case=False, na=False)] if not displacements.empty else pd.DataFrame()
    delta_cols = delta_coordinate_columns(data)
    if data.empty or len(delta_cols) < 2:
        return placeholder_plot(output, f"No paired {intervention} coordinate vectors available")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(np.zeros(len(data)), np.zeros(len(data)), data[delta_cols[0]], data[delta_cols[1]], angles="xy", scale_units="xy", scale=1, alpha=0.5)
    ax.set_xlabel(delta_cols[0]); ax.set_ylabel(delta_cols[1]); ax.set_title(f"{intervention.title()} Vector Field")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def write_report(input_coordinates: Path = DEFAULT_COORDINATES, output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_coordinates) if input_coordinates.exists() else pd.DataFrame()
    displacements = displacement_vectors(df) if not df.empty else pd.DataFrame()
    consistency = vector_consistency(displacements)
    prediction = morphology_prediction(displacements)
    trajectories = trajectory_analysis(df) if not df.empty else trajectory_analysis(pd.DataFrame())
    outcome, text = decision(consistency, prediction)
    summary = {"outcome": outcome, "outcome_text": text, "n_coordinate_rows": int(len(df)), "n_displacement_vectors": int(len(displacements)), "has_pairing_schema": bool(has_pairing_schema(df)) if not df.empty else False}
    report = output_dir / "accessibility_intervention_vector_hypothesis.md"
    summary_path = output_dir / "accessibility_intervention_vector_summary.json"
    displacement_path = output_dir / "accessibility_intervention_displacements.csv"
    consistency_path = output_dir / "accessibility_intervention_vector_consistency.csv"
    prediction_path = output_dir / "accessibility_intervention_morphology_prediction.csv"
    trajectory_path = output_dir / "accessibility_intervention_trajectories.csv"
    displacements.to_csv(displacement_path, index=False); consistency.to_csv(consistency_path, index=False); prediction.to_csv(prediction_path, index=False); trajectories.to_csv(trajectory_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text("\n".join(["# Accessibility Intervention Vector Hypothesis", "", "## Executive Answer", text, "", "## Vector Consistency", _markdown_table(consistency), "", "## Morphology Prediction", _markdown_table(prediction), "", "## Trajectories", _markdown_table(trajectories)]) + "\n", encoding="utf-8")
    return (
        report,
        summary_path,
        displacement_path,
        consistency_path,
        prediction_path,
        trajectory_path,
        plot_vectors(displacements, "dropout", output_dir / "vector_field_dropout.png"),
        plot_vectors(displacements, "bagging", output_dir / "vector_field_bagging.png"),
        plot_vectors(displacements, "weight", output_dir / "vector_field_weighting.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-coordinates", type=Path, default=DEFAULT_COORDINATES)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.input_coordinates, args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
