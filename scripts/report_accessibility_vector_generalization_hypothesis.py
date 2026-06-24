"""Test whether accessibility intervention vectors generalize beyond one intervention."""

from __future__ import annotations

import argparse
import json
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
DEFAULT_PAIRED = OUT / "accessibility_paired_interventions.csv"
EXPECTED_INTERVENTIONS = ["bagging", "dropout", "weighting", "oversampling", "platt_calibration", "isotonic_calibration"]
MORPH_TARGETS = ["minority_survival_cliffiness", "persistence", "minority_survival_auc", "breadth", "elevation"]


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            values.append(format(value, floatfmt) if isinstance(value, float) and pd.notna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def coord_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("coord_") and c.split("_")[-1].isdigit()]


def paired_displacements(paired: pd.DataFrame) -> pd.DataFrame:
    coords = coord_cols(paired)
    targets = [c for c in MORPH_TARGETS if c in paired.columns]
    rows = []
    for pair_id, group in paired.groupby("pair_id", sort=False):
        before = group[group["stage"].astype(str) == "before"]
        after = group[group["stage"].astype(str) == "after"]
        if before.empty or after.empty:
            # Continuous trajectories are handled by adjacent strength deltas.
            ordered = group.sort_values("strength")
            if len(ordered) < 2:
                continue
            pairs = zip(ordered.iloc[:-1].itertuples(index=False), ordered.iloc[1:].itertuples(index=False), strict=False)
        else:
            pairs = [(before.iloc[0], after.iloc[-1])]
        for b, a in pairs:
            b = pd.Series(b._asdict()) if hasattr(b, "_asdict") else b
            a = pd.Series(a._asdict()) if hasattr(a, "_asdict") else a
            row = {"pair_id": pair_id, "intervention": str(b["intervention"]), "dataset": b.get("dataset", b.get("dataset_id")), "dataset_id": b.get("dataset_id", b.get("dataset")), "task_id": b.get("task_id", ""), "seed": b.get("seed", np.nan), "split_id": b.get("split_id", np.nan), "base_model_id": b.get("base_model_id", b.get("model_id", "")), "before_model_id": b.get("model_id", ""), "after_model_id": a.get("model_id", ""), "before_strength": float(b.get("strength", 0.0)), "after_strength": float(a.get("strength", 1.0))}
            for col in coords:
                row[f"delta_{col}"] = float(a[col] - b[col])
                row[f"before_{col}"] = float(b[col])
                row[f"after_{col}"] = float(a[col])
            for target in targets:
                row[f"delta_{target}"] = float(a[target] - b[target])
                row[f"before_{target}"] = float(b[target])
                row[f"after_{target}"] = float(a[target])
            row["vector_magnitude"] = float(np.linalg.norm([row[f"delta_{c}"] for c in coords]))
            rows.append(row)
    return pd.DataFrame(rows)


def cosine_metrics(V: np.ndarray) -> tuple[float, float, float]:
    if len(V) < 2:
        return np.nan, np.nan, np.nan
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    U = V / np.maximum(norms, 1e-12)
    cos = U @ U.T
    tri = cos[np.triu_indices_from(cos, k=1)]
    centered = V - V.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    pc1 = float(s[0] ** 2 / max(1e-12, np.sum(s**2))) if len(s) else np.nan
    return float(np.mean(tri)), float(1 - np.mean(tri)), pc1


def intra_intervention_consistency(displacements: pd.DataFrame) -> pd.DataFrame:
    delta_cols = [c for c in displacements.columns if c.startswith("delta_coord_")]
    rows = []
    for intervention in EXPECTED_INTERVENTIONS:
        group = displacements[displacements["intervention"] == intervention]
        if group.empty:
            rows.append({"intervention": intervention, "available": False, "n_vectors": 0, "mean_cosine_similarity": np.nan, "angular_dispersion": np.nan, "pc1_vector_variance": np.nan, "magnitude_cv": np.nan})
            continue
        V = group[delta_cols].to_numpy(dtype=float)
        mean_cos, angular, pc1 = cosine_metrics(V)
        mags = np.linalg.norm(V, axis=1)
        rows.append({"intervention": intervention, "available": True, "n_vectors": int(len(group)), "mean_cosine_similarity": mean_cos, "angular_dispersion": angular, "pc1_vector_variance": pc1, "magnitude_cv": float(np.std(mags) / max(1e-12, np.mean(mags)))})
    return pd.DataFrame(rows)


def dataset_holdout_consistency(displacements: pd.DataFrame) -> pd.DataFrame:
    delta_cols = [c for c in displacements.columns if c.startswith("delta_coord_")]
    rows = []
    for intervention, group in displacements.groupby("intervention"):
        if group["dataset_id"].nunique() < 2:
            rows.append({"intervention": intervention, "heldout_dataset": "unavailable", "n_train": 0, "n_test": len(group), "cosine_to_train_mean": np.nan, "magnitude_ratio": np.nan})
            continue
        for dataset, test in group.groupby("dataset_id"):
            train = group[group["dataset_id"] != dataset]
            train_vec = train[delta_cols].to_numpy(dtype=float).mean(axis=0)
            test_vec = test[delta_cols].to_numpy(dtype=float).mean(axis=0)
            cosine = float(np.dot(train_vec, test_vec) / max(1e-12, np.linalg.norm(train_vec) * np.linalg.norm(test_vec)))
            rows.append({"intervention": intervention, "heldout_dataset": dataset, "n_train": int(len(train)), "n_test": int(len(test)), "cosine_to_train_mean": cosine, "magnitude_ratio": float(np.linalg.norm(test_vec) / max(1e-12, np.linalg.norm(train_vec)))})
    return pd.DataFrame(rows)


def vector_arithmetic(displacements: pd.DataFrame) -> pd.DataFrame:
    delta_cols = [c for c in displacements.columns if c.startswith("delta_coord_")]
    target_cols = [c for c in displacements.columns if c.startswith("delta_") and c not in set(delta_cols)]
    rows = []
    for target in target_cols:
        valid = displacements[delta_cols + [target, "dataset_id"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 10 or valid[target].nunique() < 2:
            rows.append({"target": target, "cv_r2": np.nan, "mae": np.nan, "n_vectors": int(len(valid))})
            continue
        groups = valid["dataset_id"].astype(str)
        cv = GroupKFold(n_splits=min(5, groups.nunique())) if groups.nunique() >= 2 else None
        X = valid[delta_cols]
        y = valid[target]
        best_r2, best_mae = -np.inf, np.nan
        for model in [make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)), RandomForestRegressor(n_estimators=140, min_samples_leaf=4, random_state=191, n_jobs=-1)]:
            pred = cross_val_predict(model, X, y, cv=cv, groups=groups) if cv is not None else np.repeat(y.mean(), len(y))
            score = float(r2_score(y, pred))
            if score > best_r2:
                best_r2, best_mae = score, float(mean_absolute_error(y, pred))
        rows.append({"target": target, "cv_r2": best_r2, "mae": best_mae, "n_vectors": int(len(valid))})
    return pd.DataFrame(rows)


def arithmetic_counterfactual_error(displacements: pd.DataFrame) -> pd.DataFrame:
    delta_cols = [c for c in displacements.columns if c.startswith("delta_coord_")]
    rows = []
    for intervention, group in displacements.groupby("intervention"):
        mean_v = group[delta_cols].to_numpy(dtype=float).mean(axis=0)
        before = group[[f"before_{c.replace('delta_', '')}" for c in delta_cols]].to_numpy(dtype=float)
        after = group[[f"after_{c.replace('delta_', '')}" for c in delta_cols]].to_numpy(dtype=float)
        pred_after = before + mean_v
        error = np.linalg.norm(pred_after - after, axis=1)
        actual_step = np.linalg.norm(after - before, axis=1)
        rows.append({"intervention": intervention, "n_vectors": int(len(group)), "mean_coordinate_prediction_error": float(error.mean()), "mean_actual_step_length": float(actual_step.mean()), "relative_error": float(error.mean() / max(1e-12, actual_step.mean()))})
    return pd.DataFrame(rows)


def decision(consistency: pd.DataFrame, holdout: pd.DataFrame, arithmetic: pd.DataFrame) -> tuple[str, str]:
    available = consistency[consistency["available"]]
    if available.empty:
        return "blocked", "No intervention vectors are available."
    n_interventions = int(available["intervention"].nunique())
    mean_cos = float(available["mean_cosine_similarity"].mean())
    holdout_cos = float(holdout["cosine_to_train_mean"].dropna().mean()) if not holdout.empty else np.nan
    cliff = arithmetic[arithmetic["target"] == "delta_minority_survival_cliffiness"]
    cliff_r2 = float(cliff["cv_r2"].iloc[0]) if not cliff.empty and pd.notna(cliff["cv_r2"].iloc[0]) else np.nan
    if n_interventions >= 2 and mean_cos >= 0.70 and holdout_cos >= 0.70 and cliff_r2 >= 0.75:
        return "strong_support", "Multiple interventions produce coherent, dataset-stable vectors and coordinate motion predicts accessibility changes."
    if mean_cos >= 0.70 and cliff_r2 >= 0.75:
        return "single_intervention_support", "The available intervention has coherent vectors, but generality beyond that intervention is not established."
    if mean_cos >= 0.45 or (pd.notna(cliff_r2) and cliff_r2 >= 0.40):
        return "moderate_support", "Intervention vectors show partial consistency or predictive value."
    return "failure", "Intervention vectors are not coherent or predictive."


def plot_consistency(consistency: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(consistency["intervention"], consistency["mean_cosine_similarity"].fillna(0), color=["steelblue" if a else "lightgray" for a in consistency["available"]])
    ax.axhline(0.70, color="green", linestyle="--")
    ax.set_ylabel("mean cosine similarity")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Intervention Vector Consistency")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def plot_holdout(holdout: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    data = holdout.dropna(subset=["cosine_to_train_mean"])
    ax.scatter(data["intervention"], data["cosine_to_train_mean"], alpha=0.7)
    ax.axhline(0.70, color="green", linestyle="--")
    ax.set_ylabel("held-out dataset cosine")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Cross-Dataset Vector Stability")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig); return output


def write_report(input_path: Path = DEFAULT_PAIRED, output_dir: Path = OUT) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(input_path) if input_path.exists() else pd.DataFrame()
    displacements = paired_displacements(paired) if not paired.empty else pd.DataFrame()
    consistency = intra_intervention_consistency(displacements) if not displacements.empty else intra_intervention_consistency(pd.DataFrame(columns=["intervention"]))
    holdout = dataset_holdout_consistency(displacements) if not displacements.empty else pd.DataFrame()
    arithmetic = vector_arithmetic(displacements) if not displacements.empty else pd.DataFrame()
    counterfactual = arithmetic_counterfactual_error(displacements) if not displacements.empty else pd.DataFrame()
    outcome, text = decision(consistency, holdout, arithmetic)
    summary = {"outcome": outcome, "outcome_text": text, "n_vectors": int(len(displacements)), "n_available_interventions": int(consistency["available"].sum()) if "available" in consistency else 0, "available_interventions": consistency.loc[consistency["available"], "intervention"].tolist() if "available" in consistency else []}
    report = output_dir / "accessibility_vector_generalization_hypothesis.md"
    summary_path = output_dir / "accessibility_vector_generalization_summary.json"
    disp_path = output_dir / "accessibility_vector_generalization_displacements.csv"
    consistency_path = output_dir / "accessibility_vector_generalization_consistency.csv"
    holdout_path = output_dir / "accessibility_vector_generalization_dataset_holdout.csv"
    arithmetic_path = output_dir / "accessibility_vector_generalization_arithmetic.csv"
    counterfactual_path = output_dir / "accessibility_vector_generalization_counterfactual_error.csv"
    displacements.to_csv(disp_path, index=False); consistency.to_csv(consistency_path, index=False); holdout.to_csv(holdout_path, index=False); arithmetic.to_csv(arithmetic_path, index=False); counterfactual.to_csv(counterfactual_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text("\n".join(["# Accessibility Vector Generalization Hypothesis", "", "## Executive Answer", text, "", "## Intra-Intervention Consistency", _markdown_table(consistency), "", "## Cross-Dataset Holdout", _markdown_table(holdout), "", "## Vector Arithmetic", _markdown_table(arithmetic), "", "## Counterfactual Coordinate Error", _markdown_table(counterfactual)]) + "\n", encoding="utf-8")
    return (report, summary_path, disp_path, consistency_path, holdout_path, arithmetic_path, counterfactual_path, plot_consistency(consistency, output_dir / "accessibility_vector_generalization_consistency.png"), plot_holdout(holdout, output_dir / "accessibility_vector_generalization_dataset_holdout.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for output in write_report(args.input, args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
