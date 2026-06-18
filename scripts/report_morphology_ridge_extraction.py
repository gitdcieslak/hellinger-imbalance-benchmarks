"""Extract allocation morphology ridges for accessibility outcomes."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "results" / "topology" / "density_threshold_sweep.csv",
    ROOT / "results" / "topology" / "density_separability_factorial_dense.csv",
    ROOT / "results" / "topology" / "fragmentation_sweep.csv",
    ROOT / "results" / "topology" / "weighted_dropout_sweep_dense.csv",
    ROOT / "results" / "topology" / "mlp_objectives_topology.csv",
    ROOT / "results" / "topology" / "allocation_trajectory.csv",
]
TARGETS = ["minority_survival_auc", "minority_survival_cliffiness"]
REQUIRED_BASE = ["breadth", "elevation", *TARGETS]
FEATURE_SETS = {
    "conventional": (["auroc", "average_precision"], []),
    "identity": ([], ["experiment_family", "model"]),
    "morphology": (["breadth", "elevation"], []),
    "morphology_conventional": (["breadth", "elevation", "auroc", "average_precision"], []),
    "full": (["breadth", "elevation", "auroc", "average_precision"], ["experiment_family", "model"]),
}


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join([str(c) for c in df.columns]) + " |"]
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float) or isinstance(value, np.floating):
                values.append(format(float(value), floatfmt) if pd.notna(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def experiment_family_from_path(path: Path) -> str:
    name = path.stem
    if name == "density_separability_factorial_dense":
        return "density_separability"
    return name.replace("_sweep", "")


def normalize_result_file(path: Path) -> tuple[pd.DataFrame | None, str]:
    if not path.exists():
        return None, f"missing: {path}"
    df = pd.read_csv(path)
    family = experiment_family_from_path(path)
    column_map = {}
    if "breadth" not in df.columns and "positive_histogram_entropy" in df.columns:
        column_map["positive_histogram_entropy"] = "breadth"
    if "elevation" not in df.columns and "positive_top_bin_mass" in df.columns:
        column_map["positive_top_bin_mass"] = "elevation"
    df = df.rename(columns=column_map)

    model_col = next((col for col in ["model_id", "objective", "model_name"] if col in df.columns), None)
    missing = [col for col in REQUIRED_BASE if col not in df.columns]
    if model_col is None and "dropout_rate" not in df.columns:
        missing.append("model_id/objective/model_name")
    if missing:
        return None, f"skipped incomplete: {path} missing {', '.join(missing)}"

    out = pd.DataFrame(
        {
            "experiment_family": family,
            "model": df[model_col].astype(str) if model_col is not None else "dropout_" + df["dropout_rate"].astype(str),
            "seed": df["seed"] if "seed" in df.columns else np.arange(len(df)),
            "breadth": pd.to_numeric(df["breadth"], errors="coerce"),
            "elevation": pd.to_numeric(df["elevation"], errors="coerce"),
            "minority_survival_auc": pd.to_numeric(df["minority_survival_auc"], errors="coerce"),
            "minority_survival_cliffiness": pd.to_numeric(df["minority_survival_cliffiness"], errors="coerce"),
            "auroc": pd.to_numeric(df["auroc"], errors="coerce") if "auroc" in df.columns else np.nan,
            "average_precision": pd.to_numeric(df["average_precision"], errors="coerce") if "average_precision" in df.columns else np.nan,
        }
    )
    out = out.dropna(subset=["breadth", "elevation", *TARGETS]).copy()
    out["seed"] = out["seed"].astype(str)
    if family == "mlp_objectives_topology":
        out = out.drop_duplicates(subset=["experiment_family", "model", "seed"])
    else:
        out = out.drop_duplicates()
    return out, f"used: {path} rows={len(out)}"


def load_inputs(paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    messages = []
    for path in paths:
        frame, message = normalize_result_file(path)
        messages.append(message)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["experiment_family", "model", "seed", *REQUIRED_BASE, "auroc", "average_precision"]), messages
    return pd.concat(frames, ignore_index=True).drop_duplicates(), messages


def _make_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers, remainder="drop")),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def _cv_strategy(df: pd.DataFrame):
    family_count = df["experiment_family"].nunique()
    if family_count >= 3:
        n_splits = min(5, family_count)
        return GroupKFold(n_splits=n_splits), df["experiment_family"], "experiment_family"
    seed_count = df["seed"].nunique()
    if seed_count >= 3:
        n_splits = min(5, seed_count)
        return GroupKFold(n_splits=n_splits), df["seed"], "seed"
    n_splits = min(5, len(df))
    return KFold(n_splits=n_splits, shuffle=True, random_state=7), None, "row"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fit_predictive_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for target in TARGETS:
        for feature_set, (num_features, cat_features) in FEATURE_SETS.items():
            needed = [*num_features, *cat_features, target]
            data = df.dropna(subset=needed).copy()
            if len(data) < 3:
                rows.append({"target": target, "feature_set": feature_set, "cv_group": "n/a", "cv_r2": np.nan, "cv_mae": np.nan, "cv_rmse": np.nan, "in_sample_r2": np.nan, "n_rows": len(data)})
                continue
            pipeline = _make_pipeline(num_features, cat_features)
            cv, groups, group_name = _cv_strategy(data)
            y = data[target].to_numpy(dtype=float)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if groups is None:
                        pred = cross_val_predict(pipeline, data, y, cv=cv)
                    else:
                        pred = cross_val_predict(pipeline, data, y, cv=cv, groups=groups)
            except Exception:
                pred = np.full_like(y, float(np.mean(y)), dtype=float)
                group_name = "mean_fallback"
            pipeline.fit(data, y)
            in_pred = pipeline.predict(data)
            rows.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "cv_group": group_name,
                    "cv_r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else 0.0,
                    "cv_mae": float(mean_absolute_error(y, pred)),
                    "cv_rmse": _rmse(y, pred),
                    "in_sample_r2": float(r2_score(y, in_pred)) if len(np.unique(y)) > 1 else 0.0,
                    "n_rows": len(data),
                }
            )
            if feature_set == "morphology":
                key = f"{target}_morphology"
                full_pred = np.full(len(df), np.nan, dtype=float)
                full_pred[data.index.to_numpy()] = pred
                predictions[key] = full_pred
    return pd.DataFrame(rows), predictions


def morphology_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        data = df.dropna(subset=["breadth", "elevation", target]).copy()
        if len(data) < 2:
            continue
        pipeline = _make_pipeline(["breadth", "elevation"], [])
        pipeline.fit(data, data[target].to_numpy(dtype=float))
        coefs = pipeline.named_steps["ridge"].coef_
        rows.append({"target": target, "feature": "breadth", "standardized_coefficient": float(coefs[0])})
        rows.append({"target": target, "feature": "elevation", "standardized_coefficient": float(coefs[1])})
    return pd.DataFrame(rows)


def sign_stability(df: pd.DataFrame, n_bootstrap: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for target in TARGETS:
        data = df.dropna(subset=["breadth", "elevation", target]).copy()
        if len(data) < 5:
            continue
        signs = {"breadth": [], "elevation": []}
        for _ in range(n_bootstrap):
            sample_idx = rng.choice(np.arange(len(data)), size=len(data), replace=True)
            sample = data.iloc[sample_idx]
            pipeline = _make_pipeline(["breadth", "elevation"], [])
            pipeline.fit(sample, sample[target].to_numpy(dtype=float))
            coef = pipeline.named_steps["ridge"].coef_
            signs["breadth"].append(np.sign(coef[0]))
            signs["elevation"].append(np.sign(coef[1]))
        for feature, arr in signs.items():
            values = np.asarray(arr)
            nonzero = values[values != 0]
            dominant = 0 if nonzero.size == 0 else int(np.sign(np.mean(nonzero)))
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "dominant_sign": dominant,
                    "dominant_sign_fraction": float(np.mean(values == dominant)) if dominant != 0 else float(np.mean(values == 0)),
                    "positive_fraction": float(np.mean(values > 0)),
                    "negative_fraction": float(np.mean(values < 0)),
                }
            )
    return pd.DataFrame(rows)


def ridge_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    data = df.dropna(subset=["breadth", "elevation"])
    if len(data) < 2:
        return pd.DataFrame([{"diagnostic": "insufficient_rows", "value": float(len(data))}])
    X = data[["breadth", "elevation"]].to_numpy(dtype=float)
    corr = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]) if np.std(X[:, 0]) > 0 and np.std(X[:, 1]) > 0 else 0.0
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2).fit(Xs)
    pc1 = pca.components_[0]
    slope = float(pc1[1] / pc1[0]) if abs(pc1[0]) > 1e-12 else np.inf
    residual_variance = float(pca.explained_variance_ratio_[1])
    return pd.DataFrame(
        [
            {"diagnostic": "breadth_elevation_correlation", "value": corr},
            {"diagnostic": "pc1_explained_variance_ratio", "value": float(pca.explained_variance_ratio_[0])},
            {"diagnostic": "pc2_residual_variance_ratio", "value": residual_variance},
            {"diagnostic": "approximate_ridge_slope_standardized", "value": slope},
            {"diagnostic": "near_one_dimensional_ridge", "value": bool(pca.explained_variance_ratio_[0] >= 0.80)},
        ]
    )


def per_family_holdout(df: pd.DataFrame) -> pd.DataFrame:
    if df["experiment_family"].nunique() < 3:
        return pd.DataFrame()
    rows = []
    for target in TARGETS:
        for family in sorted(df["experiment_family"].unique()):
            train = df[df["experiment_family"] != family].dropna(subset=["breadth", "elevation", target])
            test = df[df["experiment_family"] == family].dropna(subset=["breadth", "elevation", target])
            if len(train) < 2 or test.empty:
                continue
            pipeline = _make_pipeline(["breadth", "elevation"], [])
            pipeline.fit(train, train[target].to_numpy(dtype=float))
            pred = pipeline.predict(test)
            y = test[target].to_numpy(dtype=float)
            rows.append({"target": target, "holdout_family": family, "r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else 0.0, "mae": float(mean_absolute_error(y, pred)), "rmse": _rmse(y, pred), "n_test": len(test)})
    return pd.DataFrame(rows)


def interpretation(df: pd.DataFrame, perf: pd.DataFrame, coefs: pd.DataFrame, stability: pd.DataFrame, diagnostics: pd.DataFrame, holdout: pd.DataFrame) -> list[str]:
    lines = []
    for target, question in [
        ("minority_survival_auc", "Can breadth/elevation predict survival AUC across experiment families?"),
        ("minority_survival_cliffiness", "Can breadth/elevation predict cliffiness across experiment families?"),
    ]:
        row = perf[(perf["target"] == target) & (perf["feature_set"] == "morphology")]
        cv_r2 = float(row.iloc[0]["cv_r2"]) if not row.empty and pd.notna(row.iloc[0]["cv_r2"]) else np.nan
        lines.append(f"- {question} {'Yes' if pd.notna(cv_r2) and cv_r2 > 0.25 else 'Weak or inconclusive'}; morphology-only CV R2={cv_r2:.4f}.")
    better_rows = []
    for target in TARGETS:
        morph = perf[(perf["target"] == target) & (perf["feature_set"] == "morphology")]
        conv = perf[(perf["target"] == target) & (perf["feature_set"] == "conventional")]
        if not morph.empty and not conv.empty:
            better_rows.append(float(morph.iloc[0]["cv_r2"]) >= float(conv.iloc[0]["cv_r2"]))
    lines.append(f"- Do morphology features outperform AUROC/AP for accessibility outcomes? {'Yes' if better_rows and all(better_rows) else 'Mixed or no'} across the available targets.")
    improve_rows = []
    for target in TARGETS:
        morph = perf[(perf["target"] == target) & (perf["feature_set"] == "morphology")]
        combo = perf[(perf["target"] == target) & (perf["feature_set"] == "morphology_conventional")]
        if not morph.empty and not combo.empty:
            improve_rows.append(float(combo.iloc[0]["cv_r2"]) > float(morph.iloc[0]["cv_r2"]) + 0.02)
    lines.append(f"- Does adding AUROC/AP improve over morphology alone? {'Yes' if improve_rows and any(improve_rows) else 'No material improvement'} by a +0.02 CV R2 threshold.")
    stable = stability["dominant_sign_fraction"].min() if not stability.empty else np.nan
    lines.append(f"- Are breadth/elevation coefficient signs stable? {'Yes' if pd.notna(stable) and stable >= 0.80 else 'Mixed'}; minimum dominant-sign fraction={stable:.4f}.")
    pc1 = diagnostics[diagnostics["diagnostic"] == "pc1_explained_variance_ratio"]
    pc1_value = float(pc1.iloc[0]["value"]) if not pc1.empty else np.nan
    lines.append(f"- Does the morphology space look approximately one-dimensional? {'Yes' if pd.notna(pc1_value) and pc1_value >= 0.80 else 'No'}; PC1 explained variance={pc1_value:.4f}.")
    if not holdout.empty:
        worst = holdout.sort_values("rmse", ascending=False).iloc[0]
        lines.append(f"- Which experiment family has the largest residuals? {worst['holdout_family']} for {worst['target']} by holdout RMSE={float(worst['rmse']):.4f}.")
    else:
        by_family = df.groupby("experiment_family")[TARGETS].count().sum(axis=1).sort_values(ascending=False)
        family = by_family.index[0] if not by_family.empty else "n/a"
        lines.append(f"- Which experiment family has the largest residuals? Not estimable by family holdout with fewer than three families; largest available family is {family}.")
    lines.append("- Are density, fragmentation, and dropout perturbations moving through a common morphology space? Treat as supported only if morphology-only CV performance is competitive and family holdout residuals are not dominated by a single experiment family; see tables above.")
    return lines


def plot_predicted_actual(df: pd.DataFrame, predictions: dict[str, np.ndarray], target: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred = predictions.get(f"{target}_morphology", np.full(len(df), np.nan))
    plot_df = df.assign(predicted=pred).dropna(subset=[target, "predicted"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for family, group in plot_df.groupby("experiment_family"):
        ax.scatter(group[target], group["predicted"], label=family, alpha=0.75)
    if not plot_df.empty:
        lo = float(min(plot_df[target].min(), plot_df["predicted"].min()))
        hi = float(max(plot_df[target].max(), plot_df["predicted"].max()))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title(f"Morphology ridge predicted vs actual: {target}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_morphology_space(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    for idx, (family, group) in enumerate(df.groupby("experiment_family")):
        sc = ax.scatter(group["breadth"], group["elevation"], c=group["minority_survival_auc"], marker=markers[idx % len(markers)], alpha=0.75, label=family, cmap="viridis")
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Morphology space by experiment family")
    ax.legend(fontsize=7)
    fig.colorbar(sc, ax=ax, label="minority_survival_auc")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_residuals(df: pd.DataFrame, predictions: dict[str, np.ndarray], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred = predictions.get("minority_survival_auc_morphology", np.full(len(df), np.nan))
    plot_df = df.assign(residual=df["minority_survival_auc"] - pred).dropna(subset=["residual"])
    fig, ax = plt.subplots(figsize=(8, 5))
    families = sorted(plot_df["experiment_family"].unique())
    data = [plot_df[plot_df["experiment_family"] == family]["residual"].to_numpy(dtype=float) for family in families]
    ax.boxplot(data, tick_labels=families, orientation="vertical")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("survival AUC residual")
    ax.set_title("Morphology-only residuals by family")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def build_report(df: pd.DataFrame, messages: list[str], perf: pd.DataFrame, coefs: pd.DataFrame, stability: pd.DataFrame, diagnostics: pd.DataFrame, holdout: pd.DataFrame) -> str:
    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(df)},
            {"metric": "experiment_families", "value": df["experiment_family"].nunique() if not df.empty else 0},
            {"metric": "models", "value": df["model"].nunique() if not df.empty else 0},
            {"metric": "seeds", "value": df["seed"].nunique() if not df.empty else 0},
        ]
    )
    lines = [
        "# Morphology Ridge Extraction",
        "",
        "## Inputs Used",
        *[f"- {message}" for message in messages],
        "",
        "## Normalized Dataset Summary",
        _markdown_table(summary),
        "",
        "## Predictive Performance",
        _markdown_table(perf),
        "",
        "## Morphology Coefficients",
        _markdown_table(coefs),
        "",
        "## Sign Stability",
        _markdown_table(stability),
        "",
        "## Morphology Ridge Diagnostics",
        _markdown_table(diagnostics),
        "",
        "## Per-Family Holdout Results",
        _markdown_table(holdout),
        "",
        "## Interpretation",
        *interpretation(df, perf, coefs, stability, diagnostics, holdout),
        "",
        "## Limitations",
        "- Ridge results are associational diagnostics, not causal morphology theory.",
        "- Available experiment families may be imbalanced in row count and generated under different protocols.",
        "- Missing input files are skipped, so conclusions are conditional on the available local artifacts.",
        "",
        "## Next Steps",
        "- Re-run after all target experiment families are present.",
        "- Inspect high-residual families to identify morphology coordinates that require additional state variables.",
        "- Compare morphology-only ridge results with nonlinear smoothers if ridge performance is weak.",
    ]
    return "\n".join(lines) + "\n"


def write_report(input_paths: list[Path], output_md: Path) -> tuple[Path, ...]:
    df, messages = load_inputs(input_paths)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        output_md.write_text(build_report(df, messages, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()), encoding="utf-8")
        return (output_md,)
    perf, predictions = fit_predictive_models(df)
    coefs = morphology_coefficients(df)
    stability = sign_stability(df)
    diagnostics = ridge_diagnostics(df)
    holdout = per_family_holdout(df)
    output_md.write_text(build_report(df, messages, perf, coefs, stability, diagnostics, holdout), encoding="utf-8")
    paths = [
        output_md,
        plot_predicted_actual(df, predictions, "minority_survival_auc", output_md.parent / "morphology_ridge_survival_predicted_vs_actual.png"),
        plot_predicted_actual(df, predictions, "minority_survival_cliffiness", output_md.parent / "morphology_ridge_cliffiness_predicted_vs_actual.png"),
        plot_morphology_space(df, output_md.parent / "morphology_ridge_space_by_family.png"),
        plot_residuals(df, predictions, output_md.parent / "morphology_ridge_residuals_by_family.png"),
    ]
    return tuple(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "topology" / "morphology_ridge_extraction_summary.md")
    args = parser.parse_args()
    for path in write_report(args.inputs, args.output_md):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
