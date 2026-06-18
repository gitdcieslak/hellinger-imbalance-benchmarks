"""Test whether topology augments allocation morphology for accessibility dynamics."""

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
from sklearn.impute import SimpleImputer
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
TOPOLOGY_FEATURES = [
    "n_islands",
    "minority_cluster_count",
    "positives_per_cluster",
    "largest_island_fraction",
    "island_entropy",
    "island_gini_or_concentration",
    "n_components",
    "giant_component_fraction",
    "isolated_positive_fraction",
    "component_entropy",
    "mean_component_size",
    "median_component_size",
]
KEY_COEFFICIENTS = [
    "breadth",
    "elevation",
    "n_islands",
    "positives_per_cluster",
    "largest_island_fraction",
    "island_entropy",
    "isolated_positive_fraction",
    "component_entropy",
    "giant_component_fraction",
]
FEATURE_SETS = {
    "conventional": (["auroc", "average_precision"], []),
    "morphology": (["breadth", "elevation"], []),
    "topology": (TOPOLOGY_FEATURES, []),
    "morphology_topology": (["breadth", "elevation", *TOPOLOGY_FEATURES], []),
    "full": (["breadth", "elevation", *TOPOLOGY_FEATURES, "auroc", "average_precision"], ["experiment_family", "model_id"]),
}


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float) or isinstance(value, np.floating):
                vals.append(format(float(value), floatfmt) if pd.notna(value) else "nan")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def experiment_family_from_path(path: Path) -> str:
    name = path.stem
    return {"density_separability_factorial_dense": "density_separability"}.get(name, name.replace("_sweep", ""))


def derive_fragmentation_topology(df: pd.DataFrame, messages: list[str]) -> pd.DataFrame:
    out = df.copy()
    if "n_islands" not in out.columns and "minority_cluster_count" in out.columns:
        out["n_islands"] = out["minority_cluster_count"]
    if "n_islands" in out.columns:
        if "minority_count" not in out.columns:
            if "positives_per_cluster" in out.columns:
                out["minority_count"] = pd.to_numeric(out["n_islands"], errors="coerce") * pd.to_numeric(out["positives_per_cluster"], errors="coerce")
                messages.append("inferred minority_count from n_islands * positives_per_cluster")
            else:
                out["minority_count"] = 100.0
                messages.append("assumed minority_count=100 for topology derivation")
        if "positives_per_cluster" not in out.columns:
            out["positives_per_cluster"] = pd.to_numeric(out["minority_count"], errors="coerce") / pd.to_numeric(out["n_islands"], errors="coerce")
        n = pd.to_numeric(out["n_islands"], errors="coerce")
        minority_count = pd.to_numeric(out["minority_count"], errors="coerce")
        ppc = pd.to_numeric(out["positives_per_cluster"], errors="coerce")
        out["minority_cluster_count"] = out.get("minority_cluster_count", n)
        out["largest_island_fraction"] = ppc / minority_count
        out["island_entropy"] = np.log(n.clip(lower=1))
        out["island_gini_or_concentration"] = 1.0 / n.replace(0, np.nan)
    return out


def normalize_result_file(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    messages: list[str] = []
    if not path.exists():
        return None, [f"missing: {path}"]
    raw = pd.read_csv(path)
    family = experiment_family_from_path(path)
    raw = raw.rename(columns={"positive_histogram_entropy": "breadth", "positive_top_bin_mass": "elevation"})
    raw = derive_fragmentation_topology(raw, messages)
    model_col = next((col for col in ["model_id", "objective", "model_name"] if col in raw.columns), None)
    missing = [col for col in ["breadth", "elevation", *TARGETS] if col not in raw.columns]
    if model_col is None and "dropout_rate" not in raw.columns:
        missing.append("model_id/objective/model_name")
    if missing:
        return None, [f"skipped incomplete: {path} missing {', '.join(missing)}", *messages]
    model = raw[model_col].astype(str) if model_col is not None else "dropout_" + raw["dropout_rate"].astype(str)
    out = pd.DataFrame(
        {
            "experiment_family": family,
            "model_id": model,
            "seed": raw["seed"].astype(str) if "seed" in raw.columns else np.arange(len(raw)).astype(str),
            "breadth": pd.to_numeric(raw["breadth"], errors="coerce"),
            "elevation": pd.to_numeric(raw["elevation"], errors="coerce"),
            "minority_survival_auc": pd.to_numeric(raw["minority_survival_auc"], errors="coerce"),
            "minority_survival_cliffiness": pd.to_numeric(raw["minority_survival_cliffiness"], errors="coerce"),
            "auroc": pd.to_numeric(raw["auroc"], errors="coerce") if "auroc" in raw.columns else np.nan,
            "average_precision": pd.to_numeric(raw["average_precision"], errors="coerce") if "average_precision" in raw.columns else np.nan,
        }
    )
    for feature in TOPOLOGY_FEATURES:
        out[feature] = pd.to_numeric(raw[feature], errors="coerce") if feature in raw.columns else np.nan
    out = out.dropna(subset=["breadth", "elevation", *TARGETS]).copy()
    if family == "mlp_objectives_topology":
        out = out.drop_duplicates(subset=["experiment_family", "model_id", "seed"])
    else:
        out = out.drop_duplicates()
    return out, [f"used: {path} rows={len(out)}", *messages]


def load_inputs(paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    frames, messages = [], []
    for path in paths:
        frame, msg = normalize_result_file(path)
        messages.extend(msg)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), messages
    return pd.concat(frames, ignore_index=True).drop_duplicates(), messages


def make_pipeline(num: list[str], cat: list[str]) -> Pipeline:
    transformers = []
    if num:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), num))
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return Pipeline([("preprocess", ColumnTransformer(transformers, remainder="drop")), ("ridge", Ridge(alpha=1.0))])


def cv_strategy(df: pd.DataFrame):
    if df["experiment_family"].nunique() >= 3:
        return GroupKFold(min(5, df["experiment_family"].nunique())), df["experiment_family"], "experiment_family"
    if df["seed"].nunique() >= 3:
        return GroupKFold(min(5, df["seed"].nunique())), df["seed"], "seed"
    return KFold(min(5, len(df)), shuffle=True, random_state=7), None, "row"


def rmse(y, pred) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def fit_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows, predictions = [], {}
    for target in TARGETS:
        for name, (num, cat) in FEATURE_SETS.items():
            needed = [target, *cat]
            data = df.dropna(subset=needed).copy()
            if len(data) < 3:
                rows.append({"target": target, "feature_set": name, "cv_group": "n/a", "cv_r2": np.nan, "cv_mae": np.nan, "cv_rmse": np.nan, "in_sample_r2": np.nan, "n_rows": len(data)})
                continue
            y = data[target].to_numpy(float)
            pipe = make_pipeline(num, cat)
            cv, groups, group_name = cv_strategy(data)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = cross_val_predict(pipe, data, y, cv=cv, groups=groups) if groups is not None else cross_val_predict(pipe, data, y, cv=cv)
            except Exception:
                pred = np.full_like(y, y.mean())
                group_name = "mean_fallback"
            pipe.fit(data, y)
            ins = pipe.predict(data)
            rows.append({"target": target, "feature_set": name, "cv_group": group_name, "cv_r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else 0.0, "cv_mae": float(mean_absolute_error(y, pred)), "cv_rmse": rmse(y, pred), "in_sample_r2": float(r2_score(y, ins)) if len(np.unique(y)) > 1 else 0.0, "n_rows": len(data)})
            if name in {"morphology", "morphology_topology"}:
                full = np.full(len(df), np.nan)
                full[data.index.to_numpy()] = pred
                predictions[f"{target}_{name}"] = full
    return pd.DataFrame(rows), predictions


def improvement_classification(delta: float) -> str:
    if delta < 0.10:
        return "weak support"
    if delta < 0.30:
        return "moderate support"
    return "strong support"


def improvement_table(perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        morph = perf[(perf.target == target) & (perf.feature_set == "morphology")]
        aug = perf[(perf.target == target) & (perf.feature_set == "morphology_topology")]
        if morph.empty or aug.empty:
            continue
        delta = float(aug.iloc[0].cv_r2) - float(morph.iloc[0].cv_r2)
        rows.append({"target": target, "morphology_cv_r2": float(morph.iloc[0].cv_r2), "morphology_topology_cv_r2": float(aug.iloc[0].cv_r2), "delta_cv_r2": delta, "support_level": improvement_classification(delta)})
    return pd.DataFrame(rows)


def coefficients(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    num = ["breadth", "elevation", *TOPOLOGY_FEATURES]
    for target in TARGETS:
        data = df.dropna(subset=[target]).copy()
        if len(data) < 2:
            continue
        pipe = make_pipeline(num, [])
        pipe.fit(data, data[target].to_numpy(float))
        feats = num
        coefs = pipe.named_steps["ridge"].coef_
        for feat, coef in zip(feats, coefs, strict=False):
            if feat in KEY_COEFFICIENTS:
                rows.append({"target": target, "feature": feat, "standardized_coefficient": float(coef)})
    return pd.DataFrame(rows)


def sign_stability(df: pd.DataFrame, bootstrap_samples: int = 100, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    num = ["breadth", "elevation", *TOPOLOGY_FEATURES]
    rows = []
    for target in TARGETS:
        data = df.dropna(subset=[target]).copy()
        if len(data) < 5:
            continue
        signs = {feat: [] for feat in KEY_COEFFICIENTS if feat in num}
        for _ in range(bootstrap_samples):
            sample = data.iloc[rng.choice(np.arange(len(data)), size=len(data), replace=True)]
            pipe = make_pipeline(num, [])
            pipe.fit(sample, sample[target].to_numpy(float))
            for feat, coef in zip(num, pipe.named_steps["ridge"].coef_, strict=False):
                if feat in signs:
                    signs[feat].append(np.sign(coef))
        for feat, values in signs.items():
            arr = np.asarray(values)
            nonzero = arr[arr != 0]
            dominant = 0 if nonzero.size == 0 else int(np.sign(np.mean(nonzero)))
            rows.append({"target": target, "feature": feat, "dominant_sign": dominant, "dominant_sign_fraction": float(np.mean(arr == dominant)) if dominant else float(np.mean(arr == 0)), "positive_fraction": float(np.mean(arr > 0)), "negative_fraction": float(np.mean(arr < 0))})
    return pd.DataFrame(rows)


def topology_coverage(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"feature": f, "non_null_rows": int(df[f].notna().sum()), "coverage_fraction": float(df[f].notna().mean())} for f in TOPOLOGY_FEATURES if f in df.columns])


def per_family_holdout(df: pd.DataFrame) -> pd.DataFrame:
    if df.experiment_family.nunique() < 3:
        return pd.DataFrame()
    rows = []
    for target in TARGETS:
        for fs, (num, cat) in {"morphology": FEATURE_SETS["morphology"], "morphology_topology": FEATURE_SETS["morphology_topology"]}.items():
            for fam in sorted(df.experiment_family.unique()):
                train = df[df.experiment_family != fam].dropna(subset=[target])
                test = df[df.experiment_family == fam].dropna(subset=[target])
                if len(train) < 2 or test.empty:
                    continue
                pipe = make_pipeline(num, cat)
                pipe.fit(train, train[target].to_numpy(float))
                pred = pipe.predict(test)
                y = test[target].to_numpy(float)
                rows.append({"target": target, "feature_set": fs, "holdout_family": fam, "r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else 0.0, "mae": float(mean_absolute_error(y, pred)), "rmse": rmse(y, pred), "n_test": len(test)})
    return pd.DataFrame(rows)


def residual_improvement_by_family(holdout: pd.DataFrame) -> pd.DataFrame:
    if holdout.empty:
        return pd.DataFrame()
    morph = holdout[holdout.feature_set == "morphology"]
    aug = holdout[holdout.feature_set == "morphology_topology"]
    merged = morph.merge(aug, on=["target", "holdout_family"], suffixes=("_morphology", "_morphology_topology"))
    merged["rmse_improvement"] = merged["rmse_morphology"] - merged["rmse_morphology_topology"]
    merged["mae_improvement"] = merged["mae_morphology"] - merged["mae_morphology_topology"]
    return merged[["target", "holdout_family", "rmse_morphology", "rmse_morphology_topology", "rmse_improvement", "mae_improvement"]]


def plot_predicted(df, preds, target, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    p = preds.get(f"{target}_morphology_topology", np.full(len(df), np.nan))
    data = df.assign(predicted=p).dropna(subset=[target, "predicted"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for fam, g in data.groupby("experiment_family"):
        ax.scatter(g[target], g.predicted, label=fam, alpha=0.75)
    if not data.empty:
        lo, hi = min(data[target].min(), data.predicted.min()), max(data[target].max(), data.predicted.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title(target)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_residuals(df, preds, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    p = preds.get("minority_survival_cliffiness_morphology_topology", np.full(len(df), np.nan))
    data = df.assign(residual=df.minority_survival_cliffiness - p).dropna(subset=["residual"])
    fams = sorted(data.experiment_family.unique())
    vals = [data[data.experiment_family == f].residual.to_numpy(float) for f in fams]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(vals, tick_labels=fams, orientation="vertical")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("cliffiness residual")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_improvement(improvement, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    data = improvement[improvement.target == "minority_survival_cliffiness"] if not improvement.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not data.empty:
        x = np.arange(len(data))
        ax.bar(x - 0.2, data.rmse_morphology, width=0.4, label="morphology")
        ax.bar(x + 0.2, data.rmse_morphology_topology, width=0.4, label="morphology + topology")
        ax.set_xticks(x, data.holdout_family, rotation=25, ha="right")
    ax.set_ylabel("holdout RMSE")
    ax.set_title("Cliffiness residual improvement by family")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def interpretation(perf, imp, coefs, stability, residuals) -> list[str]:
    cliff = imp[imp.target == "minority_survival_cliffiness"]
    surv = imp[imp.target == "minority_survival_auc"]
    cliff_delta = float(cliff.iloc[0].delta_cv_r2) if not cliff.empty else np.nan
    surv_delta = float(surv.iloc[0].delta_cv_r2) if not surv.empty else np.nan
    cliff_assoc = coefs[coefs.target == "minority_survival_cliffiness"].copy()
    top_var = "n/a" if cliff_assoc.empty else cliff_assoc.assign(abscoef=cliff_assoc.standardized_coefficient.abs()).sort_values("abscoef", ascending=False).iloc[0].feature
    stable = stability.dominant_sign_fraction.min() if not stability.empty else np.nan
    frag = residuals[(residuals.target == "minority_survival_cliffiness") & (residuals.holdout_family == "fragmentation")]
    concentrated = bool(not frag.empty and frag.iloc[0].rmse_improvement > residuals[residuals.target == "minority_survival_cliffiness"].rmse_improvement.median()) if not residuals.empty else False
    poor = "n/a" if residuals.empty else ", ".join(residuals.sort_values("rmse_morphology_topology", ascending=False).head(3).holdout_family.astype(str).unique())
    return [
        f"- Does topology improve prediction of cliffiness beyond breadth/elevation? {improvement_classification(cliff_delta) if pd.notna(cliff_delta) else 'not estimable'}; delta CV R2={cliff_delta:.4f}.",
        f"- Does topology improve prediction of survival AUC beyond breadth/elevation? {improvement_classification(surv_delta) if pd.notna(surv_delta) else 'not estimable'}; delta CV R2={surv_delta:.4f}.",
        f"- Is the improvement concentrated in the fragmentation family? {'Yes' if concentrated else 'No or mixed'} based on per-family cliffiness RMSE improvement.",
        f"- Which topology variable is most associated with cliffiness? {top_var} by absolute standardized coefficient in the morphology+topology ridge model.",
        f"- Are topology coefficient signs stable? {'Yes' if pd.notna(stable) and stable >= 0.80 else 'Mixed'}; minimum dominant-sign fraction={stable:.4f}.",
        f"- Does this support Accessibility Level ~= Morphology and Accessibility Dynamics ~= Morphology + Topology? {'Yes' if pd.notna(cliff_delta) and cliff_delta >= 0.10 else 'Weak support'} for dynamics; compare survival/cliffiness improvement tables above.",
        f"- Which experiment families remain poorly explained? {poor} by morphology+topology holdout RMSE.",
    ]


def build_report(df, messages, coverage, perf, imp, coefs, stability, holdout, residuals) -> str:
    summary = pd.DataFrame([{"metric": "rows", "value": len(df)}, {"metric": "experiment_families", "value": df.experiment_family.nunique() if not df.empty else 0}, {"metric": "models", "value": df.model_id.nunique() if not df.empty else 0}, {"metric": "seeds", "value": df.seed.nunique() if not df.empty else 0}])
    lines = [
        "# Topology-Augmented Morphology",
        "",
        "## Inputs Used",
        *[f"- {m}" for m in messages],
        "",
        "## Normalized Dataset Summary",
        _markdown_table(summary),
        "",
        "## Topology Feature Coverage",
        _markdown_table(coverage),
        "",
        "## Predictive Performance",
        _markdown_table(perf),
        "",
        "## Cliffiness Improvement Test",
        _markdown_table(imp[imp.target == "minority_survival_cliffiness"] if not imp.empty else imp),
        "",
        "## Survival AUC Improvement Test",
        _markdown_table(imp[imp.target == "minority_survival_auc"] if not imp.empty else imp),
        "",
        "## Coefficients",
        _markdown_table(coefs),
        "",
        "## Sign Stability",
        _markdown_table(stability),
        "",
        "## Per-Family Holdout Results",
        _markdown_table(holdout),
        "",
        "## Residual Improvement By Family",
        _markdown_table(residuals),
        "",
        "## Interpretation",
        *interpretation(perf, imp, coefs, stability, residuals),
        "",
        "## Limitations",
        "- Topology variables are sparse outside topology/fragmentation-style experiments and are median-imputed inside pipelines.",
        "- Ridge coefficients are associational and depend on available experiment families.",
        "- Fragmentation-derived topology is controlled by generator assumptions and should not be read as universal topology.",
        "",
        "## Next Steps",
        "- Add explicit topology extraction to density/dropout families to reduce topology missingness.",
        "- Inspect high-residual families for additional dynamic state variables.",
        "- Compare ridge results to nonlinear models after the linear diagnostic is stable.",
    ]
    return "\n".join(lines) + "\n"


def write_report(input_paths: list[Path], output_md: Path, bootstrap_samples: int = 100) -> tuple[Path, ...]:
    df, messages = load_inputs(input_paths)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        output_md.write_text(build_report(df, messages, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()), encoding="utf-8")
        return (output_md,)
    coverage = topology_coverage(df)
    perf, preds = fit_models(df)
    imp = improvement_table(perf)
    coefs = coefficients(df)
    stability = sign_stability(df, bootstrap_samples=bootstrap_samples)
    holdout = per_family_holdout(df)
    residuals = residual_improvement_by_family(holdout)
    output_md.write_text(build_report(df, messages, coverage, perf, imp, coefs, stability, holdout, residuals), encoding="utf-8")
    return (
        output_md,
        plot_predicted(df, preds, "minority_survival_auc", output_md.parent / "topology_augmented_survival_predicted_vs_actual.png"),
        plot_predicted(df, preds, "minority_survival_cliffiness", output_md.parent / "topology_augmented_cliffiness_predicted_vs_actual.png"),
        plot_residuals(df, preds, output_md.parent / "topology_augmented_residuals_by_family.png"),
        plot_improvement(residuals, output_md.parent / "topology_augmented_cliffiness_residual_improvement.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "topology" / "topology_augmented_morphology_summary.md")
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    args = parser.parse_args()
    for path in write_report(args.inputs, args.output_md, bootstrap_samples=args.bootstrap_samples):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
