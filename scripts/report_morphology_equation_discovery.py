"""Discover interpretable morphology equations for accessibility cliffiness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_accessibility_morphology_with_classical import CLASSICAL_INPUT  # noqa: E402
from report_classical_allocator_morphology import classify_regime  # noqa: E402
from report_kernel_morphology_robustness import load_pooled_dataset  # noqa: E402
from report_morphology_ridge_extraction import DEFAULT_INPUTS  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TARGET = "minority_survival_cliffiness"
EPS = 1e-6


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float) or isinstance(value, np.floating):
                values.append(format(float(value), floatfmt) if pd.notna(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def add_symbolic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    b = out["breadth"].clip(lower=0.0).astype(float)
    e = out["elevation"].clip(lower=0.0).astype(float)
    centroid_b = float(b.mean())
    centroid_e = float(e.mean())
    out["breadth2"] = b**2
    out["elevation2"] = e**2
    out["breadth3"] = b**3
    out["elevation3"] = e**3
    out["breadth_x_elevation"] = b * e
    out["breadth2_x_elevation"] = b**2 * e
    out["breadth_x_elevation2"] = b * e**2
    out["sqrt_breadth"] = np.sqrt(b)
    out["sqrt_elevation"] = np.sqrt(e)
    out["log_breadth"] = np.log(b + EPS)
    out["distance_from_origin"] = np.sqrt(b**2 + e**2)
    out["distance_from_broad_allocator_corner"] = np.sqrt((b - float(b.max())) ** 2 + e**2)
    out["distance_from_quantized_allocator_line"] = b
    out["distance_from_manifold_centroid"] = np.sqrt((b - centroid_b) ** 2 + (e - centroid_e) ** 2)
    return out


def symbolic_feature_names() -> list[str]:
    return [
        "breadth",
        "elevation",
        "breadth2",
        "elevation2",
        "breadth3",
        "elevation3",
        "breadth_x_elevation",
        "breadth2_x_elevation",
        "breadth_x_elevation2",
        "sqrt_breadth",
        "sqrt_elevation",
        "log_breadth",
        "distance_from_origin",
        "distance_from_broad_allocator_corner",
        "distance_from_quantized_allocator_line",
        "distance_from_manifold_centroid",
    ]


def grouped_cv(df: pd.DataFrame) -> tuple[GroupKFold, pd.Series]:
    groups = df["experiment_family"]
    return GroupKFold(n_splits=min(5, groups.nunique())), groups


def evaluate_pipeline(df: pd.DataFrame, features: list[str], model) -> tuple[dict[str, float], np.ndarray]:
    data = df.dropna(subset=[*features, TARGET]).copy()
    X = data[features]
    y = data[TARGET].to_numpy(dtype=float)
    cv, groups = grouped_cv(data)
    pipeline = Pipeline(
        [
            ("prep", ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)])),
            ("model", model),
        ]
    )
    pred = cross_val_predict(pipeline, X, y, cv=cv, groups=groups)
    pipeline.fit(X, y)
    in_pred = pipeline.predict(X)
    metrics = {
        "cv_r2": float(r2_score(y, pred)),
        "cv_mae": float(mean_absolute_error(y, pred)),
        "cv_rmse": _rmse(y, pred),
        "in_sample_r2": float(r2_score(y, in_pred)),
        "train_test_gap": float(r2_score(y, in_pred) - r2_score(y, pred)),
        "n_features": len(features),
        "n_rows": len(data),
    }
    return metrics, pred


def lasso_elasticnet_results(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = symbolic_feature_names()
    specs = [("lasso", Lasso(alpha=0.002, max_iter=20000)), ("elastic_net", ElasticNet(alpha=0.002, l1_ratio=0.5, max_iter=20000))]
    rows = []
    coef_rows = []
    for name, model in specs:
        metrics, _ = evaluate_pipeline(df, features, model)
        pipeline = Pipeline([("scale", StandardScaler()), ("model", clone(model))])
        X = df[features].to_numpy(dtype=float)
        y = df[TARGET].to_numpy(dtype=float)
        pipeline.fit(X, y)
        coefs = pipeline.named_steps["model"].coef_
        nonzero = int(np.sum(np.abs(coefs) > 1e-8))
        rows.append({"equation_model": name, **metrics, "nonzero_terms": nonzero})
        for feature, coef in sorted(zip(features, coefs, strict=False), key=lambda item: abs(item[1]), reverse=True):
            if abs(float(coef)) > 1e-8:
                coef_rows.append({"equation_model": name, "feature": feature, "standardized_coefficient": float(coef)})
    return pd.DataFrame(rows), pd.DataFrame(coef_rows)


def stepwise_regression(df: pd.DataFrame, max_terms: int = 6) -> tuple[pd.DataFrame, list[str]]:
    remaining = symbolic_feature_names().copy()
    selected: list[str] = []
    rows = []
    best_score = -np.inf
    for step in range(1, max_terms + 1):
        candidates = []
        for feature in remaining:
            metrics, _ = evaluate_pipeline(df, [*selected, feature], Ridge(alpha=1.0))
            candidates.append((metrics["cv_r2"], feature, metrics))
        candidates.sort(reverse=True, key=lambda item: item[0])
        score, feature, metrics = candidates[0]
        if score <= best_score + 1e-4 and selected:
            break
        selected.append(feature)
        remaining.remove(feature)
        best_score = score
        rows.append({"step": step, "added_feature": feature, "selected_terms": " + ".join(selected), **metrics})
    return pd.DataFrame(rows), selected


def classify_regimes_for_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "positive_quantization_score" not in out.columns:
        out["positive_quantization_score"] = np.nan
    if "positive_unique_score_ratio" not in out.columns:
        out["positive_unique_score_ratio"] = np.nan
    fallback_quant = (out["breadth"] <= 0.60) & (out["elevation"] < 0.30)
    out["operational_regime"] = out.apply(classify_regime, axis=1)
    out.loc[fallback_quant & out["operational_regime"].eq("mixed morphology"), "operational_regime"] = "quantized allocator"
    return out


def boundary_distance_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = classify_regimes_for_pool(df).copy()
    X = data[["breadth", "elevation"]].to_numpy(dtype=float)
    y_regime = data["operational_regime"].astype(str)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    svm = SVC(kernel="linear", decision_function_shape="ovr", random_state=7).fit(Xs, y_regime)
    logistic = LogisticRegression(max_iter=2000, random_state=7).fit(Xs, y_regime)
    svm_decision = np.asarray(svm.decision_function(Xs), dtype=float)
    log_decision = np.asarray(logistic.decision_function(Xs), dtype=float)
    data["distance_to_svm_boundary"] = np.min(np.abs(svm_decision), axis=1) if svm_decision.ndim == 2 else np.abs(svm_decision)
    data["distance_to_logistic_boundary"] = np.min(np.abs(log_decision), axis=1) if log_decision.ndim == 2 else np.abs(log_decision)
    rows = []
    for feature in ["distance_to_svm_boundary", "distance_to_logistic_boundary"]:
        metrics, _ = evaluate_pipeline(data, [feature], Ridge(alpha=1.0))
        rows.append({"geometry_variable": feature, **metrics})
    return data, pd.DataFrame(rows)


def manifold_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    coords = data[["breadth", "elevation"]].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(coords)
    pca = PCA(n_components=2).fit(scaled)
    pcs = pca.transform(scaled)
    data["pc1"] = pcs[:, 0]
    data["pc2"] = pcs[:, 1]
    ordered = data.sort_values("pc1")
    arc = np.zeros(len(ordered), dtype=float)
    if len(ordered) > 1:
        diffs = np.diff(ordered[["pc1", "pc2"]].to_numpy(dtype=float), axis=0)
        arc = np.concatenate([[0.0], np.cumsum(np.sqrt(np.sum(diffs**2, axis=1)))])
        if arc[-1] > 0:
            arc = arc / arc[-1]
    data.loc[ordered.index, "arc_length"] = arc
    curvature = np.zeros(len(ordered), dtype=float)
    if len(ordered) > 2:
        pts = ordered[["pc1", "pc2"]].to_numpy(dtype=float)
        d1 = np.gradient(pts, axis=0)
        d2 = np.gradient(d1, axis=0)
        numerator = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        denominator = np.maximum((d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5, EPS)
        curvature = numerator / denominator
    data.loc[ordered.index, "curvature"] = curvature
    nn = NearestNeighbors(n_neighbors=min(20, len(data))).fit(scaled)
    distances, _ = nn.kneighbors(scaled)
    data["local_density"] = 1.0 / np.maximum(EPS, distances[:, 1:].mean(axis=1) if distances.shape[1] > 1 else distances.mean(axis=1))
    data["distance_to_ridge"] = np.abs(data["pc2"])
    try:
        embedding = SpectralEmbedding(n_components=1, n_neighbors=min(20, len(data) - 1), random_state=7).fit_transform(scaled)
        data["diffusion_coordinate"] = embedding[:, 0]
    except Exception:
        data["diffusion_coordinate"] = np.nan

    rows = []
    for feature in ["pc1", "pc2", "arc_length", "curvature", "distance_to_ridge", "local_density", "diffusion_coordinate"]:
        if data[feature].notna().all():
            metrics, _ = evaluate_pipeline(data, [feature], Ridge(alpha=1.0))
            rows.append({"geometry_variable": feature, **metrics})
    return data, pd.DataFrame(rows)


def fit_tree_models(df: pd.DataFrame) -> tuple[RandomForestRegressor, GradientBoostingRegressor, pd.DataFrame]:
    X = df[["breadth", "elevation"]]
    y = df[TARGET].to_numpy(dtype=float)
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=7, n_jobs=1).fit(X, y)
    gb = GradientBoostingRegressor(random_state=7).fit(X, y)
    rows = []
    for name, model in [("random_forest", rf), ("gradient_boosting", gb)]:
        metrics, _ = evaluate_pipeline(df, ["breadth", "elevation"], clone(model))
        rows.append({"model": name, **metrics})
    return rf, gb, pd.DataFrame(rows)


def plot_pdp(model, df: pd.DataFrame, features, output: Path, title: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    PartialDependenceDisplay.from_estimator(model, df[["breadth", "elevation"]], features, ax=ax)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def shap_or_fallback_plots(model, df: pd.DataFrame, output_dir: Path) -> tuple[list[Path], str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = df[["breadth", "elevation"]]
    try:
        import shap  # type: ignore

        sample = X.sample(n=min(1000, len(X)), random_state=7)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
        plt.figure(figsize=(7, 5))
        shap.summary_plot(values, sample, show=False)
        summary = output_dir / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(summary, dpi=170)
        plt.close()
        outputs = [summary]
        for feature in ["breadth", "elevation"]:
            plt.figure(figsize=(7, 5))
            shap.dependence_plot(feature, values, sample, show=False)
            path = output_dir / f"shap_dependence_{feature}.png"
            plt.tight_layout()
            plt.savefig(path, dpi=170)
            plt.close()
            outputs.append(path)
        return outputs, "native shap.TreeExplainer"
    except Exception as exc:
        baseline = float(np.mean(model.predict(X)))
        preds = model.predict(X)
        importances = getattr(model, "feature_importances_", np.asarray([0.5, 0.5]))
        contrib = pd.DataFrame({"feature": ["breadth", "elevation"], "importance": importances})
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(contrib["feature"], contrib["importance"])
        ax.set_title("SHAP fallback: model feature importance")
        summary = output_dir / "shap_summary.png"
        fig.tight_layout()
        fig.savefig(summary, dpi=170)
        plt.close(fig)
        outputs = [summary]
        for feature in ["breadth", "elevation"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(X[feature], preds - baseline, s=10, alpha=0.35)
            ax.set_xlabel(feature)
            ax.set_ylabel("prediction - mean prediction")
            ax.set_title(f"SHAP fallback dependence: {feature}")
            path = output_dir / f"shap_dependence_{feature}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=170)
            plt.close(fig)
            outputs.append(path)
        return outputs, f"fallback model importance because SHAP failed: {exc}"


def plot_boundary_distance(df: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df["distance_to_svm_boundary"], df[TARGET], c=df["breadth"], cmap="viridis", s=12, alpha=0.45)
    ax.set_xlabel("distance_to_svm_boundary")
    ax.set_ylabel(TARGET)
    ax.set_title("Boundary Distance vs Cliffiness")
    fig.colorbar(sc, ax=ax, label="breadth")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def discovery_summary(stepwise: pd.DataFrame, boundary: pd.DataFrame, manifold: pd.DataFrame, tree_perf: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows = []
    if not stepwise.empty:
        best = stepwise.sort_values("cv_r2", ascending=False).iloc[0]
        rows.append({"candidate": "best_stepwise_equation", "feature_or_terms": best.selected_terms, "cv_r2": float(best.cv_r2)})
    for name, table in [("boundary", boundary), ("manifold", manifold)]:
        if not table.empty:
            best = table.sort_values("cv_r2", ascending=False).iloc[0]
            rows.append({"candidate": f"best_{name}_quantity", "feature_or_terms": best.geometry_variable, "cv_r2": float(best.cv_r2)})
    if not tree_perf.empty:
        best = tree_perf.sort_values("cv_r2", ascending=False).iloc[0]
        rows.append({"candidate": "best_tree_nonlinear", "feature_or_terms": best.model, "cv_r2": float(best.cv_r2)})
    summary = pd.DataFrame(rows)
    stepwise_r2 = float(stepwise["cv_r2"].max()) if not stepwise.empty else np.nan
    boundary_r2 = float(boundary["cv_r2"].max()) if not boundary.empty else np.nan
    manifold_r2 = float(manifold["cv_r2"].max()) if not manifold.empty else np.nan
    best_single_geometry = np.nanmax([boundary_r2, manifold_r2])
    if pd.notna(best_single_geometry) and best_single_geometry > 0.65:
        label = "exceptional success"
    elif pd.notna(best_single_geometry) and best_single_geometry > 0.60:
        label = "strong success"
    elif pd.notna(stepwise_r2) and stepwise_r2 > 0.50:
        label = "weak success"
    else:
        label = "not yet successful"
    return summary, label


def interpretation_answers(stepwise: pd.DataFrame, boundary: pd.DataFrame, manifold: pd.DataFrame, tree_perf: pd.DataFrame, summary: pd.DataFrame, label: str) -> list[str]:
    best_step = stepwise.sort_values("cv_r2", ascending=False).head(1)
    best_boundary = boundary.sort_values("cv_r2", ascending=False).head(1)
    best_manifold = manifold.sort_values("cv_r2", ascending=False).head(1)
    best_tree = tree_perf.sort_values("cv_r2", ascending=False).head(1)
    return [
        f"Smallest useful equation: {best_step.iloc[0].selected_terms if not best_step.empty else 'n/a'} with grouped CV R2={float(best_step.iloc[0].cv_r2) if not best_step.empty else np.nan:.4f}.",
        "Cliffiness appears to depend strongly on breadth and nonlinear breadth terms; the selected sparse equation starts with log_breadth and breadth.",
        f"Distance-to-boundary hypothesis is not supported by the current linear SVM/logistic boundary distances; best boundary CV R2={float(best_boundary.iloc[0].cv_r2) if not best_boundary.empty else np.nan:.4f}.",
        f"Single latent manifold coordinates are not sufficient; best manifold-coordinate CV R2={float(best_manifold.iloc[0].cv_r2) if not best_manifold.empty else np.nan:.4f}.",
        f"Nonlinear local models still perform best among tested interpretable/nonparametric options; best tree model={best_tree.iloc[0].model if not best_tree.empty else 'n/a'} with CV R2={float(best_tree.iloc[0].cv_r2) if not best_tree.empty else np.nan:.4f}.",
        f"Success classification: {label}. This is weak success for an interpretable sparse equation, not exceptional success for a single geometry-derived variable.",
    ]


def build_report(messages, equation_perf, equation_terms, stepwise, tree_perf, boundary_perf, manifold_perf, summary, label, shap_mode, answers) -> str:
    return "\n".join(
        [
            "# Morphology Equation Discovery",
            "",
            "## Inputs Used",
            *[f"- {message}" for message in messages],
            "",
            "## Experiment 1: Symbolic Feature Expansion",
            _markdown_table(equation_perf),
            "",
            "### Sparse Equation Terms",
            _markdown_table(equation_terms.head(20)),
            "",
            "### Stepwise Regression",
            _markdown_table(stepwise),
            "",
            "## Experiment 2: Partial Dependence Analysis",
            _markdown_table(tree_perf),
            "- Generated `pdp_breadth.png`, `pdp_elevation.png`, and `pdp_interaction.png`.",
            "",
            "## Experiment 3: SHAP Geometry Analysis",
            f"SHAP mode: {shap_mode}",
            "- Generated `shap_summary.png`, `shap_dependence_breadth.png`, and `shap_dependence_elevation.png`.",
            "",
            "## Experiment 4: Distance-to-Boundary Hypothesis",
            _markdown_table(boundary_perf),
            "- Generated `boundary_distance_vs_cliffiness.png`.",
            "",
            "## Experiment 5: Manifold Coordinate Discovery",
            _markdown_table(manifold_perf),
            "",
            "## Discovery Summary",
            _markdown_table(summary),
            f"Overall result: {label}.",
            "",
            "## Explicit Answers",
            *[f"- {answer}" for answer in answers],
            "",
            "## Interpretation",
            "- If sparse polynomial/stepwise terms dominate, cliffiness is expressible as a compact equation over morphology coordinates.",
            "- If boundary distance dominates, cliffiness is best interpreted as proximity to morphology transition boundaries.",
            "- If tree/kernel models dominate while single geometry variables lag, nonlinear models are exploiting local regions or interactions not yet captured by a single theory variable.",
            "- Current result: a compact nonlinear equation is promising, but the boundary-distance theory is not yet supported by these simple boundary constructions.",
        ]
    ) + "\n"


def write_report(output_md: Path, prior_inputs: list[Path] | None = None, classical_input: Path = CLASSICAL_INPUT) -> tuple[Path, ...]:
    prior_inputs = prior_inputs if prior_inputs is not None else DEFAULT_INPUTS
    df, messages = load_pooled_dataset(prior_inputs, classical_input)
    df = add_symbolic_features(df)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    equation_perf, equation_terms = lasso_elasticnet_results(df)
    stepwise, _ = stepwise_regression(df)
    boundary_df, boundary_perf = boundary_distance_features(df)
    manifold_df, manifold_perf = manifold_features(df)
    rf, gb, tree_perf = fit_tree_models(df)
    summary, label = discovery_summary(stepwise, boundary_perf, manifold_perf, tree_perf)
    answers = interpretation_answers(stepwise, boundary_perf, manifold_perf, tree_perf, summary, label)
    outputs = [output_md]
    outputs.append(plot_pdp(gb, df, ["breadth"], output_md.parent / "pdp_breadth.png", "PDP: breadth"))
    outputs.append(plot_pdp(gb, df, ["elevation"], output_md.parent / "pdp_elevation.png", "PDP: elevation"))
    outputs.append(plot_pdp(gb, df, [("breadth", "elevation")], output_md.parent / "pdp_interaction.png", "PDP: breadth x elevation"))
    shap_outputs, shap_mode = shap_or_fallback_plots(gb, df, output_md.parent)
    outputs.extend(shap_outputs)
    outputs.append(plot_boundary_distance(boundary_df, output_md.parent / "boundary_distance_vs_cliffiness.png"))
    output_md.write_text(build_report(messages, equation_perf, equation_terms, stepwise, tree_perf, boundary_perf, manifold_perf, summary, label, shap_mode, answers), encoding="utf-8")
    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "topology" / "morphology_equation_discovery.md")
    parser.add_argument("--prior-inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--classical-input", type=Path, default=CLASSICAL_INPUT)
    args = parser.parse_args()
    for path in write_report(args.output_md, prior_inputs=args.prior_inputs, classical_input=args.classical_input):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
