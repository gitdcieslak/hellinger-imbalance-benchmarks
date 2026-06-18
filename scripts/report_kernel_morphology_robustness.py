"""Check robustness of nonlinear morphology smoothers for accessibility outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_accessibility_morphology_with_classical import CLASSICAL_INPUT, normalize_classical_for_pool  # noqa: E402
from report_morphology_ridge_extraction import DEFAULT_INPUTS, load_inputs  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["minority_survival_auc", "minority_survival_cliffiness"]
GAMMAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
K_VALUES = [5, 10, 25, 50, 100]


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


def load_pooled_dataset(prior_inputs: list[Path], classical_input: Path) -> tuple[pd.DataFrame, list[str]]:
    prior, messages = load_inputs(prior_inputs)
    classical, message = normalize_classical_for_pool(classical_input)
    messages.append(message)
    frames = []
    if not prior.empty:
        frames.append(prior[["experiment_family", "model", "seed", "breadth", "elevation", *TARGETS]].copy())
    if not classical.empty:
        frames.append(classical[["experiment_family", "model", "seed", "breadth", "elevation", *TARGETS]].copy())
    if not frames:
        return pd.DataFrame(columns=["experiment_family", "model", "seed", "breadth", "elevation", *TARGETS]), messages
    df = pd.concat(frames, ignore_index=True).dropna(subset=["breadth", "elevation", *TARGETS]).copy()
    df["source_family"] = np.where(df["experiment_family"].eq("classical_allocator"), "classical", "prior")
    return df, messages


def cv_schemes(df: pd.DataFrame) -> dict[str, tuple[object, pd.Series | None]]:
    schemes: dict[str, tuple[object, pd.Series | None]] = {
        "random_kfold": (KFold(n_splits=min(5, len(df)), shuffle=True, random_state=7), None),
    }
    if df["experiment_family"].nunique() >= 2:
        schemes["group_experiment_family"] = (GroupKFold(n_splits=min(5, df["experiment_family"].nunique())), df["experiment_family"])
    if df["model"].nunique() >= 2:
        schemes["group_model_id"] = (GroupKFold(n_splits=min(5, df["model"].nunique())), df["model"])
    if df["source_family"].nunique() >= 2:
        schemes["group_source_family"] = (GroupKFold(n_splits=2), df["source_family"])
    return schemes


def _num_preprocess() -> ColumnTransformer:
    return ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["breadth", "elevation"])])


def make_model_specs(n_rows: int, gammas: list[float] | None = None, k_values: list[int] | None = None) -> list[dict[str, object]]:
    gammas = gammas or GAMMAS
    k_values = k_values or K_VALUES
    n_components = int(min(300, max(10, n_rows)))
    specs: list[dict[str, object]] = [
        {"model_family": "linear", "model_name": "linear_regression", "parameter": "", "pipeline": Pipeline([("prep", _num_preprocess()), ("model", LinearRegression())])},
        {"model_family": "linear", "model_name": "ridge", "parameter": "alpha=1", "pipeline": Pipeline([("prep", _num_preprocess()), ("model", Ridge(alpha=1.0))])},
        {"model_family": "polynomial", "model_name": "poly_degree_2", "parameter": "degree=2", "pipeline": Pipeline([("prep", _num_preprocess()), ("poly", PolynomialFeatures(degree=2, include_bias=False)), ("model", Ridge(alpha=1.0))])},
        {"model_family": "polynomial", "model_name": "poly_degree_3", "parameter": "degree=3", "pipeline": Pipeline([("prep", _num_preprocess()), ("poly", PolynomialFeatures(degree=3, include_bias=False)), ("model", Ridge(alpha=1.0))])},
        {"model_family": "ensemble", "model_name": "random_forest", "parameter": "n_estimators=100", "pipeline": Pipeline([("prep", _num_preprocess()), ("model", RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=7, n_jobs=1))])},
        {"model_family": "ensemble", "model_name": "gradient_boosting", "parameter": "default", "pipeline": Pipeline([("prep", _num_preprocess()), ("model", GradientBoostingRegressor(random_state=7))])},
    ]
    for gamma in gammas:
        specs.append(
            {
                "model_family": "rbf_kernel",
                "model_name": "rbf_nystroem_ridge",
                "parameter": f"gamma={gamma:g}",
                "gamma": float(gamma),
                "pipeline": Pipeline([("prep", _num_preprocess()), ("rbf", Nystroem(kernel="rbf", gamma=float(gamma), n_components=n_components, random_state=7)), ("model", Ridge(alpha=1e-2))]),
            }
        )
    for k in k_values:
        specs.append(
            {
                "model_family": "knn",
                "model_name": "knn_regressor",
                "parameter": f"k={k}",
                "k": int(k),
                "pipeline": Pipeline([("prep", _num_preprocess()), ("model", KNeighborsRegressor(n_neighbors=min(int(k), max(1, n_rows))))]),
            }
        )
    return specs


def evaluate_models(df: pd.DataFrame, gammas: list[float] | None = None, k_values: list[int] | None = None, schemes: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    all_schemes = cv_schemes(df)
    if schemes is not None:
        all_schemes = {name: value for name, value in all_schemes.items() if name in schemes}
    specs = make_model_specs(len(df), gammas=gammas, k_values=k_values)
    X = df[["breadth", "elevation"]]
    for target in TARGETS:
        y = df[target].to_numpy(dtype=float)
        for scheme_name, (cv, groups) in all_schemes.items():
            for spec in specs:
                pipeline = spec["pipeline"]
                if groups is None:
                    pred = cross_val_predict(pipeline, X, y, cv=cv)
                else:
                    pred = cross_val_predict(pipeline, X, y, cv=cv, groups=groups)
                pipeline.fit(X, y)
                in_pred = pipeline.predict(X)
                cv_r2 = float(r2_score(y, pred)) if len(np.unique(y)) > 1 else 0.0
                in_r2 = float(r2_score(y, in_pred)) if len(np.unique(y)) > 1 else 0.0
                rows.append(
                    {
                        "target": target,
                        "validation_scheme": scheme_name,
                        "model_family": spec["model_family"],
                        "model_name": spec["model_name"],
                        "parameter": spec["parameter"],
                        "cv_r2": cv_r2,
                        "cv_mae": float(mean_absolute_error(y, pred)),
                        "cv_rmse": _rmse(y, pred),
                        "in_sample_r2": in_r2,
                        "train_test_gap": float(in_r2 - cv_r2),
                        "n_rows": len(df),
                    }
                )
    return pd.DataFrame(rows)


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {"metric": "rows", "value": len(df)},
            {"metric": "experiment_families", "value": df["experiment_family"].nunique()},
            {"metric": "models", "value": df["model"].nunique()},
            {"metric": "source_families", "value": df["source_family"].nunique()},
        ]
    )


def best_by_target(results: pd.DataFrame, scheme: str = "group_experiment_family") -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    data = results[results["validation_scheme"] == scheme].copy()
    if data.empty:
        return pd.DataFrame()
    return data.sort_values(["target", "cv_r2"], ascending=[True, False]).groupby("target", as_index=False).head(1).reset_index(drop=True)


def robustness_interpretation(results: pd.DataFrame) -> tuple[str, pd.DataFrame, list[str]]:
    if results.empty:
        return "fragile", pd.DataFrame(), ["No model results were available."]
    main = results[(results["target"] == "minority_survival_cliffiness") & (results["validation_scheme"] == "group_experiment_family")].copy()
    linear = main[main["model_name"] == "linear_regression"]
    linear_r2 = float(linear.iloc[0].cv_r2) if not linear.empty else np.nan
    nonlinear = main[~main["model_family"].isin(["linear"])]
    beaters = nonlinear[nonlinear["cv_r2"] > linear_r2]
    rbf = main[main["model_family"] == "rbf_kernel"].copy()
    rbf["beats_linear_by_0_25"] = rbf["cv_r2"] >= linear_r2 + 0.25
    stable_rbf_count = int(rbf["beats_linear_by_0_25"].sum()) if not rbf.empty else 0
    random_best = results[(results["target"] == "minority_survival_cliffiness") & (results["validation_scheme"] == "random_kfold")]["cv_r2"].max()
    grouped_best = main["cv_r2"].max()
    if len(beaters) >= 3 and stable_rbf_count >= 3 and pd.notna(grouped_best) and grouped_best > 0.25:
        label = "robust"
    elif len(beaters) >= 2 and stable_rbf_count >= 1 and pd.notna(grouped_best) and grouped_best > 0.0:
        label = "partially robust"
    else:
        label = "fragile"
    summary = pd.DataFrame(
        [
            {"diagnostic": "linear_grouped_cv_r2", "value": linear_r2},
            {"diagnostic": "nonlinear_models_beating_linear_grouped", "value": float(len(beaters))},
            {"diagnostic": "rbf_bandwidths_beating_linear_by_0_25", "value": float(stable_rbf_count)},
            {"diagnostic": "best_random_cv_r2", "value": float(random_best) if pd.notna(random_best) else np.nan},
            {"diagnostic": "best_grouped_cv_r2", "value": float(grouped_best) if pd.notna(grouped_best) else np.nan},
        ]
    )
    bullets = [
        f"Cliffiness nonlinear morphology is {label}.",
        f"Under experiment-family grouped CV, {len(beaters)} nonlinear models beat linear morphology.",
        f"The RBF sweep has {stable_rbf_count} bandwidths with CV R2 at least 0.25 above linear morphology.",
        f"Best random-CV cliffiness R2 is {float(random_best):.4f}; best grouped-CV cliffiness R2 is {float(grouped_best):.4f}.",
    ]
    return label, summary, bullets


def plot_bandwidth(results: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = results[(results["model_family"] == "rbf_kernel") & (results["target"] == "minority_survival_cliffiness")].copy()
    data["gamma"] = data["parameter"].str.replace("gamma=", "", regex=False).astype(float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for scheme, group in data.groupby("validation_scheme"):
        group = group.sort_values("gamma")
        ax.plot(group["gamma"], group["cv_r2"], marker="o", label=scheme)
    ax.set_xscale("log")
    ax.set_xlabel("RBF gamma")
    ax.set_ylabel("Cliffiness CV R2")
    ax.set_title("Kernel Robustness Bandwidth Sweep")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_model_comparison(results: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = results[(results["target"] == "minority_survival_cliffiness") & (results["validation_scheme"] == "group_experiment_family")].copy()
    best = data.sort_values("cv_r2", ascending=False).groupby("model_name", as_index=False).head(1).sort_values("cv_r2")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(best["model_name"] + "\n" + best["parameter"].astype(str), best["cv_r2"])
    ax.tick_params(axis="x", rotation=70)
    ax.set_ylabel("Grouped CV R2")
    ax.set_title("Kernel Robustness Model Comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_validation(results: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = results[results["target"] == "minority_survival_cliffiness"].copy()
    best = data.sort_values("cv_r2", ascending=False).groupby(["validation_scheme", "model_family"], as_index=False).head(1)
    pivot = best.pivot(index="model_family", columns="validation_scheme", values="cv_r2").fillna(np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Best cliffiness CV R2")
    ax.set_title("Kernel Robustness Validation Schemes")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_gap(results: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = results[(results["target"] == "minority_survival_cliffiness") & (results["validation_scheme"] == "group_experiment_family")].copy()
    best = data.sort_values("cv_r2", ascending=False).groupby("model_name", as_index=False).head(1).sort_values("train_test_gap")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(best["model_name"] + "\n" + best["parameter"].astype(str), best["train_test_gap"])
    ax.tick_params(axis="x", rotation=70)
    ax.set_ylabel("In-sample R2 - CV R2")
    ax.set_title("Kernel Robustness Train/Test Gap")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def build_report(messages: list[str], summary: pd.DataFrame, results: pd.DataFrame, best: pd.DataFrame, robustness: pd.DataFrame, bullets: list[str], label: str) -> str:
    linear = results[results["model_family"] == "linear"]
    rbf = results[results["model_family"] == "rbf_kernel"]
    alt = results[~results["model_family"].isin(["linear", "rbf_kernel"])]
    validation = results.groupby(["target", "validation_scheme"], as_index=False)["cv_r2"].max() if not results.empty else pd.DataFrame()
    return "\n".join(
        [
            "# Kernel Morphology Robustness Check",
            "",
            "## Inputs Used",
            *[f"- {message}" for message in messages],
            "",
            "## Dataset Summary",
            _markdown_table(summary),
            "",
            "## Linear Baseline Results",
            _markdown_table(linear),
            "",
            "## RBF Bandwidth Sweep",
            _markdown_table(rbf),
            "",
            "## Alternative Nonlinear Smoothers",
            _markdown_table(alt),
            "",
            "## Validation-Scheme Sensitivity",
            _markdown_table(validation),
            "",
            "## Best Model by Target",
            _markdown_table(best),
            "",
            "## Robustness Interpretation",
            _markdown_table(robustness),
            *[f"- {bullet}" for bullet in bullets],
            "",
            "## Paper 2 Recommendation",
            f"Use the nonlinear morphology claim with robustness label: {label}.",
            "The manuscript should emphasize the direction and stability of nonlinear improvement over linear morphology, not the single best R2 value.",
        ]
    ) + "\n"


def write_report(
    output_md: Path,
    prior_inputs: list[Path] | None = None,
    classical_input: Path = CLASSICAL_INPUT,
    gammas: list[float] | None = None,
    k_values: list[int] | None = None,
    schemes: list[str] | None = None,
) -> tuple[Path, ...]:
    prior_inputs = prior_inputs if prior_inputs is not None else DEFAULT_INPUTS
    df, messages = load_pooled_dataset(prior_inputs, classical_input)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    results = evaluate_models(df, gammas=gammas, k_values=k_values, schemes=schemes)
    best = best_by_target(results)
    label, robustness, bullets = robustness_interpretation(results)
    output_md.write_text(build_report(messages, dataset_summary(df), results, best, robustness, bullets, label), encoding="utf-8")
    return (
        output_md,
        plot_bandwidth(results, output_md.parent / "kernel_robustness_bandwidth_sweep.png"),
        plot_model_comparison(results, output_md.parent / "kernel_robustness_model_comparison.png"),
        plot_validation(results, output_md.parent / "kernel_robustness_validation_schemes.png"),
        plot_gap(results, output_md.parent / "kernel_robustness_train_test_gap.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "topology" / "kernel_morphology_robustness_summary.md")
    parser.add_argument("--prior-inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--classical-input", type=Path, default=CLASSICAL_INPUT)
    args = parser.parse_args()
    for path in write_report(args.output_md, prior_inputs=args.prior_inputs, classical_input=args.classical_input):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
