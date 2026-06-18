"""Test morphology frontier geometry as an explanation for cliffiness."""

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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_accessibility_morphology_with_classical import CLASSICAL_INPUT  # noqa: E402
from report_kernel_morphology_robustness import load_pooled_dataset  # noqa: E402
from report_morphology_ridge_extraction import DEFAULT_INPUTS  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TARGET = "minority_survival_cliffiness"
EPS = 1e-9


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


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return arr
    out = arr.copy()
    half = max(1, window // 2)
    for idx in range(arr.size):
        lo = max(0, idx - half)
        hi = min(arr.size, idx + half + 1)
        out[idx] = float(np.nanmean(arr[lo:hi]))
    return out


def binwise_frontier(df: pd.DataFrame, q_upper: float = 0.95, q_lower: float = 0.05, n_bins: int = 40) -> pd.DataFrame:
    data = df.dropna(subset=["breadth", "elevation"]).copy()
    bins = np.linspace(float(data["breadth"].min()), float(data["breadth"].max()), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    upper = []
    lower = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        part = data[(data["breadth"] >= lo) & (data["breadth"] <= hi)]
        if part.empty:
            upper.append(np.nan)
            lower.append(np.nan)
        else:
            upper.append(float(part["elevation"].quantile(q_upper)))
            lower.append(float(part["elevation"].quantile(q_lower)))
    upper_arr = pd.Series(upper).interpolate(limit_direction="both").to_numpy(dtype=float)
    lower_arr = pd.Series(lower).interpolate(limit_direction="both").to_numpy(dtype=float)
    upper_arr = np.maximum.accumulate(_smooth(upper_arr, window=5)[::-1])[::-1]
    lower_arr = np.minimum(_smooth(lower_arr, window=5), upper_arr - EPS)
    return pd.DataFrame({"breadth": centers, "upper": upper_arr, "lower": lower_arr})


def quantile_model_frontiers(df: pd.DataFrame, quantiles: list[float] | None = None) -> pd.DataFrame:
    quantiles = quantiles or [0.90, 0.95, 0.99, 0.05]
    data = df.dropna(subset=["breadth", "elevation"]).copy()
    x_grid = np.linspace(float(data["breadth"].min()), float(data["breadth"].max()), 160)
    rows = []
    for q in quantiles:
        model = GradientBoostingRegressor(loss="quantile", alpha=float(q), n_estimators=120, max_depth=2, random_state=7)
        model.fit(data[["breadth"]], data["elevation"])
        pred = model.predict(pd.DataFrame({"breadth": x_grid}))
        for x, y in zip(x_grid, pred, strict=False):
            rows.append({"quantile": q, "breadth": float(x), "elevation": float(y)})
    return pd.DataFrame(rows)


def attach_frontier_features(df: pd.DataFrame, frontier: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    b = out["breadth"].to_numpy(dtype=float)
    upper = np.interp(b, frontier["breadth"], frontier["upper"])
    lower = np.interp(b, frontier["breadth"], frontier["lower"])
    e = out["elevation"].to_numpy(dtype=float)
    width = np.maximum(upper - lower, EPS)
    out["elevation_upper"] = upper
    out["elevation_lower"] = lower
    out["vertical_slack"] = np.maximum(0.0, upper - e)
    out["normalized_vertical_slack"] = out["vertical_slack"] / np.maximum(upper, EPS)
    out["distance_to_lower_envelope"] = np.maximum(0.0, e - lower)
    out["normalized_position_between_envelopes"] = np.clip((e - lower) / width, 0.0, 1.0)
    out["signed_frontier_distance"] = e - upper
    out["nearest_frontier_distance"] = np.minimum(np.abs(e - upper), np.abs(e - lower))
    grid_b = frontier["breadth"].to_numpy(dtype=float)
    grid_u = frontier["upper"].to_numpy(dtype=float)
    slope = np.gradient(grid_u, grid_b)
    curvature = np.abs(np.gradient(slope, grid_b)) / np.maximum((1.0 + slope**2) ** 1.5, EPS)
    out["frontier_slope"] = np.interp(b, grid_b, slope)
    out["frontier_curvature"] = np.interp(b, grid_b, curvature)
    out["local_frontier_width"] = width
    out["convex_hull_distance"] = convex_hull_distance(out[["breadth", "elevation"]].to_numpy(dtype=float))
    return out


def convex_hull_distance(coords: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        return np.full(coords.shape[0], np.nan)
    if coords.shape[0] < 3:
        return np.full(coords.shape[0], np.nan)
    try:
        hull = ConvexHull(coords)
        vertices = coords[hull.vertices]
        distances = []
        for point in coords:
            distances.append(float(np.min(np.sqrt(np.sum((vertices - point) ** 2, axis=1)))))
        return np.asarray(distances)
    except Exception:
        return np.full(coords.shape[0], np.nan)


def cv_schemes(df: pd.DataFrame) -> dict[str, tuple[object, pd.Series | None]]:
    schemes: dict[str, tuple[object, pd.Series | None]] = {"random_kfold": (KFold(n_splits=5, shuffle=True, random_state=7), None)}
    if df["experiment_family"].nunique() >= 2:
        schemes["group_experiment_family"] = (GroupKFold(n_splits=min(5, df["experiment_family"].nunique())), df["experiment_family"])
    if df["model"].nunique() >= 2:
        schemes["group_model_id"] = (GroupKFold(n_splits=min(5, df["model"].nunique())), df["model"])
    return schemes


def _pipeline(features: list[str], model) -> Pipeline:
    return Pipeline(
        [
            ("prep", ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)])),
            ("model", model),
        ]
    )


def evaluate_feature_sets(df: pd.DataFrame) -> pd.DataFrame:
    feature_sets = {
        "linear_morphology": (["breadth", "elevation"], Ridge(alpha=1.0)),
        "compact_equation": (["log_breadth", "breadth", "breadth2_x_elevation"], Ridge(alpha=1.0)),
        "frontier_only": (["vertical_slack", "normalized_vertical_slack", "normalized_position_between_envelopes", "nearest_frontier_distance", "frontier_curvature"], Ridge(alpha=1.0)),
        "frontier_morphology": (["breadth", "elevation", "vertical_slack", "normalized_vertical_slack", "normalized_position_between_envelopes", "nearest_frontier_distance", "frontier_curvature", "local_frontier_width"], Ridge(alpha=1.0)),
        "random_forest_reference": (["breadth", "elevation"], RandomForestRegressor(n_estimators=150, min_samples_leaf=5, random_state=7, n_jobs=1)),
        "rbf_reference": (["breadth", "elevation"], Pipeline([("rbf", Nystroem(kernel="rbf", gamma=30.0, n_components=min(300, len(df)), random_state=7)), ("ridge", Ridge(alpha=1e-2))])),
    }
    rows = []
    for scheme_name, (cv, groups) in cv_schemes(df).items():
        for name, (features, model) in feature_sets.items():
            data = df.dropna(subset=[*features, TARGET]).copy()
            X = data[features]
            y = data[TARGET].to_numpy(dtype=float)
            pipe = _pipeline(features, model)
            group_values = data["experiment_family"] if scheme_name == "group_experiment_family" else data["model"] if scheme_name == "group_model_id" else None
            pred = cross_val_predict(pipe, X, y, cv=cv, groups=group_values)
            pipe.fit(X, y)
            in_pred = pipe.predict(X)
            rows.append({"feature_set": name, "validation_scheme": scheme_name, "cv_r2": float(r2_score(y, pred)), "cv_mae": float(mean_absolute_error(y, pred)), "cv_rmse": _rmse(y, pred), "in_sample_r2": float(r2_score(y, in_pred)), "train_test_gap": float(r2_score(y, in_pred) - r2_score(y, pred)), "n_rows": len(data)})
    return pd.DataFrame(rows)


def frontier_importance(df: pd.DataFrame) -> pd.DataFrame:
    features = ["breadth", "elevation", "vertical_slack", "normalized_vertical_slack", "normalized_position_between_envelopes", "nearest_frontier_distance", "frontier_curvature", "local_frontier_width"]
    data = df.dropna(subset=[*features, TARGET]).copy()
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=7, n_jobs=1).fit(data[features], data[TARGET])
    return pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)


def hypothesis_label(results: pd.DataFrame) -> tuple[str, list[str]]:
    main = results[results["validation_scheme"] == "group_experiment_family"]
    def score(name: str) -> float:
        row = main[main["feature_set"] == name]
        return float(row.iloc[0].cv_r2) if not row.empty else np.nan
    linear = score("linear_morphology")
    frontier = score("frontier_only")
    frontier_morph = score("frontier_morphology")
    reference = max(score("random_forest_reference"), score("rbf_reference"))
    if frontier > 0.50 and frontier_morph >= reference - 0.10:
        label = "supported"
    elif frontier_morph > linear + 0.05 or frontier > linear + 0.05:
        label = "partially supported"
    else:
        label = "not supported"
    bullets = [
        f"Linear morphology grouped CV R2={linear:.4f}.",
        f"Frontier-only grouped CV R2={frontier:.4f}.",
        f"Frontier + morphology grouped CV R2={frontier_morph:.4f}.",
        f"Best nonlinear reference grouped CV R2={reference:.4f}.",
        "Frontier-only geometry is not sufficient when evaluated by experiment-family grouped CV." if frontier <= linear else "Frontier-only geometry improves over linear morphology.",
        "The stronger signal remains local nonlinear morphology rather than envelope distance alone." if frontier_morph < reference - 0.25 else "Frontier features approach the nonlinear reference when combined with morphology.",
    ]
    return label, bullets


def plot_frontier(df: pd.DataFrame, frontier: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    prior = df[df["experiment_family"] != "classical_allocator"]
    classical = df[df["experiment_family"] == "classical_allocator"]
    ax.scatter(prior["breadth"], prior["elevation"], s=8, alpha=0.18, label="prior")
    ax.scatter(classical["breadth"], classical["elevation"], s=22, alpha=0.65, label="classical")
    ax.plot(frontier["breadth"], frontier["upper"], color="black", linewidth=2, label="upper envelope")
    ax.plot(frontier["breadth"], frontier["lower"], color="tab:red", linewidth=2, label="lower envelope")
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Morphology Upper and Lower Frontiers")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_scatter(df: pd.DataFrame, x: str, output: Path, title: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    color = pd.factorize(df["source_family"] if "source_family" in df.columns else df["experiment_family"])[0]
    sc = ax.scatter(df[x], df[TARGET], c=color, cmap="tab10", s=12, alpha=0.35)
    ax.set_xlabel(x)
    ax.set_ylabel(TARGET)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_importance(importance: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    data = importance.sort_values("importance")
    ax.barh(data["feature"], data["importance"])
    ax.set_title("Frontier Feature Importance")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_model_comparison(results: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = results[results["validation_scheme"] == "group_experiment_family"].sort_values("cv_r2")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(data["feature_set"], data["cv_r2"])
    ax.set_xlabel("Grouped CV R2")
    ax.set_title("Frontier Model Comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def build_report(messages, frontier_models, results, importance, label, bullets) -> str:
    return "\n".join([
        "# Frontier Morphology Geometry",
        "",
        "## Inputs Used",
        *[f"- {message}" for message in messages],
        "",
        "## Experiment 1: Upper Envelope Fit",
        "Upper envelopes use binwise quantile smoothing as the primary frontier and gradient-boosting quantile curves as diagnostics.",
        _markdown_table(frontier_models.head(20)),
        "",
        "## Experiment 2: Lower Envelope / Quantized Floor",
        "Lower envelope is estimated with binwise low-quantile smoothing. Normalized position is clipped to [0, 1].",
        "",
        "## Experiment 3: Local Frontier Curvature",
        "Frontier slope, curvature, and local frontier width are computed from the smoothed upper/lower envelopes and attached by breadth.",
        "",
        "## Experiment 4: Predictive Tests",
        _markdown_table(results),
        "",
        "## Frontier Feature Importance",
        _markdown_table(importance),
        "",
        "## Interpretation",
        f"Frontier hypothesis: {label}.",
        *[f"- {bullet}" for bullet in bullets],
        "",
        "## Paper 2 Interpretation",
        "If supported: cliffiness is largely explained by allocator position relative to the feasible morphology frontier.",
        "If partially supported: frontier geometry contributes to cliffiness, but local nonlinear morphology remains necessary.",
        "If not supported: nonlinear cliffiness is not reducible to frontier distance and likely reflects local morphology regions or family-specific trajectories.",
    ]) + "\n"


def write_report(output_md: Path, prior_inputs: list[Path] | None = None, classical_input: Path = CLASSICAL_INPUT) -> tuple[Path, ...]:
    prior_inputs = prior_inputs if prior_inputs is not None else DEFAULT_INPUTS
    df, messages = load_pooled_dataset(prior_inputs, classical_input)
    df = df.copy()
    df["breadth2_x_elevation"] = df["breadth"] ** 2 * df["elevation"]
    df["log_breadth"] = np.log(np.clip(df["breadth"], 0.0, None) + EPS)
    frontier = binwise_frontier(df)
    q_frontiers = quantile_model_frontiers(df)
    enriched = attach_frontier_features(df, frontier)
    results = evaluate_feature_sets(enriched)
    importance = frontier_importance(enriched)
    label, bullets = hypothesis_label(results)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_report(messages, q_frontiers, results, importance, label, bullets), encoding="utf-8")
    return (
        output_md,
        plot_frontier(enriched, frontier, output_md.parent / "morphology_upper_frontier.png"),
        plot_scatter(enriched, "vertical_slack", output_md.parent / "frontier_distance_vs_cliffiness.png", "Vertical Slack vs Cliffiness"),
        plot_scatter(enriched, "normalized_position_between_envelopes", output_md.parent / "normalized_frontier_position_vs_cliffiness.png", "Normalized Frontier Position vs Cliffiness"),
        plot_importance(importance, output_md.parent / "frontier_feature_importance.png"),
        plot_model_comparison(results, output_md.parent / "frontier_model_comparison.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "topology" / "frontier_morphology_summary.md")
    parser.add_argument("--prior-inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--classical-input", type=Path, default=CLASSICAL_INPUT)
    args = parser.parse_args()
    for path in write_report(args.output_md, prior_inputs=args.prior_inputs, classical_input=args.classical_input):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
