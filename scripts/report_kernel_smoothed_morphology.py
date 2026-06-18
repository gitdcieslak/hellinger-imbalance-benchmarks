"""Report kernel-smoothed morphology fields for accessibility dynamics."""

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
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_morphology_ridge_extraction import DEFAULT_INPUTS, load_inputs


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["minority_survival_auc", "minority_survival_cliffiness"]


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


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _cv_strategy(df: pd.DataFrame):
    if df["experiment_family"].nunique() >= 3:
        return GroupKFold(n_splits=min(5, df["experiment_family"].nunique())), df["experiment_family"], "experiment_family"
    if df["seed"].nunique() >= 3:
        return GroupKFold(n_splits=min(5, df["seed"].nunique())), df["seed"], "seed"
    return KFold(n_splits=min(5, len(df)), shuffle=True, random_state=7), None, "row"


def make_linear_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("prep", ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["breadth", "elevation"])])),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def make_kernel_pipeline(gamma: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("prep", ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["breadth", "elevation"])])),
            ("model", KernelRidge(alpha=1e-2, kernel="rbf", gamma=gamma)),
        ]
    )


def evaluate_smoothers(df: pd.DataFrame, gamma: float = 1.0) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for target in TARGETS:
        data = df.dropna(subset=["breadth", "elevation", target]).copy()
        if len(data) < 3:
            continue
        y = data[target].to_numpy(dtype=float)
        cv, groups, group_name = _cv_strategy(data)
        for name, pipeline in [("linear_morphology", make_linear_pipeline()), ("kernel_morphology", make_kernel_pipeline(gamma))]:
            if groups is None:
                pred = cross_val_predict(pipeline, data, y, cv=cv)
            else:
                pred = cross_val_predict(pipeline, data, y, cv=cv, groups=groups)
            pipeline.fit(data, y)
            in_pred = pipeline.predict(data)
            rows.append(
                {
                    "target": target,
                    "model": name,
                    "cv_group": group_name,
                    "cv_r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else 0.0,
                    "cv_mae": float(mean_absolute_error(y, pred)),
                    "cv_rmse": _rmse(y, pred),
                    "in_sample_r2": float(r2_score(y, in_pred)) if len(np.unique(y)) > 1 else 0.0,
                    "n_rows": len(data),
                }
            )
            full_pred = np.full(len(df), np.nan)
            full_pred[data.index.to_numpy()] = pred
            predictions[f"{target}_{name}"] = full_pred
    return pd.DataFrame(rows), predictions


def build_field_grid(df: pd.DataFrame, target: str, gamma: float = 1.0, grid_size: int = 80) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    data = df.dropna(subset=["breadth", "elevation", target]).copy()
    pipeline = make_kernel_pipeline(gamma)
    pipeline.fit(data, data[target].to_numpy(dtype=float))
    bx = np.linspace(float(data["breadth"].min()), float(data["breadth"].max()), grid_size)
    ey = np.linspace(float(data["elevation"].min()), float(data["elevation"].max()), grid_size)
    xx, yy = np.meshgrid(bx, ey, indexing="ij")
    grid_df = pd.DataFrame({"breadth": xx.ravel(), "elevation": yy.ravel()})
    zz = pipeline.predict(grid_df).reshape(xx.shape)
    db, de = np.gradient(zz, bx, ey, edge_order=2)
    grad = np.sqrt(db * db + de * de)
    lap = np.gradient(db, bx, axis=0, edge_order=2) + np.gradient(de, ey, axis=1, edge_order=2)

    coords = data[["breadth", "elevation"]].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(coords)
    grid_scaled = StandardScaler().fit(coords).transform(np.column_stack([xx.ravel(), yy.ravel()]))
    nn = NearestNeighbors(n_neighbors=min(10, len(data))).fit(scaled)
    distances, indices = nn.kneighbors(grid_scaled)
    density = 1.0 / np.maximum(1e-9, distances.mean(axis=1)).reshape(xx.shape)
    residuals = data[target].to_numpy(dtype=float) - pipeline.predict(data)
    local_var = np.asarray([float(np.var(residuals[idx], ddof=0)) for idx in indices]).reshape(xx.shape)
    field = pd.DataFrame(
        {
            "breadth": xx.ravel(),
            "elevation": yy.ravel(),
            f"predicted_{target}": zz.ravel(),
            f"gradient_{target}": grad.ravel(),
            f"laplacian_{target}": lap.ravel(),
            "local_sample_density": density.ravel(),
            f"local_residual_variance_{target}": local_var.ravel(),
        }
    )
    return field, xx, yy, zz


def field_diagnostics(field: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        pred = f"predicted_{target}"
        grad = f"gradient_{target}"
        lap = f"laplacian_{target}"
        var = f"local_residual_variance_{target}"
        if pred not in field:
            continue
        rows.extend(
            [
                {"target": target, "diagnostic": "predicted_min", "value": float(field[pred].min())},
                {"target": target, "diagnostic": "predicted_max", "value": float(field[pred].max())},
                {"target": target, "diagnostic": "mean_gradient", "value": float(field[grad].mean())},
                {"target": target, "diagnostic": "max_gradient", "value": float(field[grad].max())},
                {"target": target, "diagnostic": "mean_abs_laplacian", "value": float(field[lap].abs().mean())},
                {"target": target, "diagnostic": "mean_local_residual_variance", "value": float(field[var].mean())},
                {"target": target, "diagnostic": "density_vs_predicted_corr", "value": float(field["local_sample_density"].corr(field[pred]))},
                {"target": target, "diagnostic": "density_vs_residual_variance_corr", "value": float(field["local_sample_density"].corr(field[var]))},
            ]
        )
    return pd.DataFrame(rows)


def residuals_by_family(df: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        pred = predictions.get(f"{target}_kernel_morphology")
        if pred is None:
            continue
        data = df.assign(predicted=pred).dropna(subset=[target, "predicted"])
        data["residual"] = data[target] - data["predicted"]
        for family, group in data.groupby("experiment_family"):
            rows.append(
                {
                    "target": target,
                    "experiment_family": family,
                    "mean_residual": float(group["residual"].mean()),
                    "mae": float(np.mean(np.abs(group["residual"]))),
                    "rmse": _rmse(group[target].to_numpy(dtype=float), group["predicted"].to_numpy(dtype=float)),
                    "n_rows": len(group),
                }
            )
    return pd.DataFrame(rows)


def manifold_arcs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in df.groupby("experiment_family"):
        rows.append(
            {
                "experiment_family": family,
                "breadth_min": float(group["breadth"].min()),
                "breadth_max": float(group["breadth"].max()),
                "elevation_min": float(group["elevation"].min()),
                "elevation_max": float(group["elevation"].max()),
                "mean_breadth": float(group["breadth"].mean()),
                "mean_elevation": float(group["elevation"].mean()),
                "mean_cliffiness": float(group["minority_survival_cliffiness"].mean()),
            }
        )
    return pd.DataFrame(rows)


def interpretation(perf: pd.DataFrame, diag: pd.DataFrame, residuals: pd.DataFrame) -> list[str]:
    cliff_linear = perf[(perf.target == "minority_survival_cliffiness") & (perf.model == "linear_morphology")]
    cliff_kernel = perf[(perf.target == "minority_survival_cliffiness") & (perf.model == "kernel_morphology")]
    delta = float(cliff_kernel.iloc[0].cv_r2 - cliff_linear.iloc[0].cv_r2) if not cliff_linear.empty and not cliff_kernel.empty else np.nan
    high_grad = diag[(diag.target == "minority_survival_cliffiness") & (diag.diagnostic == "max_gradient")]
    density_var = diag[(diag.target == "minority_survival_cliffiness") & (diag.diagnostic == "density_vs_residual_variance_corr")]
    worst = residuals[residuals.target == "minority_survival_cliffiness"].sort_values("rmse", ascending=False).head(1)
    return [
        f"- Is cliffiness nonlinear in morphology space? {'Yes' if pd.notna(delta) and delta > 0.05 else 'Weak or no'}; kernel-minus-linear CV R2 delta={delta:.4f}.",
        f"- Are high-cliffiness regions coherent? Inspect the cliffiness field plot; max local gradient={float(high_grad.iloc[0].value) if not high_grad.empty else np.nan:.4f}.",
        f"- Do residuals cluster by experiment family after smoothing? Largest cliffiness residual family is {worst.iloc[0].experiment_family if not worst.empty else 'n/a'}.",
        f"- Does local sample density explain uncertainty or cliffiness? density-vs-residual-variance correlation={float(density_var.iloc[0].value) if not density_var.empty else np.nan:.4f}.",
        "- Do dropout/density/fragmentation perturbations occupy different arcs of the same manifold? Compare the family arc table and morphology-space plot; separation in mean breadth/elevation indicates distinct arcs on the pooled manifold.",
    ]


def plot_field(field: pd.DataFrame, target: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in [
        (axes[0], f"predicted_{target}", f"Predicted {target}"),
        (axes[1], f"gradient_{target}", "Gradient magnitude"),
        (axes[2], f"local_residual_variance_{target}", "Local residual variance"),
    ]:
        sc = ax.scatter(field["breadth"], field["elevation"], c=field[col], cmap="viridis", s=8)
        ax.set_xlabel("breadth")
        ax.set_ylabel("elevation")
        ax.set_title(title)
        fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_family_space(df: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for family, group in df.groupby("experiment_family"):
        ax.scatter(group["breadth"], group["elevation"], label=family, alpha=0.45, s=12)
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Family arcs in morphology space")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def build_report(messages, summary, perf, diag, residuals, arcs) -> str:
    lines = [
        "# Kernel-Smoothed Morphology Field",
        "",
        "## Inputs Used",
        *[f"- {m}" for m in messages],
        "",
        "## Dataset Summary",
        _markdown_table(summary),
        "",
        "## Linear vs Kernel Performance",
        _markdown_table(perf),
        "",
        "## Local Field Diagnostics",
        _markdown_table(diag),
        "",
        "## Residuals By Family",
        _markdown_table(residuals),
        "",
        "## Family Morphology Arcs",
        _markdown_table(arcs),
        "",
        "## Interpretation",
        *interpretation(perf, diag, residuals),
        "",
        "## Limitations",
        "- Kernel smoothing is diagnostic and sensitive to bandwidth/gamma.",
        "- Leave-family-out validation is intentionally harsh when families occupy different morphology arcs.",
        "- Local residual variance is estimated from nearest observed points, not a formal GP posterior variance.",
        "",
        "## Next Steps",
        "- Tune kernel bandwidth under nested grouped CV.",
        "- Compare with Gaussian process regression for calibrated uncertainty.",
        "- Add candidate state variables for families with clustered residuals.",
    ]
    return "\n".join(lines) + "\n"


def write_report(input_paths: list[Path], output_md: Path, gamma: float = 1.0, grid_size: int = 80) -> tuple[Path, ...]:
    df, messages = load_inputs(input_paths)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        output_md.write_text(build_report(messages, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()), encoding="utf-8")
        return (output_md,)
    perf, predictions = evaluate_smoothers(df, gamma=gamma)
    fields = []
    for target in TARGETS:
        field, _, _, _ = build_field_grid(df, target, gamma=gamma, grid_size=grid_size)
        fields.append(field)
    field = fields[0].merge(fields[1], on=["breadth", "elevation", "local_sample_density"], how="outer")
    diag = field_diagnostics(field)
    residuals = residuals_by_family(df, predictions)
    arcs = manifold_arcs(df)
    summary = pd.DataFrame([{"metric": "rows", "value": len(df)}, {"metric": "experiment_families", "value": df.experiment_family.nunique()}, {"metric": "models", "value": df.model.nunique()}])
    output_md.write_text(build_report(messages, summary, perf, diag, residuals, arcs), encoding="utf-8")
    return (
        output_md,
        plot_field(field, "minority_survival_cliffiness", output_md.parent / "kernel_morphology_cliffiness_field.png"),
        plot_field(field, "minority_survival_auc", output_md.parent / "kernel_morphology_survival_field.png"),
        plot_family_space(df, output_md.parent / "kernel_morphology_family_arcs.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "topology" / "kernel_smoothed_morphology_summary.md")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=80)
    args = parser.parse_args()
    for path in write_report(args.inputs, args.output_md, gamma=args.gamma, grid_size=args.grid_size):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
