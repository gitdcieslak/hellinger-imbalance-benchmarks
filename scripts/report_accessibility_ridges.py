"""Report local accessibility field geometry and ridge diagnostics."""

from __future__ import annotations

import argparse
from collections import deque
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
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "results" / "topology" / "density_threshold_sweep.csv",
    ROOT / "results" / "topology" / "density_separability_factorial_dense.csv",
    ROOT / "results" / "topology" / "fragmentation_sweep.csv",
    ROOT / "results" / "topology" / "weighted_dropout_sweep_dense.csv",
    ROOT / "results" / "topology" / "allocation_trajectory.csv",
]
TARGETS = ["minority_survival_cliffiness", "minority_survival_max_drop", "minority_survival_effective_drop_count"]
RIDGE_FEATURES = [
    "gradient_magnitude",
    "abs_laplacian",
    "max_abs_hessian_eigenvalue",
    "ridge_indicator",
    "ridge_strength",
    "family_ridge_density",
    "family_mean_ridge_strength",
    "family_max_ridge_strength",
]
EDGE_FEATURES = [
    "sobel_strength",
    "edge_indicator",
    "family_edge_pixel_fraction",
    "family_mean_edge_strength",
    "family_max_edge_strength",
    "family_connected_edge_components",
]
TOPOLOGY_FEATURES = ["n_islands", "minority_cluster_count", "positives_per_cluster", "component_entropy", "giant_component_fraction"]


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


def family_from_path(path: Path) -> str:
    return {"density_separability_factorial_dense": "density_separability"}.get(path.stem, path.stem.replace("_sweep", ""))


def normalize_file(path: Path) -> tuple[pd.DataFrame | None, str]:
    if not path.exists():
        return None, f"missing: {path}"
    raw = pd.read_csv(path).rename(columns={"positive_histogram_entropy": "breadth", "positive_top_bin_mass": "elevation"})
    family = family_from_path(path)
    model_col = next((c for c in ["model_id", "objective", "model_name"] if c in raw.columns), None)
    if model_col is None and "dropout_rate" in raw.columns:
        model = "dropout_" + raw["dropout_rate"].astype(str)
    elif model_col is not None:
        model = raw[model_col].astype(str)
    else:
        model = pd.Series(["unknown"] * len(raw))
    required = ["breadth", "elevation", "minority_survival_auc", "minority_survival_cliffiness"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        return None, f"skipped incomplete: {path} missing {', '.join(missing)}"
    out = pd.DataFrame({
        "experiment_family": family,
        "model_id": model,
        "seed": raw["seed"].astype(str) if "seed" in raw.columns else np.arange(len(raw)).astype(str),
        "breadth": pd.to_numeric(raw["breadth"], errors="coerce"),
        "elevation": pd.to_numeric(raw["elevation"], errors="coerce"),
        "minority_survival_auc": pd.to_numeric(raw["minority_survival_auc"], errors="coerce"),
        "minority_survival_cliffiness": pd.to_numeric(raw["minority_survival_cliffiness"], errors="coerce"),
        "minority_survival_max_drop": pd.to_numeric(raw["minority_survival_max_drop"], errors="coerce") if "minority_survival_max_drop" in raw.columns else np.nan,
        "minority_survival_effective_drop_count": pd.to_numeric(raw["minority_survival_effective_drop_count"], errors="coerce") if "minority_survival_effective_drop_count" in raw.columns else np.nan,
        "auroc": pd.to_numeric(raw["auroc"], errors="coerce") if "auroc" in raw.columns else np.nan,
        "average_precision": pd.to_numeric(raw["average_precision"], errors="coerce") if "average_precision" in raw.columns else np.nan,
    })
    for col in ["minority_cov", "centroid_distance", "n_islands", "island_cov", "dropout_rate", "skew_ratio", "epoch"] + TOPOLOGY_FEATURES:
        out[col] = pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else np.nan
    if "positives_per_cluster" not in out or out["positives_per_cluster"].isna().all():
        if "n_islands" in out and "minority_count" in raw.columns:
            out["positives_per_cluster"] = pd.to_numeric(raw["minority_count"], errors="coerce") / out["n_islands"]
    return out.dropna(subset=["breadth", "elevation", "minority_survival_auc", "minority_survival_cliffiness"]), f"used: {path} rows={len(out)}"


def load_inputs(paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    frames, messages = [], []
    for path in paths:
        frame, msg = normalize_file(path)
        messages.append(msg)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), messages
    return pd.concat(frames, ignore_index=True).drop_duplicates(), messages


def grid_axes_for_family(family: str) -> tuple[str, str] | None:
    if family in {"density_threshold", "density_separability"}:
        return "minority_cov", "centroid_distance"
    if family == "weighted_dropout_dense":
        return "dropout_rate", "skew_ratio"
    if family == "fragmentation":
        return "n_islands", "positives_per_cluster"
    return None


def _connected_components(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not mask[i, j] or seen[i, j]:
                continue
            count += 1
            q = deque([(i, j)])
            seen[i, j] = True
            while q:
                x, y = q.popleft()
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and mask[nx, ny] and not seen[nx, ny]:
                            seen[nx, ny] = True
                            q.append((nx, ny))
    return count


def _interior_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if shape[0] > 2 and shape[1] > 2:
        mask[1:-1, 1:-1] = True
    else:
        mask[:, :] = True
    return mask


def _sobel(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded = np.pad(z, 1, mode="edge")
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    ky = kx.T
    gx = np.zeros_like(z, dtype=float)
    gy = np.zeros_like(z, dtype=float)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            window = padded[i:i + 3, j:j + 3]
            gx[i, j] = float(np.sum(window * kx))
            gy[i, j] = float(np.sum(window * ky))
    mag = np.sqrt(gx * gx + gy * gy)
    return gx, gy, mag


def _threshold_positive_signal(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or float(np.max(arr)) <= 0.0:
        return float("inf")
    return float(np.mean(arr) + np.std(arr, ddof=0))


def _curvature_sign_change(lap: np.ndarray) -> np.ndarray:
    sign = np.sign(lap)
    changes = np.zeros_like(lap, dtype=bool)
    for i in range(lap.shape[0]):
        for j in range(lap.shape[1]):
            local = sign[max(0, i - 1): i + 2, max(0, j - 1): j + 2]
            vals = local[np.isfinite(local)]
            changes[i, j] = np.any(vals > 0) and np.any(vals < 0)
    return changes


def compute_surface_geometry(z: np.ndarray, x_values: np.ndarray | None = None, y_values: np.ndarray | None = None) -> dict[str, np.ndarray | float | int]:
    z_arr = np.asarray(z, dtype=float)
    if x_values is None:
        x_values = np.arange(z_arr.shape[0], dtype=float)
    if y_values is None:
        y_values = np.arange(z_arr.shape[1], dtype=float)
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    edge_order = 2 if z_arr.shape[0] >= 3 and z_arr.shape[1] >= 3 else 1
    dx, dy = np.gradient(z_arr, x_arr, y_arr, edge_order=edge_order)
    grad = np.sqrt(dx * dx + dy * dy)
    dxx = np.gradient(dx, x_arr, axis=0, edge_order=edge_order)
    dyy = np.gradient(dy, y_arr, axis=1, edge_order=edge_order)
    dxy = np.gradient(dx, y_arr, axis=1, edge_order=edge_order)
    lap = dxx + dyy
    trace = dxx + dyy
    det = dxx * dyy - dxy * dxy
    disc = np.maximum(0.0, trace * trace - 4.0 * det)
    eig1 = 0.5 * (trace + np.sqrt(disc))
    eig2 = 0.5 * (trace - np.sqrt(disc))
    maxeig = np.maximum(np.abs(eig1), np.abs(eig2))
    _, _, sobel = _sobel(z_arr)
    interior = _interior_mask(z_arr.shape)
    grad_threshold = _threshold_positive_signal(grad[interior])
    sobel_threshold = _threshold_positive_signal(sobel[interior])
    ridge = (grad >= grad_threshold) & _curvature_sign_change(lap) & interior
    edge = (sobel >= sobel_threshold) & interior
    ridge_strength = grad * maxeig
    return {
        "dx": dx,
        "dy": dy,
        "gradient_magnitude": grad,
        "dxx": dxx,
        "dyy": dyy,
        "dxy": dxy,
        "laplacian": lap,
        "max_abs_hessian_eigenvalue": maxeig,
        "ridge_mask": ridge,
        "ridge_strength": ridge_strength,
        "sobel_strength": sobel,
        "edge_mask": edge,
        "gradient_threshold": grad_threshold,
        "edge_threshold": sobel_threshold,
        "interior_pixel_count": int(np.sum(interior)),
        "edge_pixel_count": int(np.sum(edge)),
        "ridge_pixel_count": int(np.sum(ridge)),
        "connected_edge_components": int(_connected_components(edge)),
    }


def _array_preview(arr: np.ndarray, max_rows: int = 8, max_cols: int = 8) -> str:
    sub = np.asarray(arr)[:max_rows, :max_cols]
    return np.array2string(sub, precision=3, suppress_small=True, max_line_width=200).replace("\n", " ")


def compute_field_geometry(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, summaries = [], []
    for family, group in df.groupby("experiment_family"):
        axes = grid_axes_for_family(family)
        if axes is None:
            summaries.append({"experiment_family": family, "status": "skipped_no_regular_generating_grid"})
            continue
        x_col, y_col = axes
        if group[x_col].nunique(dropna=True) < 2 or group[y_col].nunique(dropna=True) < 2:
            summaries.append({"experiment_family": family, "status": "skipped_insufficient_grid_axes"})
            continue
        mean_grid = group.groupby([x_col, y_col], as_index=False)["minority_survival_auc"].mean()
        pivot = mean_grid.pivot(index=x_col, columns=y_col, values="minority_survival_auc").sort_index().sort_index(axis=1)
        if pivot.isna().any().any():
            pivot = pivot.interpolate(axis=0).interpolate(axis=1).ffill().bfill()
        z = pivot.to_numpy(dtype=float)
        if z.shape[0] < 2 or z.shape[1] < 2:
            summaries.append({"experiment_family": family, "status": "skipped_grid_too_small"})
            continue
        x_values = pivot.index.to_numpy(dtype=float)
        y_values = pivot.columns.to_numpy(dtype=float)
        geometry = compute_surface_geometry(z, x_values, y_values)
        grad = geometry["gradient_magnitude"]
        lap = geometry["laplacian"]
        maxeig = geometry["max_abs_hessian_eigenvalue"]
        ridge = geometry["ridge_mask"]
        ridge_strength = geometry["ridge_strength"]
        sobel = geometry["sobel_strength"]
        edge = geometry["edge_mask"]
        interior_count = max(1, int(geometry["interior_pixel_count"]))
        summaries.append({
            "experiment_family": family,
            "status": "used",
            "grid_shape": f"{z.shape[0]}x{z.shape[1]}",
            "x_axis": x_col,
            "y_axis": y_col,
            "x_values": ",".join(f"{v:g}" for v in x_values),
            "y_values": ",".join(f"{v:g}" for v in y_values),
            "gradient_threshold": float(geometry["gradient_threshold"]),
            "edge_threshold": float(geometry["edge_threshold"]),
            "family_ridge_density": float(np.sum(ridge) / interior_count),
            "family_mean_ridge_strength": float(np.mean(ridge_strength[ridge])) if np.any(ridge) else 0.0,
            "family_max_ridge_strength": float(np.max(ridge_strength)) if ridge_strength.size else 0.0,
            "family_edge_pixel_fraction": float(np.sum(edge) / interior_count),
            "family_mean_edge_strength": float(np.mean(sobel[edge])) if np.any(edge) else 0.0,
            "family_max_edge_strength": float(np.max(sobel)) if sobel.size else 0.0,
            "family_connected_edge_components": int(_connected_components(edge)),
            "raw_grid_preview": _array_preview(z),
            "gradient_preview": _array_preview(grad),
            "edge_mask_preview": _array_preview(edge.astype(int)),
            "ridge_mask_preview": _array_preview(ridge.astype(int)),
        })
        for i, x in enumerate(pivot.index):
            for j, y in enumerate(pivot.columns):
                rows.append({
                    "experiment_family": family,
                    x_col: float(x),
                    y_col: float(y),
                    "gradient_magnitude": float(grad[i, j]),
                    "laplacian": float(lap[i, j]),
                    "abs_laplacian": float(abs(lap[i, j])),
                    "max_abs_hessian_eigenvalue": float(maxeig[i, j]),
                    "ridge_indicator": float(ridge[i, j]),
                    "ridge_strength": float(ridge_strength[i, j]),
                    "sobel_strength": float(sobel[i, j]),
                    "edge_indicator": float(edge[i, j]),
                })
    geom = pd.DataFrame(rows)
    summaries_df = pd.DataFrame(summaries)
    if geom.empty:
        return geom, summaries_df
    merged_parts = []
    geometry_cols = [
        "gradient_magnitude",
        "laplacian",
        "abs_laplacian",
        "max_abs_hessian_eigenvalue",
        "ridge_indicator",
        "ridge_strength",
        "sobel_strength",
        "edge_indicator",
    ]
    for family in geom.experiment_family.unique():
        axes = grid_axes_for_family(family)
        if axes is None:
            continue
        x_col, y_col = axes
        fam_geom = geom[geom.experiment_family == family][["experiment_family", x_col, y_col, *geometry_cols]]
        fam_summary = summaries_df[summaries_df.experiment_family == family].iloc[0].to_dict()
        part = df[df.experiment_family == family].merge(fam_geom, on=["experiment_family", x_col, y_col], how="left")
        for col in ["family_ridge_density", "family_mean_ridge_strength", "family_max_ridge_strength", "family_edge_pixel_fraction", "family_mean_edge_strength", "family_max_edge_strength", "family_connected_edge_components"]:
            part[col] = fam_summary.get(col, np.nan)
        merged_parts.append(part)
    used_families = set(geom.experiment_family.unique())
    remainder = df[~df.experiment_family.isin(used_families)].copy()
    if not remainder.empty:
        merged_parts.append(remainder)
    return pd.concat(merged_parts, ignore_index=True) if merged_parts else df, summaries_df


def make_pipeline(features: list[str]) -> Pipeline:
    return Pipeline([("prep", ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), features)])), ("ridge", Ridge(alpha=1.0))])


def feature_sets(mode: str) -> dict[str, list[str]]:
    ridge_edge = RIDGE_FEATURES if mode == "ridge" else EDGE_FEATURES
    return {
        "morphology": ["breadth", "elevation"],
        "morphology_topology": ["breadth", "elevation", *TOPOLOGY_FEATURES],
        "morphology_ridge": ["breadth", "elevation", *RIDGE_FEATURES],
        "morphology_edge": ["breadth", "elevation", *EDGE_FEATURES],
        "morphology_ridge_edge": ["breadth", "elevation", *RIDGE_FEATURES, *EDGE_FEATURES],
        f"morphology_{mode}": ["breadth", "elevation", *ridge_edge],
    }


def evaluate_models(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows, preds = [], {}
    sets = feature_sets(mode)
    for target in TARGETS:
        if target not in df.columns or df[target].isna().all():
            continue
        data = df.dropna(subset=[target]).copy()
        if data.experiment_family.nunique() >= 2:
            cv = GroupKFold(n_splits=min(5, data.experiment_family.nunique()))
            groups = data.experiment_family
        else:
            continue
        y = data[target].to_numpy(float)
        for name, feats in sets.items():
            pipe = make_pipeline(feats)
            try:
                pred = cross_val_predict(pipe, data, y, cv=cv, groups=groups)
            except Exception:
                pred = np.full_like(y, y.mean())
            pipe.fit(data, y)
            ins = pipe.predict(data)
            rows.append({"target": target, "feature_set": name, "cv_r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else 0.0, "cv_mae": float(mean_absolute_error(y, pred)), "cv_rmse": float(np.sqrt(mean_squared_error(y, pred))), "in_sample_r2": float(r2_score(y, ins)) if len(np.unique(y)) > 1 else 0.0, "n_rows": len(data)})
            if target == "minority_survival_cliffiness" and name == f"morphology_{mode}":
                full = np.full(len(df), np.nan)
                full[data.index.to_numpy()] = pred
                preds[f"{target}_{mode}"] = full
    return pd.DataFrame(rows), preds


def improvement_label(delta: float) -> str:
    if delta > 0.10:
        return "strong evidence"
    if delta > 0.05:
        return "moderate evidence"
    return "weak evidence"


def improvement_summary(perf: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        base = perf[(perf.target == target) & (perf.feature_set == "morphology")]
        aug = perf[(perf.target == target) & (perf.feature_set == f"morphology_{mode}")]
        if base.empty or aug.empty:
            continue
        delta = float(aug.iloc[0].cv_r2) - float(base.iloc[0].cv_r2)
        rows.append({"target": target, "morphology_cv_r2": float(base.iloc[0].cv_r2), f"morphology_{mode}_cv_r2": float(aug.iloc[0].cv_r2), "delta_cv_r2": delta, "evidence": improvement_label(delta)})
    return pd.DataFrame(rows)


def plot_surfaces(df: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    families = [f for f in df.experiment_family.unique() if grid_axes_for_family(f) is not None][:4]
    fig, axes = plt.subplots(1, max(1, len(families)), figsize=(5 * max(1, len(families)), 4))
    axes = np.atleast_1d(axes)
    for ax, family in zip(axes, families, strict=False):
        x_col, y_col = grid_axes_for_family(family)
        g = df[df.experiment_family == family].groupby([x_col, y_col], as_index=False).minority_survival_auc.mean()
        pivot = g.pivot(index=x_col, columns=y_col, values="minority_survival_auc").sort_index().sort_index(axis=1)
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)), labels=[f"{v:g}" for v in pivot.columns], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)), labels=[f"{v:g}" for v in pivot.index])
        ax.set_title(family)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.colorbar(image, ax=ax, label="survival_auc")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_geometry(df: pd.DataFrame, feature: str, output: Path, title: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = df.dropna(subset=[feature, "breadth", "elevation"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(data.breadth, data.elevation, c=data[feature], cmap="magma", alpha=0.75)
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=feature)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_prediction(df: pd.DataFrame, preds: dict[str, np.ndarray], mode: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    pred = preds.get(f"minority_survival_cliffiness_{mode}", np.full(len(df), np.nan))
    data = df.assign(predicted=pred).dropna(subset=["minority_survival_cliffiness", "predicted"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for fam, g in data.groupby("experiment_family"):
        ax.scatter(g.minority_survival_cliffiness, g.predicted, label=fam, alpha=0.75)
    if not data.empty:
        lo = min(data.minority_survival_cliffiness.min(), data.predicted.min())
        hi = max(data.minority_survival_cliffiness.max(), data.predicted.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("actual cliffiness")
    ax.set_ylabel("predicted cliffiness")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def build_report(title: str, mode: str, messages, field_summary, perf, improvement) -> str:
    primary = improvement[improvement.target == "minority_survival_cliffiness"]
    if primary.empty:
        answer = "not estimable"
    else:
        row = primary.iloc[0]
        answer = f"{row.evidence}; delta CV R2={float(row.delta_cv_r2):.4f}"
    lines = [
        f"# {title}",
        "",
        "## Inputs Used",
        *[f"- {m}" for m in messages],
        "",
        "## Accessibility Field Summary",
        _markdown_table(field_summary),
        "",
        "## Predictive Performance",
        _markdown_table(perf),
        "",
        "## Improvement Summary",
        _markdown_table(improvement),
        "",
        "## Interpretation",
        f"- Does local {mode} geometry improve cliffiness prediction beyond breadth/elevation? {answer}.",
        "- If improvement is weak, current local field geometry features do not yet explain accessibility dynamics beyond morphology in a leave-family-out ridge test.",
        "",
        "## Limitations",
        "- Field geometry requires regular numeric generating grids; families without such grids are skipped for derivative construction.",
        "- Finite-difference derivatives depend on grid resolution and smoothing choices.",
        "- These are linear ridge diagnostics, not causal surface estimates.",
        "",
        "## Next Steps",
        "- Add finer grids and repeated surfaces for each family.",
        "- Compare finite differences with smoothed Gaussian-process or spline surfaces.",
        "- Evaluate nonlinear models only after linear geometry diagnostics stabilize.",
    ]
    return "\n".join(lines) + "\n"


def write_report(mode: str, input_paths: list[Path], output_md: Path) -> tuple[Path, ...]:
    df, messages = load_inputs(input_paths)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        output_md.write_text(build_report("Accessibility Ridge Summary" if mode == "ridge" else "Accessibility Edge Summary", mode, messages, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()), encoding="utf-8")
        return (output_md,)
    geom_df, field_summary = compute_field_geometry(df)
    if geom_df.empty:
        geom_df = df
    perf, preds = evaluate_models(geom_df, mode)
    improvement = improvement_summary(perf, mode)
    title = "Accessibility Ridge Summary" if mode == "ridge" else "Accessibility Edge Summary"
    output_md.write_text(build_report(title, mode, messages, field_summary, perf, improvement), encoding="utf-8")
    if mode == "ridge":
        return (
            output_md,
            plot_surfaces(df, output_md.parent / "accessibility_surface_examples.png"),
            plot_geometry(geom_df, "gradient_magnitude", output_md.parent / "accessibility_gradient_fields.png", "Accessibility Gradient Magnitude"),
            plot_geometry(geom_df, "ridge_strength", output_md.parent / "accessibility_ridges.png", "Accessibility Ridge Strength"),
            plot_prediction(geom_df, preds, mode, output_md.parent / "ridge_prediction_vs_actual.png"),
        )
    return (
        output_md,
        plot_surfaces(df, output_md.parent / "accessibility_surface_examples.png"),
        plot_geometry(geom_df, "sobel_strength", output_md.parent / "accessibility_edges.png", "Accessibility Edge Strength"),
        plot_prediction(geom_df, preds, mode, output_md.parent / "edge_prediction_vs_actual.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ridge", "edge"], default="ridge")
    parser.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "topology" / "accessibility_ridge_summary.md")
    args = parser.parse_args()
    for path in write_report(args.mode, args.inputs, args.output_md):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
