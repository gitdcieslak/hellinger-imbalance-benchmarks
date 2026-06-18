"""Report morphology manifold coordinates for accessibility outcomes."""

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
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_morphology_ridge_extraction import DEFAULT_INPUTS, load_inputs  # noqa: E402


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


def add_pca_coordinates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    coords = out[["breadth", "elevation"]].to_numpy(dtype=float)
    scaler = StandardScaler().fit(coords)
    standardized = scaler.transform(coords)
    pca = PCA(n_components=2).fit(standardized)
    scores = pca.transform(standardized)
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1]
    out["global_pc1_position"] = MinMaxScaler().fit_transform(out[["pc1"]]).ravel()
    kernel_scores = KernelPCA(n_components=1, kernel="rbf", gamma=1.0, random_state=7).fit_transform(standardized)
    out["kernel_pc1_position"] = MinMaxScaler().fit_transform(kernel_scores).ravel()
    diagnostics = pd.DataFrame(
        [
            {"diagnostic": "pc1_explained_variance_ratio", "value": float(pca.explained_variance_ratio_[0])},
            {"diagnostic": "pc2_explained_variance_ratio", "value": float(pca.explained_variance_ratio_[1])},
            {"diagnostic": "breadth_elevation_correlation", "value": float(np.corrcoef(coords[:, 0], coords[:, 1])[0, 1]) if np.std(coords[:, 0]) > 0 and np.std(coords[:, 1]) > 0 else 0.0},
            {"diagnostic": "pc1_loading_breadth", "value": float(pca.components_[0, 0])},
            {"diagnostic": "pc1_loading_elevation", "value": float(pca.components_[0, 1])},
        ]
    )
    return out, diagnostics


def add_arc_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "global_pc1_position" not in out.columns:
        out["global_pc1_position"] = MinMaxScaler().fit_transform(out[["pc1"]]).ravel()
    out["family_arc_position"] = np.nan
    out["global_arc_position"] = np.nan
    for family, idx in out.groupby("experiment_family").groups.items():
        part = out.loc[idx].sort_values("pc1")
        coords = part[["pc1", "pc2"]].to_numpy(dtype=float)
        if len(part) <= 1:
            arc = np.zeros(len(part), dtype=float)
        else:
            distances = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
            arc = np.concatenate([[0.0], np.cumsum(distances)])
            if arc[-1] > 0:
                arc = arc / arc[-1]
        out.loc[part.index, "family_arc_position"] = arc
    global_part = out.sort_values("pc1")
    coords = global_part[["pc1", "pc2"]].to_numpy(dtype=float)
    if len(global_part) <= 1:
        arc = np.zeros(len(global_part), dtype=float)
    else:
        distances = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        arc = np.concatenate([[0.0], np.cumsum(distances)])
        if arc[-1] > 0:
            arc = arc / arc[-1]
    out.loc[global_part.index, "global_arc_position"] = arc
    out["accessibility_coordinate_pc1"] = out["global_pc1_position"]
    out["accessibility_coordinate_arc"] = out["global_arc_position"]
    if "kernel_pc1_position" in out.columns:
        out["accessibility_coordinate_kernel"] = out["kernel_pc1_position"]
    else:
        out["accessibility_coordinate_kernel"] = 0.5 * out["global_pc1_position"] + 0.5 * out["family_arc_position"]
    return out


def prepare_dataset(input_paths: list[Path]) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    df, messages = load_inputs(input_paths)
    if df.empty:
        return df, messages, pd.DataFrame()
    df, diagnostics = add_pca_coordinates(df)
    df = add_arc_coordinates(df)
    return df, messages, diagnostics


def _pipeline(features: list[str], kernel: bool = False) -> Pipeline:
    model = KernelRidge(alpha=1e-2, kernel="rbf", gamma=1.0) if kernel else Ridge(alpha=1.0)
    return Pipeline(
        [
            ("prep", ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)])),
            ("model", model),
        ]
    )


def _cv(df: pd.DataFrame):
    if df["experiment_family"].nunique() >= 3:
        return GroupKFold(min(5, df["experiment_family"].nunique())), df["experiment_family"], "experiment_family"
    if df["seed"].nunique() >= 3:
        return GroupKFold(min(5, df["seed"].nunique())), df["seed"], "seed"
    return KFold(min(5, len(df)), shuffle=True, random_state=7), None, "row"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_feature_sets(df: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "breadth_elevation": (["breadth", "elevation"], False),
        "pc1_only": (["pc1"], False),
        "pc1_pc2": (["pc1", "pc2"], False),
        "kernel_breadth_elevation": (["breadth", "elevation"], True),
        "kernel_pc1_pc2": (["pc1", "pc2"], True),
    }
    rows = []
    for target in TARGETS:
        data = df.dropna(subset=[target]).copy()
        y = data[target].to_numpy(dtype=float)
        cv, groups, group_name = _cv(data)
        for name, (features, kernel) in specs.items():
            pipe = _pipeline(features, kernel=kernel)
            pred = cross_val_predict(pipe, data, y, cv=cv, groups=groups) if groups is not None else cross_val_predict(pipe, data, y, cv=cv)
            pipe.fit(data, y)
            ins = pipe.predict(data)
            rows.append({"target": target, "feature_set": name, "cv_group": group_name, "cv_r2": float(r2_score(y, pred)), "cv_mae": float(mean_absolute_error(y, pred)), "cv_rmse": _rmse(y, pred), "in_sample_r2": float(r2_score(y, ins)), "n_rows": len(data)})
    return pd.DataFrame(rows)


def evaluate_scalar_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "family_arc_position": ["family_arc_position"],
        "global_pc1_position": ["global_pc1_position"],
        "global_arc_position": ["global_arc_position"],
        "accessibility_coordinate_pc1": ["accessibility_coordinate_pc1"],
        "accessibility_coordinate_arc": ["accessibility_coordinate_arc"],
        "accessibility_coordinate_kernel": ["accessibility_coordinate_kernel"],
    }
    rows = []
    for coord, features in specs.items():
        target_scores = []
        for target in TARGETS:
            data = df.dropna(subset=[target, *features]).copy()
            y = data[target].to_numpy(dtype=float)
            cv, groups, group_name = _cv(data)
            pipe = _pipeline(features, kernel=False)
            pred = cross_val_predict(pipe, data, y, cv=cv, groups=groups) if groups is not None else cross_val_predict(pipe, data, y, cv=cv)
            score = float(r2_score(y, pred))
            target_scores.append(score)
            rows.append({"coordinate": coord, "target": target, "cv_group": group_name, "cv_r2": score, "cv_mae": float(mean_absolute_error(y, pred)), "cv_rmse": _rmse(y, pred), "mean_joint_score": np.nan})
        rows.append({"coordinate": coord, "target": "joint", "cv_group": group_name, "cv_r2": np.nan, "cv_mae": np.nan, "cv_rmse": np.nan, "mean_joint_score": float(np.mean(target_scores))})
    return pd.DataFrame(rows)


def family_arc_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in df.groupby("experiment_family"):
        rows.append({"experiment_family": family, "n_rows": len(group), "pc1_min": float(group["pc1"].min()), "pc1_max": float(group["pc1"].max()), "pc2_std": float(group["pc2"].std(ddof=0)), "mean_family_arc_position": float(group["family_arc_position"].mean()), "mean_survival_auc": float(group["minority_survival_auc"].mean()), "mean_cliffiness": float(group["minority_survival_cliffiness"].mean())})
    return pd.DataFrame(rows)


def interpretation(pca_diag: pd.DataFrame, exp1: pd.DataFrame, exp3: pd.DataFrame, arcs: pd.DataFrame) -> list[str]:
    pc1_var = float(pca_diag[pca_diag.diagnostic == "pc1_explained_variance_ratio"].iloc[0].value) if not pca_diag.empty else np.nan
    def score(target: str, fs: str) -> float:
        row = exp1[(exp1.target == target) & (exp1.feature_set == fs)]
        return float(row.iloc[0].cv_r2) if not row.empty else np.nan
    pc1_surv = score("minority_survival_auc", "pc1_only")
    pc1_cliff = score("minority_survival_cliffiness", "pc1_only")
    pc2_surv_delta = score("minority_survival_auc", "pc1_pc2") - pc1_surv
    pc2_cliff_delta = score("minority_survival_cliffiness", "pc1_pc2") - pc1_cliff
    kernel_delta = score("minority_survival_cliffiness", "kernel_pc1_pc2") - score("minority_survival_cliffiness", "pc1_pc2")
    family_arc = exp3[(exp3.coordinate == "family_arc_position") & (exp3.target == "minority_survival_cliffiness")]
    global_pc1 = exp3[(exp3.coordinate == "global_pc1_position") & (exp3.target == "minority_survival_cliffiness")]
    family_better = not family_arc.empty and not global_pc1.empty and float(family_arc.iloc[0].cv_r2) > float(global_pc1.iloc[0].cv_r2)
    best_joint = exp3[exp3.target == "joint"].sort_values("mean_joint_score", ascending=False).head(1)
    best_coord = best_joint.iloc[0].coordinate if not best_joint.empty else "n/a"
    best_joint_score = float(best_joint.iloc[0].mean_joint_score) if not best_joint.empty else np.nan
    best_survival = exp3[(exp3.coordinate == best_coord) & (exp3.target == "minority_survival_auc")]
    best_cliff = exp3[(exp3.coordinate == best_coord) & (exp3.target == "minority_survival_cliffiness")]
    best_survival_score = float(best_survival.iloc[0].cv_r2) if not best_survival.empty else np.nan
    best_cliff_score = float(best_cliff.iloc[0].cv_r2) if not best_cliff.empty else np.nan
    scalar_plausible = best_survival_score > 0.5 and best_cliff_score > 0.25
    deviating = arcs.assign(pc2_abs=arcs.pc2_std.abs()).sort_values("pc2_abs", ascending=False).head(3).experiment_family.tolist()
    return [
        f"- Is morphology space approximately one-dimensional? {'Yes' if pc1_var >= 0.80 else 'No'}; PC1 explains {pc1_var:.4f} of standardized breadth/elevation variance.",
        f"- Does PC1 alone predict survival AUC? {'Yes' if pc1_surv > 0.50 else 'Weak'}; CV R2={pc1_surv:.4f}.",
        f"- Does PC1 alone predict cliffiness? {'Yes' if pc1_cliff > 0.25 else 'Weak'}; CV R2={pc1_cliff:.4f}.",
        f"- Does adding PC2 improve predictions? survival delta={pc2_surv_delta:.4f}, cliffiness delta={pc2_cliff_delta:.4f}.",
        f"- Does nonlinear kernel prediction outperform linear manifold coordinates? {'Yes' if kernel_delta > 0.05 else 'No'} for cliffiness; kernel-minus-linear PC delta={kernel_delta:.4f}.",
        f"- Do experiment families trace distinct arcs? Yes if their PC ranges/PC2 spread differ; see Family Arc Summary. Largest PC2 spread families: {', '.join(deviating)}.",
        f"- Is family-local arc position more predictive than global PC1? {'Yes, but only marginally and both are poor for cliffiness' if family_better else 'No'} in this run.",
        f"- Is there evidence for a universal scalar accessibility coordinate? {'Yes, provisionally' if scalar_plausible else 'No, not yet for both outcomes'}; best joint scalar is {best_coord} with mean joint CV R2={best_joint_score:.4f}, survival CV R2={best_survival_score:.4f}, cliffiness CV R2={best_cliff_score:.4f}.",
        f"- Which families deviate most from the shared manifold? {', '.join(deviating)} by within-family PC2 spread.",
    ]


def plot_space(df: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for family, group in df.groupby("experiment_family"):
        ax.scatter(group.breadth, group.elevation, label=family, alpha=0.45, s=12)
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Morphology manifold space")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_pc(df: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df.pc1, df.pc2, c=df.minority_survival_cliffiness, cmap="magma", alpha=0.55, s=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Morphology PC scores")
    fig.colorbar(sc, ax=ax, label="cliffiness")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_arc_paths(df: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for family, group in df.sort_values("pc1").groupby("experiment_family"):
        ax.plot(group.pc1, group.pc2, marker="o", markersize=2, linewidth=1, label=family, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Family arcs through morphology manifold")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_target_vs_coordinate(df: pd.DataFrame, target: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for family, group in df.groupby("experiment_family"):
        ax.scatter(group.accessibility_coordinate_kernel, group[target], label=family, alpha=0.5, s=12)
    ax.set_xlabel("accessibility_coordinate_kernel")
    ax.set_ylabel(target)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def build_report(messages, summary, pca_diag, exp1, exp2, exp3, arcs, interp) -> str:
    lines = [
        "# Morphology Manifold Experiments",
        "",
        "## Inputs Used",
        *[f"- {m}" for m in messages],
        "",
        "## Dataset Summary",
        _markdown_table(summary),
        "",
        "## PCA / Manifold Diagnostics",
        _markdown_table(pca_diag),
        "",
        "## Experiment 1: Manifold Coordinates",
        _markdown_table(exp1),
        "",
        "## Experiment 2: Arc Length",
        _markdown_table(exp2),
        "",
        "## Experiment 3: Universal Accessibility Coordinate",
        _markdown_table(exp3),
        "",
        "## Family Arc Summary",
        _markdown_table(arcs),
        "",
        "## Interpretation",
        *interp,
        "",
        "## Limitations",
        "- PCA coordinates are linear projections of breadth/elevation only.",
        "- Arc positions are empirical orderings, not learned geodesics.",
        "- Kernel coordinates are diagnostic and require bandwidth sensitivity checks.",
        "",
        "## Next Steps",
        "- Evaluate nonlinear manifold learning only after scalar-coordinate baselines stabilize.",
        "- Add candidate state variables for families with high PC2 spread or poor scalar-coordinate fit.",
        "- Test coordinate stability under held-out experiment designs.",
    ]
    return "\n".join(lines) + "\n"


def write_report(input_paths: list[Path], output_md: Path) -> tuple[Path, ...]:
    df, messages, pca_diag = prepare_dataset(input_paths)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        output_md.write_text(build_report(messages, pd.DataFrame(), pca_diag, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []), encoding="utf-8")
        return (output_md,)
    exp1 = evaluate_feature_sets(df)
    exp2 = evaluate_scalar_coordinates(df[df["family_arc_position"].notna()].copy())
    exp2 = exp2[exp2.coordinate.isin(["family_arc_position", "global_pc1_position", "global_arc_position"])]
    exp3 = evaluate_scalar_coordinates(df)
    arcs = family_arc_summary(df)
    summary = pd.DataFrame([{"metric": "rows", "value": len(df)}, {"metric": "experiment_families", "value": df.experiment_family.nunique()}, {"metric": "models", "value": df.model.nunique()}])
    interp = interpretation(pca_diag, exp1, exp3, arcs)
    output_md.write_text(build_report(messages, summary, pca_diag, exp1, exp2, exp3, arcs, interp), encoding="utf-8")
    return (
        output_md,
        plot_space(df, output_md.parent / "morphology_manifold_space.png"),
        plot_pc(df, output_md.parent / "morphology_manifold_pc_scores.png"),
        plot_arc_paths(df, output_md.parent / "morphology_manifold_arc_paths.png"),
        plot_target_vs_coordinate(df, "minority_survival_auc", output_md.parent / "morphology_manifold_survival_vs_coordinate.png"),
        plot_target_vs_coordinate(df, "minority_survival_cliffiness", output_md.parent / "morphology_manifold_cliffiness_vs_coordinate.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "topology" / "morphology_manifold_summary.md")
    args = parser.parse_args()
    for path in write_report(args.inputs, args.output_md):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
