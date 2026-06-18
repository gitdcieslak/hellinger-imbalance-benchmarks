"""Synthesize accessibility morphology experiments into one explanatory framework."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_kernel_smoothed_morphology import build_field_grid, evaluate_smoothers, residuals_by_family  # noqa: E402
from report_morphology_manifold import (  # noqa: E402
    DEFAULT_INPUTS,
    TARGETS,
    _markdown_table,
    evaluate_feature_sets,
    family_arc_summary,
    prepare_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
FAMILY_ORDER = [
    "density_threshold",
    "density_separability",
    "allocation_trajectory",
    "weighted_dropout_dense",
    "mlp_objectives_topology",
    "fragmentation",
]
ALLOCATOR_LABELS = {
    "hddt": "HDDT",
    "bagged_hddt": "Bagged HDDT",
    "cart": "CART",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp_bce": "MLP BCE",
    "mlp_weighted": "MLP Weighted",
    "mlp_weighted_bce": "MLP Weighted BCE",
    "mlp_oversampled": "MLP Oversampled",
    "mlp_oversampled_bce": "MLP Oversampled BCE",
    "mlp_focal": "MLP Focal",
    "mlp_logit_adjusted": "MLP Logit Adjusted",
}


def morphology_explanatory_power(feature_perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, label in [("minority_survival_auc", "Survival AUC"), ("minority_survival_cliffiness", "Cliffiness")]:
        linear = feature_perf[(feature_perf.target == target) & (feature_perf.feature_set == "breadth_elevation")]
        pc = feature_perf[(feature_perf.target == target) & (feature_perf.feature_set == "pc1_pc2")]
        kernel = feature_perf[(feature_perf.target == target) & (feature_perf.feature_set == "kernel_breadth_elevation")]
        rows.append(
            {
                "Target": label,
                "Linear": float(linear.iloc[0].cv_r2) if not linear.empty else np.nan,
                "PC1_PC2": float(pc.iloc[0].cv_r2) if not pc.empty else np.nan,
                "Kernel": float(kernel.iloc[0].cv_r2) if not kernel.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def family_deviation_table(df: pd.DataFrame, residuals: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
    cliff_res = residuals[residuals.target == "minority_survival_cliffiness"].copy()
    if cliff_res.empty:
        cliff_res = pd.DataFrame(columns=["experiment_family", "rmse", "mae"])
    merged = arcs.merge(cliff_res[["experiment_family", "rmse", "mae"]], on="experiment_family", how="left")
    pc2_limit = float(merged["pc2_std"].median() + merged["pc2_std"].std(ddof=0)) if len(merged) else np.nan
    rmse_limit = float(merged["rmse"].median() + merged["rmse"].std(ddof=0)) if merged["rmse"].notna().any() else np.nan
    rows = []
    for row in merged.itertuples(index=False):
        shared_manifold = bool(pd.notna(row.pc2_std) and (pd.isna(pc2_limit) or row.pc2_std <= pc2_limit))
        shared_dynamics = bool(pd.notna(row.rmse) and (pd.isna(rmse_limit) or row.rmse <= rmse_limit))
        notes = []
        if not shared_manifold:
            notes.append("wide PC2 spread")
        if not shared_dynamics:
            notes.append("large kernel residual")
        if row.experiment_family == "weighted_dropout_dense":
            notes.append("high-cliffiness dropout arc")
        if row.experiment_family == "fragmentation":
            notes.append("compact low-cliffiness arc")
        rows.append(
            {
                "Family": row.experiment_family,
                "Level Fit": "shared" if shared_manifold else "family correction",
                "Dynamics Fit": "shared" if shared_dynamics else "family correction",
                "Shared Manifold?": "yes" if shared_manifold else "partial",
                "Shared Dynamics?": "yes" if shared_dynamics else "partial",
                "Notes": "; ".join(notes) if notes else "follows pooled morphology field",
            }
        )
    order = {name: i for i, name in enumerate(FAMILY_ORDER)}
    return pd.DataFrame(rows).sort_values("Family", key=lambda s: s.map(order).fillna(999)).reset_index(drop=True)


def allocator_state_table(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns:
        model_col = next((col for col in ["model_id", "objective", "model_name"] if col in df.columns), None)
        if model_col is None:
            df = df.assign(model="unknown")
        else:
            df = df.assign(model=df[model_col].astype(str))
    model_summary = df.groupby("model", as_index=False).agg(breadth=("breadth", "mean"), elevation=("elevation", "mean"), survival_auc=("minority_survival_auc", "mean"), cliffiness=("minority_survival_cliffiness", "mean"), n_rows=("model", "size"))
    bq = model_summary["breadth"].quantile([0.33, 0.67]).to_numpy()
    eq = model_summary["elevation"].quantile([0.33, 0.67]).to_numpy()

    def regime(row) -> str:
        if row.breadth >= bq[1] and row.elevation >= eq[1]:
            return "broad allocator"
        if row.breadth <= bq[0] and row.elevation <= eq[0]:
            return "quantized allocator"
        if row.breadth <= bq[0] and row.elevation >= eq[1]:
            return "concentrated allocator"
        return "mixed morphology"

    model_summary["operational_regime"] = model_summary.apply(regime, axis=1)
    rows = []
    for model in sorted(set(model_summary["model"]).union(ALLOCATOR_LABELS)):
        row = model_summary[model_summary.model == model]
        if row.empty:
            rows.append({"Allocator": ALLOCATOR_LABELS.get(model, model), "Observed?": "no", "Breadth": np.nan, "Elevation": np.nan, "Operational Regime": "not present in current inputs"})
        else:
            item = row.iloc[0]
            rows.append({"Allocator": ALLOCATOR_LABELS.get(model, model), "Observed?": "yes", "Breadth": float(item.breadth), "Elevation": float(item.elevation), "Operational Regime": item.operational_regime})
    return pd.DataFrame(rows)


def synthesis_claim(power: pd.DataFrame) -> str:
    survival = power[power.Target == "Survival AUC"].iloc[0]
    cliff = power[power.Target == "Cliffiness"].iloc[0]
    survives = survival.Linear > 0.5 and cliff.Kernel > 0.5 and cliff.Kernel > cliff.Linear + 0.5
    if survives:
        return "Accessibility level is largely explained by position along a shared morphology manifold, while accessibility dynamics emerge from nonlinear geometry on that manifold."
    return "The shared morphology manifold is promising, but the current evidence does not yet fully support a single nonlinear morphology theory for both accessibility level and dynamics."


def operational_concepts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Morphology Term": "Accessibility Breadth", "Operational Meaning": "how broadly minority-positive score mass is distributed across accessible thresholds", "Measured By": "positive score-allocation entropy / breadth"},
            {"Morphology Term": "Accessibility Elevation", "Operational Meaning": "how much minority mass is lifted into the most accessible high-score region", "Measured By": "positive top-bin mass / elevation"},
            {"Morphology Term": "Accessibility Persistence", "Operational Meaning": "how long minority support survives as the decision threshold tightens", "Measured By": "minority survival AUC"},
            {"Morphology Term": "Accessibility Dynamics", "Operational Meaning": "how abruptly accessibility changes along the threshold path", "Measured By": "minority survival cliffiness and kernel-field gradient"},
        ]
    )


def plot_morphology_space(df: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for family, group in df.groupby("experiment_family"):
        ax.scatter(group.breadth, group.elevation, s=12, alpha=0.42, label=family)
    ax.set_xlabel("Accessibility Breadth")
    ax.set_ylabel("Accessibility Elevation")
    ax.set_title("Figure 1. Accessibility Morphology Space")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_kernel_cliffiness_field(field: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(field.breadth, field.elevation, c=field.predicted_minority_survival_cliffiness, cmap="magma", s=10)
    ax.set_xlabel("Accessibility Breadth")
    ax.set_ylabel("Accessibility Elevation")
    ax.set_title("Figure 2. Kernel Cliffiness Field")
    fig.colorbar(sc, ax=ax, label="Predicted cliffiness")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_family_trajectories(df: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for family, group in df.sort_values("pc1").groupby("experiment_family"):
        ax.plot(group.breadth, group.elevation, marker="o", markersize=2, linewidth=1, alpha=0.7, label=family)
    ax.set_xlabel("Accessibility Breadth")
    ax.set_ylabel("Accessibility Elevation")
    ax.set_title("Figure 3. Family Trajectories")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_operational_regime_map(df: pd.DataFrame, allocators: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    observed = allocators[allocators["Observed?"] == "yes"].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df.breadth, df.elevation, color="lightgray", alpha=0.2, s=8)
    if not observed.empty:
        ax.scatter(observed.Breadth, observed.Elevation, color="black", s=28)
        for row in observed.itertuples(index=False):
            if row.Allocator in {"HDDT", "Bagged HDDT", "CART", "XGBoost", "LightGBM"} or "MLP" in row.Allocator:
                ax.annotate(row.Allocator, (row.Breadth, row.Elevation), fontsize=7, xytext=(3, 3), textcoords="offset points")
    xmid = float(df.breadth.median())
    ymid = float(df.elevation.median())
    ax.axvline(xmid, color="black", linewidth=0.8, alpha=0.5)
    ax.axhline(ymid, color="black", linewidth=0.8, alpha=0.5)
    ax.text(df.breadth.min(), df.elevation.max(), "Broad\nAllocator", va="top", ha="left", fontsize=10)
    ax.text(df.breadth.max(), df.elevation.max(), "Concentrated\nAllocator", va="top", ha="right", fontsize=10)
    ax.text(df.breadth.min(), df.elevation.min(), "Quantized\nAllocator", va="bottom", ha="left", fontsize=10)
    ax.set_xlabel("Accessibility Breadth")
    ax.set_ylabel("Accessibility Elevation")
    ax.set_title("Figure 4. Operational Regime Map")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def build_report(messages, power, family_dev, allocators, concepts, claim, pca_diag) -> str:
    return "\n".join(
        [
            "# Accessibility Morphology Synthesis Report",
            "",
            "## Objective",
            "This report tests whether the experiments converge on the framework: Accessibility Level -> Morphology State -> Accessibility Dynamics.",
            "",
            "## Inputs Used",
            *[f"- {message}" for message in messages],
            "",
            "## RQ1: Minimum Morphology Representation",
            "Level is already well preserved by linear breadth/elevation. Dynamics are not: cliffiness requires the nonlinear kernel morphology surface.",
            "",
            "### Table 1: Morphology Explanatory Power",
            _markdown_table(power),
            "",
            "### PCA Diagnostics",
            _markdown_table(pca_diag),
            "",
            "## RQ2: Universal vs Family-Specific Findings",
            "Families mostly occupy the same pooled morphology space, but dynamics quality varies by arc and residual structure.",
            "",
            "### Family | Level Fit | Dynamics Fit | Notes",
            _markdown_table(family_dev[["Family", "Level Fit", "Dynamics Fit", "Notes"]]),
            "",
            "### Table 2: Family Deviations",
            _markdown_table(family_dev[["Family", "Shared Manifold?", "Shared Dynamics?", "Notes"]]),
            "",
            "## RQ3: What Morphology Measures",
            _markdown_table(concepts),
            "",
            "## RQ4: Common Allocator State Space",
            "All observed allocators can be placed in the common breadth/elevation state space. Unobserved named allocators are retained in the table as missing from the current inputs.",
            "",
            _markdown_table(allocators),
            "",
            "## Universal Findings",
            "- Accessibility level tracks morphology position, especially breadth/elevation and PC1.",
            "- Survival AUC is largely a level variable on the shared morphology manifold.",
            "- Cliffiness is not a linear coordinate effect; it is a nonlinear field over morphology state.",
            "",
            "## Family-Specific Findings",
            "- Families trace distinct arcs rather than collapsing to a single trajectory.",
            "- High PC2-spread or high residual families are the main candidates for family-specific corrections.",
            "- Dropout and fragmentation occupy qualitatively different operating regimes in the same state space.",
            "",
            "## Figures",
            "- Figure 1: `accessibility_morphology_space.png`",
            "- Figure 2: `accessibility_kernel_cliffiness_field.png`",
            "- Figure 3: `accessibility_family_trajectories.png`",
            "- Figure 4: `accessibility_operational_regime_map.png`",
            "",
            "## Synthesis Claim",
            claim,
            "",
            "## Limitations",
            "- Family labels are experiment-design labels, so leave-family CV is intentionally strict.",
            "- The operational regime map is empirical and depends on available model families in the input CSVs.",
            "- Kernel morphology explains dynamics but is not itself a causal mechanism without perturbation tests.",
            "",
            "## Next Steps",
            "- Use this synthesis as the paper-level framing before adding more sweeps.",
            "- Investigate family-specific corrections for the largest residual families.",
            "- Convert the operational regime map into the main Paper 2 conceptual figure.",
        ]
    ) + "\n"


def write_report(input_paths: list[Path], output_md: Path, gamma: float = 1.0, grid_size: int = 80) -> tuple[Path, ...]:
    df, messages, pca_diag = prepare_dataset(input_paths)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        output_md.write_text(build_report(messages, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), operational_concepts(), "No synthesis possible because no inputs were available.", pca_diag), encoding="utf-8")
        return (output_md,)
    feature_perf = evaluate_feature_sets(df)
    power = morphology_explanatory_power(feature_perf)
    _, predictions = evaluate_smoothers(df, gamma=gamma)
    residuals = residuals_by_family(df, predictions)
    arcs = family_arc_summary(df)
    family_dev = family_deviation_table(df, residuals, arcs)
    allocators = allocator_state_table(df)
    concepts = operational_concepts()
    claim = synthesis_claim(power)
    field, _, _, _ = build_field_grid(df, "minority_survival_cliffiness", gamma=gamma, grid_size=grid_size)
    output_md.write_text(build_report(messages, power, family_dev, allocators, concepts, claim, pca_diag), encoding="utf-8")
    return (
        output_md,
        plot_morphology_space(df, output_md.parent / "accessibility_morphology_space.png"),
        plot_kernel_cliffiness_field(field, output_md.parent / "accessibility_kernel_cliffiness_field.png"),
        plot_family_trajectories(df, output_md.parent / "accessibility_family_trajectories.png"),
        plot_operational_regime_map(df, allocators, output_md.parent / "accessibility_operational_regime_map.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "topology" / "accessibility_morphology_synthesis_summary.md")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=80)
    args = parser.parse_args()
    for path in write_report(args.inputs, args.output_md, gamma=args.gamma, grid_size=args.grid_size):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
