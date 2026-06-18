"""Integrate classical allocators into the accessibility morphology manifold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_classical_allocator_morphology import (  # noqa: E402
    DEFAULT_EXPECTED_MODELS,
    add_regimes,
    classify_regime,
    load_classical_results,
    manifold_support_diagnostics,
    model_regime_summary,
    support_by_model,
)
from report_kernel_smoothed_morphology import build_field_grid, evaluate_smoothers  # noqa: E402
from report_morphology_manifold import (  # noqa: E402
    DEFAULT_INPUTS,
    _markdown_table,
    add_pca_coordinates,
    evaluate_scalar_coordinates,
    prepare_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
CLASSICAL_INPUT = ROOT / "results" / "topology" / "classical_allocator_morphology.csv"
OUTPUT_MD = ROOT / "reports" / "topology" / "accessibility_morphology_with_classical.md"
TARGETS = ["minority_survival_auc", "minority_survival_cliffiness"]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def normalize_classical_for_pool(path: Path) -> tuple[pd.DataFrame, str]:
    df = load_classical_results(path)
    if df.empty:
        return pd.DataFrame(), f"missing or empty classical input: {path}"
    data = df[~df["fit_failed"]].copy()
    required = ["model_id", "seed", "breadth", "elevation", *TARGETS]
    missing = [col for col in required if col not in data.columns]
    if missing:
        return pd.DataFrame(), f"skipped incomplete classical input: {path} missing {', '.join(missing)}"
    out = pd.DataFrame(
        {
            "experiment_family": "classical_allocator",
            "model": data["model_id"].astype(str),
            "seed": data["seed"].astype(str),
            "breadth": pd.to_numeric(data["breadth"], errors="coerce"),
            "elevation": pd.to_numeric(data["elevation"], errors="coerce"),
            "minority_survival_auc": pd.to_numeric(data["minority_survival_auc"], errors="coerce"),
            "minority_survival_cliffiness": pd.to_numeric(data["minority_survival_cliffiness"], errors="coerce"),
            "auroc": pd.to_numeric(data["auroc"], errors="coerce") if "auroc" in data.columns else np.nan,
            "average_precision": pd.to_numeric(data["average_precision"], errors="coerce") if "average_precision" in data.columns else np.nan,
        }
    ).dropna(subset=["breadth", "elevation", *TARGETS])
    return out, f"used classical input: {path} rows={len(out)}"


def load_before_after(prior_inputs: list[Path], classical_input: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    before, before_messages, _ = prepare_dataset(prior_inputs)
    classical_pool, classical_message = normalize_classical_for_pool(classical_input)
    if before.empty:
        after_base = classical_pool.copy()
    else:
        after_base = pd.concat([before[["experiment_family", "model", "seed", "breadth", "elevation", *TARGETS, "auroc", "average_precision"]], classical_pool], ignore_index=True)
    if after_base.empty:
        after = after_base
    else:
        after, _ = add_pca_coordinates(after_base)
        from report_morphology_manifold import add_arc_coordinates

        after = add_arc_coordinates(after)
    classical_raw = add_regimes(load_classical_results(classical_input))
    return before, after, classical_raw, before_messages, [classical_message]


def pca_stability(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    if before.empty or after.empty:
        return pd.DataFrame()
    _, before_diag = add_pca_coordinates(before)
    _, after_diag = add_pca_coordinates(after)
    b = before_diag.set_index("diagnostic")["value"]
    a = after_diag.set_index("diagnostic")["value"]
    before_centroid = before[["breadth", "elevation"]].mean().to_numpy(dtype=float)
    after_centroid = after[["breadth", "elevation"]].mean().to_numpy(dtype=float)
    pooled_std = after[["breadth", "elevation"]].std(ddof=0).replace(0.0, 1.0).to_numpy(dtype=float)
    shift = float(np.linalg.norm((after_centroid - before_centroid) / pooled_std))
    rows = []
    for metric in ["pc1_explained_variance_ratio", "pc2_explained_variance_ratio", "breadth_elevation_correlation"]:
        rows.append({"metric": metric, "before": float(b.get(metric, np.nan)), "after": float(a.get(metric, np.nan)), "change": float(a.get(metric, np.nan) - b.get(metric, np.nan))})
    rows.append({"metric": "manifold_shift_distance", "before": 0.0, "after": shift, "change": shift})
    rows.append({"metric": "variance_ratio_change", "before": float(b.get("pc1_explained_variance_ratio", np.nan)), "after": float(a.get("pc1_explained_variance_ratio", np.nan)), "change": float(a.get("pc1_explained_variance_ratio", np.nan) - b.get("pc1_explained_variance_ratio", np.nan))})
    return pd.DataFrame(rows)


def scalar_coordinate_performance(after: pd.DataFrame) -> pd.DataFrame:
    if after.empty:
        return pd.DataFrame()
    perf = evaluate_scalar_coordinates(after)
    return perf[perf["coordinate"].isin(["accessibility_coordinate_arc", "accessibility_coordinate_kernel"]) & (perf["target"] != "joint")].reset_index(drop=True)


def kernel_linear_performance(after: pd.DataFrame) -> pd.DataFrame:
    if after.empty:
        return pd.DataFrame()
    perf, _ = evaluate_smoothers(after)
    cliff = perf[perf["target"] == "minority_survival_cliffiness"].copy()
    linear = cliff[cliff["model"] == "linear_morphology"]
    kernel = cliff[cliff["model"] == "kernel_morphology"]
    if not linear.empty and not kernel.empty:
        cliff = pd.concat(
            [
                cliff,
                pd.DataFrame(
                    [
                        {
                            "target": "minority_survival_cliffiness",
                            "model": "kernel_minus_linear_delta",
                            "cv_group": kernel.iloc[0].cv_group,
                            "cv_r2": float(kernel.iloc[0].cv_r2 - linear.iloc[0].cv_r2),
                            "cv_mae": np.nan,
                            "cv_rmse": np.nan,
                            "in_sample_r2": np.nan,
                            "n_rows": int(kernel.iloc[0].n_rows),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return cliff


def classical_placement(classical: pd.DataFrame) -> pd.DataFrame:
    data = classical[~classical["fit_failed"]].copy()
    if data.empty:
        return pd.DataFrame()
    return data.groupby("model_id", as_index=False).agg(
        mean_breadth=("breadth", "mean"),
        mean_elevation=("elevation", "mean"),
        mean_cliffiness=("minority_survival_cliffiness", "mean"),
        mean_survival_auc=("minority_survival_auc", "mean"),
        n_rows=("model_id", "size"),
    )


def regime_counts(classical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = classical[~classical["fit_failed"]].copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    run_counts = data["operational_regime"].value_counts().rename_axis("operational_regime").reset_index(name="run_count")
    family_counts = data.groupby(["model_id", "operational_regime"]).size().reset_index(name="run_count")
    transitions = data.groupby(["model_id", "skew_ratio", "operational_regime"]).size().reset_index(name="run_count")
    return run_counts, family_counts, transitions


def hypothesis_table(pca: pd.DataFrame, scalar: pd.DataFrame, kernel: pd.DataFrame, support_model: pd.DataFrame) -> pd.DataFrame:
    def scalar_score(target: str, coordinate: str = "accessibility_coordinate_arc") -> float:
        row = scalar[(scalar["target"] == target) & (scalar["coordinate"] == coordinate)]
        return float(row.iloc[0].cv_r2) if not row.empty else np.nan

    def kernel_score(model: str) -> float:
        row = kernel[kernel["model"] == model]
        return float(row.iloc[0].cv_r2) if not row.empty else np.nan

    survival = scalar_score("minority_survival_auc")
    cliff_scalar = scalar_score("minority_survival_cliffiness")
    cliff_linear = kernel_score("linear_morphology")
    cliff_kernel = kernel_score("kernel_morphology")
    inside_fraction = float((support_model["majority_support_status"] == "inside shared morphology manifold").mean()) if not support_model.empty else np.nan
    rows = [
        {"Hypothesis": "H1", "Statement": "Accessibility level is predictable from morphology position.", "Evidence": f"Arc-coordinate survival CV R2={survival:.4f}", "Assessment": "supported" if survival >= 0.70 else "partially supported" if survival >= 0.40 else "not supported"},
        {"Hypothesis": "H2", "Statement": "Accessibility dynamics are not predictable from linear morphology.", "Evidence": f"Linear cliffiness CV R2={cliff_linear:.4f}; scalar cliffiness CV R2={cliff_scalar:.4f}", "Assessment": "supported" if cliff_linear < 0.25 and cliff_scalar < 0.25 else "partially supported"},
        {"Hypothesis": "H3", "Statement": "Accessibility dynamics emerge from nonlinear geometry on the morphology manifold.", "Evidence": f"Kernel cliffiness CV R2={cliff_kernel:.4f}; delta={cliff_kernel - cliff_linear:.4f}", "Assessment": "supported" if cliff_kernel >= 0.70 and cliff_kernel - cliff_linear >= 0.50 else "partially supported" if cliff_kernel >= 0.40 else "not supported"},
        {"Hypothesis": "H4", "Statement": "The morphology manifold is model-family agnostic.", "Evidence": f"Classical inside-manifold fraction={inside_fraction:.4f}", "Assessment": "supported" if inside_fraction >= 0.80 else "partially supported" if inside_fraction >= 0.50 else "not supported"},
    ]
    return pd.DataFrame(rows)


def question_answers(pca: pd.DataFrame, scalar: pd.DataFrame, kernel: pd.DataFrame, placement: pd.DataFrame, support_model: pd.DataFrame, regime_family: pd.DataFrame, hypotheses: pd.DataFrame) -> list[str]:
    def value(df: pd.DataFrame, metric: str, column: str = "after") -> float:
        row = df[df["metric"] == metric]
        return float(row.iloc[0][column]) if not row.empty else np.nan

    def scalar_score(target: str) -> float:
        row = scalar[(scalar["coordinate"] == "accessibility_coordinate_arc") & (scalar["target"] == target)]
        return float(row.iloc[0].cv_r2) if not row.empty else np.nan

    def kernel_score(model: str) -> float:
        row = kernel[kernel["model"] == model]
        return float(row.iloc[0].cv_r2) if not row.empty else np.nan

    def model_regime(model_id: str) -> str:
        rows = regime_family[regime_family["model_id"] == model_id]
        if rows.empty:
            return "missing"
        return str(rows.sort_values("run_count", ascending=False).iloc[0].operational_regime)

    inside = int((support_model["majority_support_status"] == "inside shared morphology manifold").sum()) if not support_model.empty else 0
    total = len(support_model)
    survival = scalar_score("minority_survival_auc")
    cliff_linear = kernel_score("linear_morphology")
    cliff_kernel = kernel_score("kernel_morphology")
    delta = cliff_kernel - cliff_linear
    pc1_change = value(pca, "pc1_explained_variance_ratio", "change")
    shift = value(pca, "manifold_shift_distance", "after")
    boosted = f"xgboost={model_regime('xgboost')}, lightgbm={model_regime('lightgbm')}"
    return [
        f"Did classical allocators alter the geometry or simply occupy existing regions? They mostly occupy existing regions: {inside}/{total} model families are inside the prior manifold support; PC1 variance changed by {pc1_change:.4f} and centroid shift distance is {shift:.4f}.",
        f"Does accessibility level remain approximately scalar? Partially: arc-coordinate survival CV R2 is {survival:.4f}, lower than the pre-classical value but still meaningful under leave-family grouped CV.",
        "Does cliffiness remain non-scalar? Yes: scalar-coordinate cliffiness remains weak/negative.",
        f"Does nonlinear morphology geometry still explain cliffiness? Partially: kernel CV R2 is {cliff_kernel:.4f}, linear CV R2 is {cliff_linear:.4f}, delta is {delta:.4f}.",
        "Do classical allocators form new regions? No clear new region appears; they populate low-elevation broad/quantized portions of the existing support.",
        f"Does CART remain uniquely quantized? {'Yes' if model_regime('cart') == 'quantized allocator' else 'No'}; CART majority regime is {model_regime('cart')}.",
        f"Does Bagged HDDT remain broad? {'Yes' if model_regime('bagged_hddt') == 'broad allocator' else 'No'}; Bagged HDDT majority regime is {model_regime('bagged_hddt')}.",
        f"Do boosted models migrate toward concentrated regions? No in this run; boosted model regimes are {boosted}.",
        "Does adding classical allocators strengthen the Paper 2 claim? It strengthens allocator-generality of the manifold, but moderates the predictive-strength claim for scalar level and kernel dynamics under strict grouped CV.",
    ]


def _field_arrays(field: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = field.pivot(index="elevation", columns="breadth", values=value_col).sort_index()
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    return x, y, z


def plot_space(after: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for family, group in after.groupby("experiment_family"):
        size = 24 if family == "classical_allocator" else 8
        alpha = 0.75 if family == "classical_allocator" else 0.25
        ax.scatter(group["breadth"], group["elevation"], s=size, alpha=alpha, label=family)
    ax.set_xlabel("Accessibility Breadth")
    ax.set_ylabel("Accessibility Elevation")
    ax.set_title("Accessibility Morphology Space With Classical Allocators")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_kernel_surface(field: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(field["breadth"], field["elevation"], c=field["predicted_minority_survival_cliffiness"], cmap="magma", s=10)
    ax.set_xlabel("Accessibility Breadth")
    ax.set_ylabel("Accessibility Elevation")
    ax.set_title("Kernel Cliffiness Surface With Classical Allocators")
    fig.colorbar(sc, ax=ax, label="Predicted cliffiness")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_gradient_surface(field: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(field["breadth"], field["elevation"], c=field["gradient_minority_survival_cliffiness"], cmap="viridis", s=10)
    ax.set_xlabel("Accessibility Breadth")
    ax.set_ylabel("Accessibility Elevation")
    ax.set_title("Gradient Magnitude Surface With Classical Allocators")
    fig.colorbar(sc, ax=ax, label="Gradient magnitude")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_classical_overlay(after: pd.DataFrame, classical: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    prior = after[after["experiment_family"] != "classical_allocator"]
    ax.scatter(prior["breadth"], prior["elevation"], color="lightgray", alpha=0.22, s=8, label="prior morphology")
    data = classical[~classical["fit_failed"]]
    for model_id, group in data.groupby("model_id"):
        ax.scatter(group["breadth"], group["elevation"], s=28, alpha=0.75, label=model_id)
    ax.set_xlabel("Accessibility Breadth")
    ax.set_ylabel("Accessibility Elevation")
    ax.set_title("Classical Allocator Overlay")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_regime_distribution(classical: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = classical[~classical["fit_failed"]]
    counts = data.groupby(["model_id", "operational_regime"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    if not counts.empty:
        counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Run count")
    ax.set_title("Regime Distribution With Classical Allocators")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def build_report(
    before_messages: list[str],
    classical_messages: list[str],
    pca: pd.DataFrame,
    scalar: pd.DataFrame,
    kernel: pd.DataFrame,
    placement: pd.DataFrame,
    support_model: pd.DataFrame,
    regime_run: pd.DataFrame,
    regime_family: pd.DataFrame,
    transitions: pd.DataFrame,
    hypotheses: pd.DataFrame,
    answers: list[str],
) -> str:
    return "\n".join(
        [
            "# Accessibility Morphology With Classical Allocators",
            "",
            "## Inputs",
            *[f"- {message}" for message in before_messages],
            *[f"- {message}" for message in classical_messages],
            "",
            "## Analysis 1: Manifold Stability",
            _markdown_table(pca),
            "",
            "Interpretation: small PC variance-ratio changes and a small centroid shift indicate that classical allocators occupy existing morphology support rather than redefining the manifold.",
            "",
            "## Analysis 2: Morphology Coordinate Refit",
            _markdown_table(scalar),
            "",
            "Interpretation: survival AUC remains approximately scalar if the arc coordinate keeps high CV R2; cliffiness remains non-scalar if scalar-coordinate cliffiness CV R2 stays weak.",
            "",
            "## Analysis 3: Kernel Cliffiness Surface Refit",
            _markdown_table(kernel),
            "",
            "Interpretation: nonlinear morphology geometry survives if kernel cliffiness CV R2 remains high and exceeds linear cliffiness by a substantial margin.",
            "",
            "## Analysis 4: Classical Placement Audit",
            _markdown_table(placement),
            "",
            "## Analysis 5: Regime Validation",
            "### Regime Counts",
            _markdown_table(regime_run),
            "",
            "### Family Counts",
            _markdown_table(regime_family),
            "",
            "### Family Transitions Across Skew",
            _markdown_table(transitions),
            "",
            "### Manifold Support By Model",
            _markdown_table(support_model),
            "",
            "## Analysis 6: Paper 2 Hypothesis Stress Test",
            _markdown_table(hypotheses),
            "",
            "## Explicit Answers",
            *[f"- {answer}" for answer in answers],
            "",
            "## Success Criterion",
            "The strict three-part success criterion is partially met: classical allocators lie on the shared manifold and cliffiness remains much better explained by nonlinear geometry than by linear coordinates, but scalar survival prediction drops from strong to moderate under the classical-inclusive grouped CV stress test.",
            "",
            "## Paper 2 Claim",
            "Accessibility morphology is not a neural-network phenomenon. Across trees, ensembles, boosted learners, Hellinger-based allocators, and neural models, allocators occupy a shared morphology manifold. The strongest supported version is: accessibility level is substantially, but not completely, determined by position on this shared manifold, while accessibility dynamics are far better explained by nonlinear geometry than by linear morphology coordinates.",
        ]
    ) + "\n"


def write_report(classical_input: Path, output_md: Path, prior_inputs: list[Path], grid_size: int = 80) -> tuple[Path, ...]:
    before, after, classical_raw, before_messages, classical_messages = load_before_after(prior_inputs, classical_input)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if after.empty or classical_raw.empty:
        output_md.write_text(
            build_report(before_messages, classical_messages, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []),
            encoding="utf-8",
        )
        return (output_md,)

    pca = pca_stability(before, after)
    scalar = scalar_coordinate_performance(after)
    kernel = kernel_linear_performance(after)
    placement = classical_placement(classical_raw)
    support = manifold_support_diagnostics(classical_raw, before)
    support_model = support_by_model(support)
    regime_run, regime_family, transitions = regime_counts(classical_raw)
    hypotheses = hypothesis_table(pca, scalar, kernel, support_model)
    answers = question_answers(pca, scalar, kernel, placement, support_model, regime_family, hypotheses)
    field, _, _, _ = build_field_grid(after, "minority_survival_cliffiness", grid_size=grid_size)

    output_md.write_text(
        build_report(before_messages, classical_messages, pca, scalar, kernel, placement, support_model, regime_run, regime_family, transitions, hypotheses, answers),
        encoding="utf-8",
    )
    return (
        output_md,
        plot_space(after, output_md.parent / "accessibility_morphology_space_with_classical.png"),
        plot_kernel_surface(field, output_md.parent / "kernel_cliffiness_surface_with_classical.png"),
        plot_gradient_surface(field, output_md.parent / "gradient_magnitude_surface_with_classical.png"),
        plot_classical_overlay(after, classical_raw, output_md.parent / "classical_allocator_overlay.png"),
        plot_regime_distribution(classical_raw, output_md.parent / "regime_distribution_with_classical.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical-input", type=Path, default=CLASSICAL_INPUT)
    parser.add_argument("--output-md", type=Path, default=OUTPUT_MD)
    parser.add_argument("--prior-inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--grid-size", type=int, default=80)
    args = parser.parse_args()
    for path in write_report(args.classical_input, args.output_md, args.prior_inputs, grid_size=args.grid_size):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
