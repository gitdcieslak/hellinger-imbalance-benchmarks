"""Report classical allocator placement in accessibility morphology space."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_morphology_ridge_extraction import DEFAULT_INPUTS, load_inputs  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED_MODELS = ["cart", "hddt", "bagged_hddt", "random_forest", "xgboost", "lightgbm"]
MEAN_COLUMNS = ["auroc", "average_precision", "minority_survival_auc", "minority_survival_cliffiness", "breadth", "elevation", "positive_quantization_score"]


def parse_model_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def load_classical_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "fit_failed" in df.columns:
        df["fit_failed"] = df["fit_failed"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["fit_failed"] = False
    for col in ["breadth", "elevation", "positive_quantization_score", "positive_unique_score_ratio", *MEAN_COLUMNS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def classify_regime(row: pd.Series) -> str:
    quant = float(row.get("positive_quantization_score", np.nan))
    unique = float(row.get("positive_unique_score_ratio", np.nan))
    breadth = float(row.get("breadth", np.nan))
    elevation = float(row.get("elevation", np.nan))
    if (pd.notna(quant) and quant >= 0.90) or (pd.notna(unique) and unique <= 0.10):
        return "quantized allocator"
    if pd.notna(elevation) and pd.notna(breadth) and elevation >= 0.80 and breadth <= 0.60:
        return "concentrated allocator"
    if pd.notna(elevation) and pd.notna(breadth) and breadth >= 1.00 and elevation < 0.80:
        return "broad allocator"
    return "mixed morphology"


def add_regimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["operational_regime"] = out.apply(classify_regime, axis=1)
    return out


def availability_table(df: pd.DataFrame, expected_models: list[str]) -> pd.DataFrame:
    rows = []
    present = set(df["model_id"].astype(str)) if "model_id" in df.columns and not df.empty else set()
    for model_id in expected_models:
        model_rows = df[df["model_id"] == model_id] if model_id in present else pd.DataFrame()
        if model_rows.empty:
            status = "missing"
            failed = 0
            reason = "not present in input CSV"
        elif bool(model_rows["fit_failed"].all()):
            status = "failed"
            failed = int(model_rows["fit_failed"].sum())
            reason = str(model_rows["failure_reason"].dropna().iloc[0]) if model_rows["failure_reason"].notna().any() else "all runs failed"
        else:
            status = "available"
            failed = int(model_rows["fit_failed"].sum())
            reason = ""
        rows.append({"model_id": model_id, "status": status, "rows": len(model_rows), "failed_rows": failed, "note": reason})
    return pd.DataFrame(rows)


def classical_means(df: pd.DataFrame) -> pd.DataFrame:
    data = df[~df["fit_failed"]].copy()
    if data.empty:
        return pd.DataFrame()
    cols = [col for col in MEAN_COLUMNS if col in data.columns]
    return data.groupby(["model_id", "skew_ratio"], as_index=False)[cols].mean()


def model_regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    data = df[~df["fit_failed"]].copy()
    if data.empty:
        return pd.DataFrame()
    rows = []
    for model_id, group in data.groupby("model_id"):
        counts = group["operational_regime"].value_counts()
        rows.append({"model_id": model_id, "majority_regime": counts.index[0], "majority_fraction": float(counts.iloc[0] / counts.sum()), "n_rows": int(counts.sum())})
    return pd.DataFrame(rows)


def load_pooled_manifold(paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    pooled, messages = load_inputs(paths)
    return pooled, messages


def manifold_support_diagnostics(classical: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    data = classical[~classical["fit_failed"]].dropna(subset=["breadth", "elevation"]).copy()
    if data.empty:
        return pd.DataFrame()
    rows = []
    if pooled.empty:
        for row in data.itertuples(index=False):
            rows.append({"model_id": row.model_id, "seed": row.seed, "skew_ratio": row.skew_ratio, "nearest_neighbor_distance_to_pooled_manifold": np.nan, "within_pooled_breadth_range": False, "within_pooled_elevation_range": False, "within_convex_hull_if_scipy_available": np.nan, "support_status": "no pooled manifold"})
        return pd.DataFrame(rows)

    pool = pooled.dropna(subset=["breadth", "elevation"]).copy()
    coords = pool[["breadth", "elevation"]].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=1).fit(coords)
    distances, _ = nn.kneighbors(data[["breadth", "elevation"]].to_numpy(dtype=float))
    bmin, bmax = float(pool["breadth"].min()), float(pool["breadth"].max())
    emin, emax = float(pool["elevation"].min()), float(pool["elevation"].max())
    threshold = float(np.quantile(distances.ravel(), 0.90) + np.std(distances.ravel())) if len(distances) else 0.0
    hull_flags = _convex_hull_flags(coords, data[["breadth", "elevation"]].to_numpy(dtype=float))
    for idx, row in enumerate(data.itertuples(index=False)):
        in_range = bmin <= float(row.breadth) <= bmax and emin <= float(row.elevation) <= emax
        dist = float(distances[idx, 0])
        in_hull = hull_flags[idx]
        if in_range and (pd.isna(in_hull) or bool(in_hull)):
            status = "inside shared morphology manifold"
        elif in_range or dist <= threshold:
            status = "near boundary"
        else:
            status = "outside observed support"
        rows.append({"model_id": row.model_id, "seed": row.seed, "skew_ratio": row.skew_ratio, "nearest_neighbor_distance_to_pooled_manifold": dist, "within_pooled_breadth_range": bmin <= float(row.breadth) <= bmax, "within_pooled_elevation_range": emin <= float(row.elevation) <= emax, "within_convex_hull_if_scipy_available": in_hull, "support_status": status})
    return pd.DataFrame(rows)


def _convex_hull_flags(pool_coords: np.ndarray, test_coords: np.ndarray) -> list[bool | float]:
    try:
        from scipy.spatial import Delaunay
    except Exception:
        return [np.nan] * len(test_coords)
    if len(pool_coords) < 3:
        return [np.nan] * len(test_coords)
    try:
        hull = Delaunay(pool_coords)
        return [bool(value >= 0) for value in hull.find_simplex(test_coords)]
    except Exception:
        return [np.nan] * len(test_coords)


def support_by_model(support: pd.DataFrame) -> pd.DataFrame:
    if support.empty:
        return pd.DataFrame()
    rows = []
    for model_id, group in support.groupby("model_id"):
        status = group["support_status"].value_counts().index[0]
        rows.append({"model_id": model_id, "majority_support_status": status, "mean_nn_distance": float(group["nearest_neighbor_distance_to_pooled_manifold"].mean()), "n_rows": len(group)})
    return pd.DataFrame(rows)


def neural_comparison(classical: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    data = classical[~classical["fit_failed"]].copy()
    rows = []
    for model_id, group in data.groupby("model_id"):
        rows.append({"source": "classical", "model_id": model_id, "breadth": float(group["breadth"].mean()), "elevation": float(group["elevation"].mean()), "minority_survival_auc": float(group["minority_survival_auc"].mean()), "minority_survival_cliffiness": float(group["minority_survival_cliffiness"].mean())})
    if not pooled.empty:
        labels = ["mlp_bce", "mlp_weighted_bce", "mlp_oversampled_bce"]
        for model_id, group in pooled[pooled["model"].isin(labels)].groupby("model"):
            rows.append({"source": "prior_neural", "model_id": model_id, "breadth": float(group["breadth"].mean()), "elevation": float(group["elevation"].mean()), "minority_survival_auc": float(group["minority_survival_auc"].mean()), "minority_survival_cliffiness": float(group["minority_survival_cliffiness"].mean())})
        wd = pooled[pooled["experiment_family"] == "weighted_dropout_dense"]
        if not wd.empty:
            rows.append({"source": "prior_family", "model_id": "weighted_dropout_dense", "breadth": float(wd["breadth"].mean()), "elevation": float(wd["elevation"].mean()), "minority_survival_auc": float(wd["minority_survival_auc"].mean()), "minority_survival_cliffiness": float(wd["minority_survival_cliffiness"].mean())})
    return pd.DataFrame(rows)


def implications(availability: pd.DataFrame, regimes: pd.DataFrame, support: pd.DataFrame) -> list[str]:
    support_model = support_by_model(support)
    inside_count = int((support_model["majority_support_status"] == "inside shared morphology manifold").sum()) if not support_model.empty else 0
    total = len(support_model)
    def regime_for(model_id: str) -> str:
        row = regimes[regimes["model_id"] == model_id]
        return row.iloc[0].majority_regime if not row.empty else "missing"
    available = availability[availability["status"] == "available"]["model_id"].tolist()
    return [
        f"1. Do classical allocators occupy the same morphology space? {'Yes' if total and inside_count >= max(1, total // 2) else 'Mixed or not yet'}; {inside_count}/{total} available models are primarily inside the pooled support.",
        f"2. Which classical allocators are broad, concentrated, quantized, or mixed? See Operational Regime Classification; available model regimes: {', '.join(f'{m}={regime_for(m)}' for m in available)}.",
        f"3. Does CART exhibit quantized morphology? {'Yes' if regime_for('cart') == 'quantized allocator' else 'No or not observed'}; CART majority regime={regime_for('cart')}.",
        f"4. Do boosted models exhibit concentrated or cliff-like morphology? XGBoost={regime_for('xgboost')}; LightGBM={regime_for('lightgbm')}. Interpret as placement only, not causal evidence.",
        f"5. Does Bagged HDDT occupy a distinct morphology regime? Bagged HDDT majority regime={regime_for('bagged_hddt')}; compare its support status and regime against CART/HDDT.",
        "6. Does adding classical allocators strengthen the Paper 2 claim? It strengthens the claim if most available classical rows fall inside or near the shared manifold; missing optional models should be filled before manuscript overclaiming.",
    ]


def plot_space(classical: pd.DataFrame, pooled: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    if not pooled.empty:
        ax.scatter(pooled["breadth"], pooled["elevation"], color="lightgray", alpha=0.25, s=8, label="prior pooled manifold")
    data = classical[~classical["fit_failed"]]
    for model_id, group in data.groupby("model_id"):
        ax.scatter(group["breadth"], group["elevation"], s=30, alpha=0.75, label=model_id)
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Classical Allocator Morphology Space")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_metric_by_model(classical: pd.DataFrame, metric: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = classical[~classical["fit_failed"]].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not data.empty:
        means = data.groupby("model_id")[metric].mean().sort_values()
        ax.bar(means.index, means.values)
        ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel(metric)
    ax.set_title(metric.replace("_", " ").title() + " By Model")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_regime_map(classical: pd.DataFrame, pooled: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    if not pooled.empty:
        ax.scatter(pooled["breadth"], pooled["elevation"], color="lightgray", alpha=0.18, s=8)
    colors = {"broad allocator": "tab:blue", "concentrated allocator": "tab:orange", "quantized allocator": "tab:red", "mixed morphology": "tab:green"}
    data = classical[~classical["fit_failed"]]
    for regime, group in data.groupby("operational_regime"):
        ax.scatter(group["breadth"], group["elevation"], s=35, alpha=0.8, label=regime, color=colors.get(regime, "black"))
    ax.axvline(1.00, color="black", linewidth=0.8, alpha=0.4)
    ax.axhline(0.80, color="black", linewidth=0.8, alpha=0.4)
    ax.axvline(0.60, color="black", linewidth=0.8, alpha=0.25, linestyle="--")
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Classical Allocator Regime Map")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def build_report(messages, availability, means, regimes, support_model, comparison, implications_text) -> str:
    return "\n".join([
        "# Classical Allocator Morphology",
        "",
        "## 1. Inputs and Availability",
        *[f"- {message}" for message in messages],
        "",
        _markdown_table(availability),
        "",
        "## 2. Classical Allocator Means",
        _markdown_table(means),
        "",
        "## 3. Morphology Placement",
        "Classical allocator points are plotted over the pooled prior morphology manifold when prior files are available; otherwise plots show classical points alone.",
        "",
        "## 4. Operational Regime Classification",
        _markdown_table(regimes),
        "",
        "## 5. Manifold Consistency",
        _markdown_table(support_model),
        "",
        "## 6. Comparison to Neural/Objectives",
        _markdown_table(comparison),
        "",
        "## 7. Paper 2 Implications",
        *[f"- {line}" for line in implications_text],
    ]) + "\n"


def write_report(input_csv: Path, output_md: Path, expected_models: list[str] | None = None, prior_inputs: list[Path] | None = None) -> tuple[Path, ...]:
    expected = expected_models or DEFAULT_EXPECTED_MODELS
    prior_paths = prior_inputs if prior_inputs is not None else DEFAULT_INPUTS
    df = add_regimes(load_classical_results(input_csv))
    pooled, messages = load_pooled_manifold(prior_paths)
    messages = [f"classical input: {input_csv} rows={len(df)}", *messages]
    availability = availability_table(df, expected)
    means = classical_means(df)
    regimes = model_regime_summary(df)
    support = manifold_support_diagnostics(df, pooled)
    support_model = support_by_model(support)
    comparison = neural_comparison(df, pooled)
    implications_text = implications(availability, regimes, support)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_report(messages, availability, means, regimes, support_model, comparison, implications_text), encoding="utf-8")
    return (
        output_md,
        plot_space(df, pooled, output_md.parent / "classical_allocator_morphology_space.png"),
        plot_metric_by_model(df, "minority_survival_auc", output_md.parent / "classical_allocator_survival_by_model.png"),
        plot_metric_by_model(df, "minority_survival_cliffiness", output_md.parent / "classical_allocator_cliffiness_by_model.png"),
        plot_regime_map(df, pooled, output_md.parent / "classical_allocator_regime_map.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "topology" / "classical_allocator_morphology.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "topology" / "classical_allocator_morphology_summary.md")
    parser.add_argument("--expected-models", type=parse_model_list, default=DEFAULT_EXPECTED_MODELS)
    parser.add_argument("--prior-inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    args = parser.parse_args()
    for path in write_report(args.input, args.output_md, expected_models=args.expected_models, prior_inputs=args.prior_inputs):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
