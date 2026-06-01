from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_tables = root / "paper" / "manuscript_tables"
    out_figs = root / "paper" / "manuscript_figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    bench = pd.read_csv(root / "reports" / "neural_mlp" / "legacy_benchmark_summary.csv")
    thr = pd.read_csv(root / "reports" / "neural_mlp" / "legacy_threshold_sweep_summary.csv")
    reg = pd.read_csv(root / "reports" / "neural_mlp" / "allocation_regime_summary.csv")
    occ = pd.read_csv(root / "reports" / "neural_mlp" / "prediction_space_occupancy_summary.csv")

    models = ["cart", "hddt", "hddt_forest", "random_forest", "xgboost", "lightgbm", "mlp"]
    names = {
        "cart": "CART",
        "hddt": "HDDT",
        "hddt_forest": "Bagged HDDT (hddt_forest)",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "mlp": "MLP",
    }

    bench_m = (
        bench[bench.model_id.isin(models)]
        .groupby("model_id", as_index=False)[["auroc_mean", "average_precision_mean"]]
        .mean()
    )
    r50 = (
        thr[(thr.model_id.isin(models)) & (thr.threshold == 0.5)]
        .groupby("model_id", as_index=False)["recall_mean"]
        .mean()
        .rename(columns={"recall_mean": "recall_0_50"})
    )
    r01 = (
        thr[(thr.model_id.isin(models)) & (thr.threshold == 0.01)]
        .groupby("model_id", as_index=False)["recall_mean"]
        .mean()
        .rename(columns={"recall_mean": "recall_0_01"})
    )
    occ_m = (
        occ[occ.model_id.isin(models)]
        .groupby("model_id", as_index=False)[["threshold_occupancy_persistence_mean", "posterior_sparsity_index_mean"]]
        .mean()
        .rename(
            columns={
                "threshold_occupancy_persistence_mean": "accessibility_persistence",
                "posterior_sparsity_index_mean": "low_score_mass_proxy",
            }
        )
    )
    reg_m = reg[reg.model_id.isin(models)][
        ["model_id", "mean_operational_smoothness_index", "mean_max_recall_jump", "inferred_regime", "mean_fraction_below_0_01"]
    ].rename(
        columns={
            "mean_operational_smoothness_index": "operational_smoothness",
            "mean_max_recall_jump": "max_recall_jump",
            "inferred_regime": "recurring_pattern",
            "mean_fraction_below_0_01": "low_score_mass_lt_0_01",
        }
    )

    merged = bench_m.merge(r50, on="model_id").merge(r01, on="model_id").merge(occ_m, on="model_id").merge(reg_m, on="model_id")
    merged["recovery"] = merged["recall_0_01"] - merged["recall_0_50"]
    merged["Model"] = merged["model_id"].map(names)

    ordered = merged.set_index("model_id").loc[models].reset_index()

    # Save numeric reference
    ordered.to_csv(out_tables / "_allocator_family_summary_values_v2.csv", index=False)

    # Draw v2 figures
    colors = {
        "quantized_allocator": "#6b6b6b",
        "broad_allocator": "#1b9e77",
        "cliff_allocator": "#d95f02",
        "conservative_allocator": "#1f78b4",
    }

    def draw(xcol: str, ycol: str, title: str, out_name: str) -> None:
        plt.figure(figsize=(8.4, 6.0), dpi=300)
        for _, row in ordered.iterrows():
            c = colors.get(row["recurring_pattern"], "#333333")
            plt.scatter(row[xcol], row[ycol], s=90, color=c, edgecolor="white", linewidth=0.6)
            plt.text(row[xcol] + 0.006, row[ycol] + 0.006, row["Model"].replace(" (hddt_forest)", ""), fontsize=8)
        plt.xlabel(xcol.replace("_", " ").title())
        plt.ylabel(ycol.replace("_", " ").title())
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_figs / out_name, bbox_inches="tight")
        plt.close()

    draw(
        "operational_smoothness",
        "accessibility_persistence",
        "Accessibility Regime Map v2: Smoothness vs Threshold Occupancy Persistence",
        "figure_accessibility_regime_map_v2.png",
    )
    draw(
        "max_recall_jump",
        "accessibility_persistence",
        "Accessibility Regime Map v2A: Jump vs Threshold Occupancy Persistence",
        "figure_accessibility_regime_map_variant_a_v2.png",
    )
    draw(
        "recovery",
        "accessibility_persistence",
        "Accessibility Regime Map v2B: Recovery vs Threshold Occupancy Persistence",
        "figure_accessibility_regime_map_variant_b_v2.png",
    )


if __name__ == "__main__":
    main()
