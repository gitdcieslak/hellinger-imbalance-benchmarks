import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_fig = root / "paper" / "manuscript_figures"
    out_tab = root / "paper" / "manuscript_tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    bench = pd.read_csv(root / "reports" / "neural_mlp" / "legacy_benchmark_summary.csv")
    thr = pd.read_csv(root / "reports" / "neural_mlp" / "legacy_threshold_sweep_summary.csv")
    reg = pd.read_csv(root / "reports" / "neural_mlp" / "allocation_regime_summary.csv")

    family_models = ["cart", "hddt", "hddt_forest", "random_forest", "xgboost", "lightgbm", "mlp"]
    display = {
        "cart": "CART",
        "hddt": "HDDT",
        "hddt_forest": "Bagged HDDT (hddt_forest)",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "mlp": "MLP",
    }
    colors = {
        "cart": "#6b6b6b",
        "hddt": "#1b9e77",
        "hddt_forest": "#66a61e",
        "random_forest": "#7570b3",
        "xgboost": "#d95f02",
        "lightgbm": "#1f78b4",
        "mlp": "#e7298a",
    }

    thr_f = thr[thr["model_id"].isin(family_models)].copy()

    threshold_order = [0.50, 0.25, 0.10, 0.05, 0.01]
    reach = thr_f.groupby(["model_id", "threshold"], as_index=False)["recall_mean"].mean()
    reach["threshold"] = pd.Categorical(reach["threshold"], categories=threshold_order, ordered=True)
    reach = reach.sort_values(["model_id", "threshold"])

    plt.figure(figsize=(9, 5.8), dpi=300)
    for m in family_models:
        s = reach[reach["model_id"] == m].sort_values("threshold")
        if s.empty:
            continue
        plt.plot(
            s["threshold"].astype(float),
            s["recall_mean"],
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=colors[m],
            label=display[m],
        )

    plt.gca().invert_xaxis()
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Decision Threshold")
    plt.ylabel("Minority Reachability / Recall")
    plt.title("Family-Level Reachability Trajectories Under Severe Imbalance")
    plt.grid(alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(out_fig / "figure_family_reachability_comparison.png", bbox_inches="tight")
    plt.close()

    bench_m = (
        bench[bench["model_id"].isin(family_models)]
        .groupby("model_id", as_index=False)[["auroc_mean", "average_precision_mean"]]
        .mean()
    )
    r50 = (
        thr_f[thr_f["threshold"] == 0.5]
        .groupby("model_id", as_index=False)["recall_mean"]
        .mean()
        .rename(columns={"recall_mean": "recall_0_50"})
    )
    r01 = (
        thr_f[thr_f["threshold"] == 0.01]
        .groupby("model_id", as_index=False)["recall_mean"]
        .mean()
        .rename(columns={"recall_mean": "recall_0_01"})
    )
    reg_m = reg[reg["model_id"].isin(family_models)][
        [
            "model_id",
            "mean_fraction_below_0_01",
            "mean_operational_smoothness_index",
            "mean_max_recall_jump",
            "inferred_regime",
        ]
    ].copy()
    reg_m = reg_m.rename(
        columns={
            "mean_fraction_below_0_01": "accessibility_persistence",
            "mean_operational_smoothness_index": "operational_smoothness",
            "mean_max_recall_jump": "max_recall_jump",
            "inferred_regime": "recurring_pattern",
        }
    )

    agg = bench_m.merge(r50, on="model_id", how="left").merge(r01, on="model_id", how="left").merge(reg_m, on="model_id", how="left")
    agg["recovery"] = agg["recall_0_01"] - agg["recall_0_50"]

    plt.figure(figsize=(8.2, 5.6), dpi=300)
    for _, row in agg.iterrows():
        m = row["model_id"]
        plt.scatter(row["auroc_mean"], row["accessibility_persistence"], s=80, color=colors[m])
        plt.text(row["auroc_mean"] + 0.003, row["accessibility_persistence"] + 0.008, display[m], fontsize=8)

    plt.xlabel("AUROC (dataset mean)")
    plt.ylabel("Accessibility Persistence (mean fraction below 0.01)")
    plt.title("Ranking Quality vs Accessibility Persistence Across Families")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_fig / "figure_ranking_vs_accessibility_scatter.png", bbox_inches="tight")
    plt.close()

    agg.to_csv(out_tab / "_allocator_family_summary_values.csv", index=False)


if __name__ == "__main__":
    main()
