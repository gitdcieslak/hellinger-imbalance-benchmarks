from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def draw_scatter(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    colors = {
        "quantized_allocator": "#6b6b6b",
        "broad_allocator": "#1b9e77",
        "cliff_allocator": "#d95f02",
        "conservative_allocator": "#1f78b4",
    }
    labels = {
        "cart": "CART",
        "hddt": "HDDT",
        "hddt_forest": "Bagged HDDT",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "mlp": "MLP",
    }

    plt.figure(figsize=(8.4, 6.0), dpi=300)
    for _, row in df.iterrows():
        c = colors.get(row["recurring_pattern"], "#333333")
        plt.scatter(row[x], row[y], s=90, color=c, edgecolor="white", linewidth=0.6)
        plt.text(row[x] + 0.008, row[y] + 0.008, labels[row["model_id"]], fontsize=8)

    handles = []
    for k, c in colors.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w", label=k, markerfacecolor=c, markersize=8))

    plt.legend(handles=handles, loc="best", frameon=False, fontsize=8)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    in_path = root / "paper" / "manuscript_tables" / "_allocator_family_summary_values.csv"
    out_dir = root / "paper" / "manuscript_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    draw_scatter(
        df,
        "operational_smoothness",
        "accessibility_persistence",
        "Accessibility Regime Map: Smoothness vs Persistence",
        out_dir / "figure_accessibility_regime_map.png",
    )

    draw_scatter(
        df,
        "max_recall_jump",
        "accessibility_persistence",
        "Accessibility Regime Variant A: Jump vs Persistence",
        out_dir / "figure_accessibility_regime_map_variant_a.png",
    )

    draw_scatter(
        df,
        "recovery",
        "accessibility_persistence",
        "Accessibility Regime Variant B: Recovery vs Persistence",
        out_dir / "figure_accessibility_regime_map_variant_b.png",
    )


if __name__ == "__main__":
    main()
