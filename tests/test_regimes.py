import pandas as pd
import pytest

from hib.regimes import (
    allocation_regime_summary_to_markdown,
    combine_regime_metrics,
    infer_regime_labels,
    load_allocation_summary,
    load_elasticity_summary,
    plot_allocation_regime_scatter,
    summarize_operational_regimes,
)


def _allocation_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "boundary",
                "model_id": "cart",
                "histogram_entropy_mean": 0.2,
                "effective_support_size_mean": 1.2,
                "gini_coefficient_mean": 0.95,
                "fraction_scores_below_0_01_mean": 0.95,
                "fraction_scores_below_0_05_mean": 0.95,
            },
            {
                "dataset_id": "boundary",
                "model_id": "xgboost",
                "histogram_entropy_mean": 1.3,
                "effective_support_size_mean": 3.8,
                "gini_coefficient_mean": 0.45,
                "fraction_scores_below_0_01_mean": 0.25,
                "fraction_scores_below_0_05_mean": 0.6,
            },
        ]
    )


def _elasticity_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "boundary",
                "model_id": "cart",
                "mean_abs_recall_elasticity": 0.0,
                "mean_abs_precision_elasticity": 0.0,
                "operational_smoothness_index": 1.0,
                "max_recall_jump": 0.0,
                "max_precision_drop": 0.0,
                "threshold_of_max_recall_jump": "0.50->0.25",
            },
            {
                "dataset_id": "boundary",
                "model_id": "xgboost",
                "mean_abs_recall_elasticity": 5.0,
                "mean_abs_precision_elasticity": 1.0,
                "operational_smoothness_index": 0.15,
                "max_recall_jump": 0.5,
                "max_precision_drop": -0.2,
                "threshold_of_max_recall_jump": "0.05->0.01",
            },
        ]
    )


def test_loading_summaries_and_missing_columns(tmp_path):
    alloc = _allocation_df()
    ela = _elasticity_df()
    alloc_path = tmp_path / "alloc.csv"
    ela_path = tmp_path / "ela.csv"
    alloc.to_csv(alloc_path, index=False)
    ela.to_csv(ela_path, index=False)
    assert not load_allocation_summary(alloc_path).empty
    assert not load_elasticity_summary(ela_path).empty

    broken = alloc.drop(columns=["histogram_entropy_mean"])
    broken_path = tmp_path / "broken.csv"
    broken.to_csv(broken_path, index=False)
    with pytest.raises(ValueError, match="allocation summary missing columns"):
        load_allocation_summary(broken_path)


def test_combine_and_deterministic_labels():
    combined = combine_regime_metrics(_allocation_df(), _elasticity_df())
    summary = summarize_operational_regimes(combined)
    labeled = infer_regime_labels(summary)
    cart_label = labeled[labeled["model_id"] == "cart"].iloc[0]["inferred_regime"]
    xgb_label = labeled[labeled["model_id"] == "xgboost"].iloc[0]["inferred_regime"]
    assert cart_label == "quantized_allocator"
    assert xgb_label in {"cliff_allocator", "broad_allocator", "smooth_allocator", "conservative_allocator"}


def test_markdown_and_scatter_creation(tmp_path):
    combined = combine_regime_metrics(_allocation_df(), _elasticity_df())
    summary = summarize_operational_regimes(combined)
    md = allocation_regime_summary_to_markdown(summary)
    assert "Allocation Regime Summary" in md

    paths = plot_allocation_regime_scatter(summary, tmp_path)
    assert len(paths) == 4
    assert all(path.exists() for path in paths)
