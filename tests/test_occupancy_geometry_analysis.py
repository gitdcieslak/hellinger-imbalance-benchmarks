import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_occupancy_geometry_analysis.py"
    spec = importlib.util.spec_from_file_location("report_occupancy_geometry_analysis", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _sample_df():
    rows = []
    for dataset in ["a", "b", "c"]:
        for i in range(12):
            q = i / 11
            rows.append(
                {
                    "dataset_name": dataset,
                    "model_id": "cart" if i < 4 else "rf" if i < 8 else "lr",
                    "regime_id": 0 if i < 6 else 1,
                    "breadth": q,
                    "elevation": 1 - q,
                    "minority_survival_auc": 1 - q / 2,
                    "minority_survival_cliffiness": q,
                    "positive_effective_score_bins": 1 + 10 * q,
                    "positive_histogram_entropy": q,
                    "positive_unique_score_ratio": q,
                    "positive_top_bin_mass": 1 - q,
                    "positive_max_bin_mass": 1 - q,
                    "positive_quantization_score": 1 - q,
                    "positive_score_iqr": q,
                    "positive_score_q10_q90_width": q,
                    "positive_score_gini_or_concentration_index": 1 - q,
                    "positive_fraction": 0.2,
                    "n_rows": 100,
                    "positive_count": 20,
                    "n_features_processed": 5,
                    "auroc": 0.8,
                    "average_precision": 0.7,
                    "brier_score": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_feature_matrix_numeric_and_categorical():
    module = _load_module()
    df = _sample_df()

    X = module.feature_matrix(df, ["breadth", "missing"], ["model_id"])

    assert "breadth" in X.columns
    assert any(c.startswith("model_id_") for c in X.columns)
    assert "missing" not in X.columns


def test_grouped_regression_r2_valid():
    module = _load_module()
    df = _sample_df()
    X = module.feature_matrix(df, module.OCCUPANCY_COLS)

    score, valid, reason = module.grouped_regression_r2(df, X, "breadth")

    assert valid
    assert reason == ""
    assert np.isfinite(score)


def test_archetype_name_rules():
    module = _load_module()

    name = module.archetype_name(
        pd.Series(
            {
                "mean_positive_unique_score_ratio": 0.05,
                "mean_positive_top_bin_mass": 0.96,
                "mean_positive_score_iqr": 0.01,
                "mean_positive_histogram_entropy": 0.1,
            }
        )
    )

    assert name == "Monopolized quantized allocator"


def test_outcome_decision_occupancy_fundamental():
    module = _load_module()
    morphology_scores = pd.DataFrame(
        {
            "target": ["joint_morphology_position"],
            "grouped_cv_r2": [0.7],
        }
    )
    regime_scores = pd.DataFrame({"grouped_cv_accuracy": [0.85]})
    redundancy = pd.DataFrame(
        {
            "target": [module.TARGET, module.TARGET, module.TARGET],
            "model_spec": ["morphology_only", "occupancy_only", "morphology_plus_occupancy"],
            "grouped_cv_r2": [0.2, 0.7, 0.72],
        }
    )
    outcome, text = module.outcome_decision(morphology_scores, regime_scores, redundancy, pd.DataFrame())

    assert outcome == "Outcome A"
    assert "fundamental" in text


def test_grouped_classification_schema():
    module = _load_module()
    df = _sample_df()
    X = module.feature_matrix(df, module.OCCUPANCY_COLS)

    scores, _ = module.grouped_classification(df, X)

    assert {"classifier", "grouped_cv_accuracy", "macro_f1", "valid", "skip_reason"}.issubset(scores.columns)
