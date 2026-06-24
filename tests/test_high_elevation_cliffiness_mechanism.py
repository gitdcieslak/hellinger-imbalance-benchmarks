import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_high_elevation_cliffiness_mechanism.py"
    spec = importlib.util.spec_from_file_location("report_high_elevation_cliffiness_mechanism", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _sample_df():
    rows = []
    for dataset in ["a", "b", "c", "d"]:
        for i in range(12):
            quantized = i < 6
            rows.append(
                {
                    "dataset_name": dataset,
                    "task_id": f"{dataset}:1",
                    "model_id": "cart" if quantized else "random_forest",
                    "breadth": 0.2 + 0.01 * i,
                    "elevation": 0.9,
                    "minority_survival_auc": 0.9,
                    "minority_survival_cliffiness": 0.8 if quantized else 0.2,
                    "positive_unique_score_ratio": 0.05 if quantized else 0.5,
                    "positive_effective_score_bins": 2 if quantized else 20,
                    "positive_histogram_entropy": 0.2 if quantized else 1.2,
                    "positive_top_bin_mass": 0.9 if quantized else 0.2,
                    "positive_max_bin_mass": 0.9 if quantized else 0.2,
                    "positive_quantization_score": 0.95 if quantized else 0.1,
                    "positive_score_iqr": 0.0 if quantized else 0.2,
                    "positive_score_q10_q90_width": 0.0 if quantized else 0.4,
                    "positive_score_gini_or_concentration_index": 0.9 if quantized else 0.2,
                    "brier_score": 0.08 if quantized else 0.03,
                    "positive_fraction": 0.2,
                    "n_rows": 100,
                    "positive_count": 20,
                    "n_features_processed": 5,
                    "auroc": 0.9,
                    "average_precision": 0.8,
                }
            )
    rows.append(
        {
            "dataset_name": "outside",
            "task_id": "outside:1",
            "model_id": "cart",
            "breadth": 1.0,
            "elevation": 0.2,
            "minority_survival_auc": 0.1,
            "minority_survival_cliffiness": 0.1,
        }
    )
    return pd.DataFrame(rows)


def test_high_elevation_region_filter_and_terciles():
    module = _load_module()
    region = module.high_elevation_region(_sample_df())

    assert len(region) == 48
    assert (region["elevation"] >= 0.8).all()
    assert (region["breadth"] <= 0.7).all()
    assert set(region["cliffiness_tercile"]) == {"low", "medium", "high"}


def test_grouped_cv_r2_returns_valid_score():
    module = _load_module()
    region = module.high_elevation_region(_sample_df())
    X = module._features(region, ["positive_unique_score_ratio", "positive_effective_score_bins"])

    score, valid, reason = module.grouped_cv_r2(region, X)

    assert valid
    assert reason == ""
    assert np.isfinite(score)


def test_quantization_summary_direction():
    module = _load_module()
    region = module.high_elevation_region(_sample_df())

    summary = module.quantization_summary(region)

    unique = summary[summary["metric"] == "unique_posterior_ratio"].iloc[0]
    top_bin = summary[summary["metric"] == "top_bin_mass"].iloc[0]
    assert unique["delta_quantized_minus_other"] > 0
    assert top_bin["delta_quantized_minus_other"] > 0


def test_classify_outcome_returns_named_outcome():
    module = _load_module()
    scores = pd.DataFrame(
        {
            "model_spec": ["morphology_plus_occupancy", "morphology_plus_topology", "morphology_plus_model_family", "all_available_mechanisms"],
            "grouped_cv_r2": [0.6, np.nan, 0.2, 0.65],
        }
    )
    outcome, text = module.classify_outcome(scores, pd.DataFrame(), pd.DataFrame())

    assert outcome == "Outcome A"
    assert "occupancy" in text
