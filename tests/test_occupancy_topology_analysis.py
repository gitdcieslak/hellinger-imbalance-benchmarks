import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_occupancy_topology_analysis.py"
    spec = importlib.util.spec_from_file_location("report_occupancy_topology_analysis", script)
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
                    "model_id": "cart" if i < 6 else "rf",
                    "regime_id": 0 if i < 6 else 1,
                    "breadth": q,
                    "elevation": 1 - q,
                    "minority_survival_auc": 1 - q / 2,
                    "minority_survival_cliffiness": q,
                    "minority_survival_max_drop": q,
                    "minority_survival_effective_drop_count": 1 + 10 * (1 - q),
                    "positive_effective_score_bins": 1 + 10 * q,
                    "positive_histogram_entropy": q,
                    "positive_unique_score_ratio": q,
                    "positive_top_bin_mass": 1 - q,
                    "positive_max_bin_mass": 1 - q,
                    "positive_quantization_score": 1 - q,
                    "positive_score_iqr": q,
                    "positive_score_q10_q90_width": q,
                    "positive_score_gini_or_concentration_index": 1 - q,
                }
            )
    return pd.DataFrame(rows)


def test_build_topology_features_contains_expected_proxies():
    module = _load_module()
    features = module.build_topology_features(_sample_df())

    assert "largest_drop" in features.columns
    assert "max_gap_proxy" in features.columns
    assert "effective_drop_count" in features.columns
    assert len(features) == 36


def test_variance_decomposition_schema():
    module = _load_module()
    df = _sample_df()
    topology = module.build_topology_features(df)

    decomp = module.variance_decomposition(df, topology)

    assert {"model_spec", "grouped_cv_r2", "delta_vs_occupancy", "valid"}.issubset(decomp.columns)
    assert "occupancy_plus_topology" in set(decomp["model_spec"])


def test_mediation_summary_reduction():
    module = _load_module()
    decomp = pd.DataFrame(
        {
            "model_spec": ["occupancy_only", "occupancy_plus_model", "occupancy_plus_topology", "occupancy_plus_topology_plus_model"],
            "grouped_cv_r2": [0.4, 0.8, 0.7, 0.75],
        }
    )

    summary = module.mediation_summary(decomp)

    assert summary["topology_absorbs_model_signal"]
    assert summary["topology_mediated_fraction_of_model_effect"] > 0.5


def test_classify_outcome_model_signal_remains():
    module = _load_module()
    decomp = pd.DataFrame(
        {
            "model_spec": ["occupancy_only", "occupancy_plus_model", "occupancy_plus_topology", "occupancy_plus_topology_plus_model"],
            "grouped_cv_r2": [0.4, 0.75, 0.45, 0.78],
        }
    )
    mediation = module.mediation_summary(decomp)

    outcome, text = module.classify_outcome(decomp, mediation)

    assert outcome == "Outcome B"
    assert "Model-family" in text
