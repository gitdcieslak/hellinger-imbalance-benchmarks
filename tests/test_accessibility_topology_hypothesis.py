import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_topology_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_topology_hypothesis", script)
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
                    "minority_survival_auc": 0.2 + 0.6 * q,
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


def test_surrogate_curve_is_bounded_and_monotone():
    module = _load_module()
    curve = module.surrogate_reachability_curve(0.75, 0.8, 2, n_grid=51)

    assert curve.shape == (51,)
    assert np.all(curve >= 0)
    assert np.all(curve <= 1)
    assert np.all(np.diff(curve) <= 1e-9)


def test_build_reachability_matrix_shape():
    module = _load_module()
    matrix, grid = module.build_reachability_matrix(_sample_df(), n_grid=21)

    assert matrix.shape == (36, 21)
    assert grid.shape == (21,)


def test_metric_reconstruction_schema():
    module = _load_module()
    df = _sample_df()
    matrix, _ = module.build_reachability_matrix(df, n_grid=21)
    scores = module.metric_reconstruction(df, matrix)

    assert {"target", "feature_set", "grouped_cv_score", "metric", "valid"}.issubset(scores.columns)
    assert set([module.TARGET, module.SURVIVAL, "breadth", "elevation", "regime_id", "model_id"]).issubset(set(scores["target"]))


def test_decide_outcome_supported():
    module = _load_module()
    scores = pd.DataFrame(
        {
            "target": [module.TARGET, module.SURVIVAL, "breadth", "elevation"],
            "grouped_cv_score": [0.95, 0.95, 0.9, 0.9],
        }
    )
    outcome, text = module.decide_outcome(scores)

    assert outcome == "A"
    assert "projections" in text
