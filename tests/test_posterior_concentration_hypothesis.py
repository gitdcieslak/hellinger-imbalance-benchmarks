import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_posterior_concentration_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_posterior_concentration_hypothesis", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _matrix_and_meta():
    thresholds = np.linspace(0, 1, 11)
    curves = []
    rows = []
    for i in range(24):
        cliff = 0.8 if i % 2 else 0.2
        curve = np.clip(1 - thresholds * (0.4 + cliff), 0, 1)
        curves.append(curve)
        rows.append(
            {
                "run_id": f"r{i}",
                "dataset_id": f"d{i % 4}",
                "model_id": "cart" if i % 2 else "rf",
                "minority_survival_cliffiness": cliff,
                "minority_survival_auc": float(np.trapezoid(curve, thresholds)),
                "breadth": 0.2 + i / 100,
                "elevation": 0.8,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(curves), thresholds


def test_concentration_features_expected_columns():
    module = _load_module()
    _, matrix, thresholds = _matrix_and_meta()

    features = module.concentration_features(matrix, thresholds)

    for col in ["positive_score_entropy", "positive_score_gini", "top_5pct_mass", "effective_support", "density_around_095"]:
        assert col in features.columns
    assert len(features) == len(matrix)


def test_residual_regression_schema():
    module = _load_module()
    meta, matrix, thresholds = _matrix_and_meta()
    meta["cliffiness_residual"] = meta["minority_survival_cliffiness"] - meta["minority_survival_cliffiness"].mean()
    features = module.concentration_features(matrix, thresholds)

    scores, _ = module.residual_regression(meta, features)

    assert {"feature_group", "residual_r2", "n_features", "valid", "skip_reason"}.issubset(scores.columns)
    assert "all_concentration" in set(scores["feature_group"])


def test_variance_partition_bounds():
    module = _load_module()

    part = module.variance_partition(0.7, 0.4, 0.1)

    assert 0 <= part["topology_fraction"] <= 1
    assert 0 <= part["concentration_fraction"] <= 1
    assert 0 <= part["unexplained_fraction"] <= 1


def test_decision_thresholds():
    module = _load_module()

    assert module.decide(0.6)[0] == "Outcome A"
    assert module.decide(0.3)[0] == "Outcome B"
    assert module.decide(0.1)[0] == "Outcome C"


def test_score_distribution_from_curve_sums_to_one():
    module = _load_module()
    thresholds = np.linspace(0, 1, 11)
    curve = np.linspace(1, 0, 11)

    _, weights = module.score_distribution_from_curve(curve, thresholds)

    assert np.isclose(weights.sum(), 1.0)
