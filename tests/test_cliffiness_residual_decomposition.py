import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_cliffiness_residual_decomposition.py"
    spec = importlib.util.spec_from_file_location("report_cliffiness_residual_decomposition", script)
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


def test_curve_features_have_expected_columns():
    module = _load_module()
    _, matrix, thresholds = _matrix_and_meta()

    features = module.curve_features(matrix, thresholds)

    for col in ["largest_drop_threshold", "drop_gini", "max_curvature", "compression_ratio", "unique_ratio_proxy"]:
        assert col in features.columns
    assert len(features) == len(matrix)


def test_grouped_regression_returns_predictions():
    module = _load_module()
    meta, matrix, _ = _matrix_and_meta()

    pred, score, valid, reason = module.grouped_regression(meta, matrix, module.TARGET)

    assert valid
    assert reason == ""
    assert len(pred) == len(meta)
    assert np.isfinite(score)


def test_variance_partition_bounds():
    module = _load_module()
    scores = pd.DataFrame(
        {
            "feature_group": ["all_geometry_features", "calibration_features"],
            "residual_r2": [0.5, 0.1],
        }
    )

    part = module.variance_partition(0.7, scores)

    assert 0 <= part["topology_fraction"] <= 1
    assert 0 <= part["geometry_fraction"] <= 1
    assert 0 <= part["unexplained_fraction"] <= 1


def test_span_for_loss_positive():
    module = _load_module()
    thresholds = np.linspace(0, 1, 11)
    curve = np.array([1, 1, 1, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0, 0])

    span = module.span_for_loss(thresholds, curve, 0.5)

    assert span > 0
    assert span <= 1
