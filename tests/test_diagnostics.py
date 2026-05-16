from importlib.util import find_spec

import numpy as np
import pytest

from hib.diagnostics import (
    lightgbm_weighting_diagnostic,
    probability_differences,
    resolve_lightgbm_variant_params,
    write_lightgbm_diagnostic_markdown,
)
from hib.models import OptionalDependencyUnavailable
from hib.runner import compute_binary_class_weight_ratio
from hib.synthetic import SyntheticSkewConfig, make_train_test_split


def test_scale_pos_weight_auto_resolves_expected_ratio():
    y = np.array([0, 0, 0, 1, 1])
    assert compute_binary_class_weight_ratio(y) == 1.5


def test_diagnostic_probability_comparison_computes_differences():
    first = np.array([0.1, 0.2, 0.3])
    second = np.array([0.1, 0.1, 0.4])
    result = probability_differences(first, second)
    assert np.isclose(result["max_abs_probability_diff"], 0.1)
    assert result["identical_prediction_count"] == 1


def test_diagnostic_report_generation(tmp_path):
    report = {
        "n_positive_train": 10,
        "n_negative_train": 90,
        "computed_ratio": 9.0,
        "resolved_params": {
            "lightgbm": {"is_unbalance": None, "scale_pos_weight": 1.0},
            "lightgbm_unbalanced": {"is_unbalance": True, "scale_pos_weight": 1.0},
            "lightgbm_weighted": {"is_unbalance": None, "scale_pos_weight": 9.0},
        },
        "comparisons": {
            "lightgbm_vs_lightgbm_weighted": {
                "max_abs_probability_diff": 0.2,
                "correlation": 0.9,
                "identical_prediction_count": 3,
            }
        },
    }
    path = write_lightgbm_diagnostic_markdown(report, tmp_path / "diag.md")
    content = path.read_text(encoding="utf-8")
    assert path.exists()
    assert "LightGBM Weighting Diagnostic" in content


def test_lightgbm_param_resolution_differs_between_variants_when_available():
    if find_spec("lightgbm") is None:
        pytest.skip("lightgbm not installed")

    y = np.array([0] * 20 + [1] * 5)
    resolved = resolve_lightgbm_variant_params(y, seed=0)

    assert resolved["lightgbm_unbalanced"]["is_unbalance"] is True
    assert resolved["lightgbm_weighted"]["is_unbalance"] in (None, False)
    assert resolved["lightgbm_weighted"]["scale_pos_weight"] == 4.0


def test_lightgbm_diagnostic_skips_gracefully_when_unavailable():
    if find_spec("lightgbm") is not None:
        pytest.skip("lightgbm installed")

    config = SyntheticSkewConfig(skew_ratio=1, seed=0, minority_count=20)
    X_train, X_test, y_train, y_test = make_train_test_split(config)
    with pytest.raises(OptionalDependencyUnavailable):
        lightgbm_weighting_diagnostic(X_train, y_train, X_test, y_test, seed=0)
