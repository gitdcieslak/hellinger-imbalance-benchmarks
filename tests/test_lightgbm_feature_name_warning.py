from importlib.util import find_spec

import warnings

import pytest

from hib.metrics import positive_class_scores
from hib.models import make_model
from hib.runner import fit_or_skip_model
from hib.synthetic import SyntheticSkewConfig, make_train_test_split


def test_lightgbm_prediction_has_no_feature_name_warning_when_available():
    if find_spec("lightgbm") is None:
        pytest.skip("lightgbm not installed")

    config = SyntheticSkewConfig(skew_ratio=1, seed=13, minority_count=30)
    X_train, X_test, y_train, _ = make_train_test_split(config)
    model = make_model("lightgbm", seed=13)
    assert fit_or_skip_model(model, X_train, y_train)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _ = positive_class_scores(model, X_test)

    messages = [str(item.message) for item in captured]
    assert not any("does not have valid feature names" in message for message in messages)
