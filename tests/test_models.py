from importlib.util import find_spec

import numpy as np
import pytest

from hib.models import OptionalDependencyUnavailable, available_model_ids, make_model
from hib.runner import apply_split_dependent_model_params, compute_binary_class_weight_ratio
from hib.synthetic import SyntheticSkewConfig, make_train_test_split


def test_model_registry_instantiates_required_models():
    required = {
        "hddt",
        "bagged_hddt",
        "hddt_forest",
        "random_subspace_hddt",
        "cart",
        "random_forest",
        "xgboost",
        "lightgbm",
        "cart_balanced",
        "random_forest_balanced",
        "xgboost_weighted",
        "lightgbm_unbalanced",
        "lightgbm_weighted",
    }

    assert required == set(available_model_ids())
    for model_id in {
        "hddt",
        "bagged_hddt",
        "hddt_forest",
        "random_subspace_hddt",
        "cart",
        "random_forest",
        "cart_balanced",
        "random_forest_balanced",
    }:
        model = make_model(model_id, seed=0)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


@pytest.mark.parametrize(
    ("model_id", "package_name"),
    [("xgboost", "xgboost"), ("lightgbm", "lightgbm")],
)
def test_optional_boosted_tree_models_skip_gracefully_when_unavailable(
    model_id,
    package_name,
):
    if find_spec(package_name) is not None:
        pytest.skip(f"{package_name} is installed")

    with pytest.raises(OptionalDependencyUnavailable) as exc_info:
        make_model(model_id, seed=0)

    assert exc_info.value.model_id == model_id
    assert exc_info.value.package_name == package_name


@pytest.mark.parametrize(
    ("model_id", "package_name"),
    [
        ("xgboost", "xgboost"),
        ("xgboost_weighted", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("lightgbm_unbalanced", "lightgbm"),
        ("lightgbm_weighted", "lightgbm"),
    ],
)
def test_optional_weighted_boosted_models_skip_gracefully_when_unavailable(model_id, package_name):
    if find_spec(package_name) is not None:
        pytest.skip(f"{package_name} is installed")

    with pytest.raises(OptionalDependencyUnavailable):
        make_model(model_id, seed=0)


@pytest.mark.parametrize(
    ("model_id", "package_name"),
    [("xgboost", "xgboost"), ("lightgbm", "lightgbm")],
)
def test_optional_boosted_tree_models_fit_predict_when_available(
    model_id,
    package_name,
):
    if find_spec(package_name) is None:
        pytest.skip(f"{package_name} is not installed")

    config = SyntheticSkewConfig(skew_ratio=1, seed=11, minority_count=20)
    X_train, X_test, y_train, _ = make_train_test_split(config)
    model = make_model(model_id, seed=11)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)


@pytest.mark.parametrize(
    ("model_id", "package_name"),
    [
        ("xgboost_weighted", "xgboost"),
        ("lightgbm_unbalanced", "lightgbm"),
        ("lightgbm_weighted", "lightgbm"),
    ],
)
def test_weighted_optional_models_fit_predict_when_available(model_id, package_name):
    if find_spec(package_name) is None:
        pytest.skip(f"{package_name} is not installed")

    config = SyntheticSkewConfig(skew_ratio=10, seed=11, minority_count=20)
    X_train, X_test, y_train, _ = make_train_test_split(config)
    model = make_model(model_id, seed=11)
    model = apply_split_dependent_model_params(model, model_id, y_train)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)


def test_compute_binary_class_weight_ratio_resolves_known_case():
    y_train = [0, 0, 0, 1]
    ratio = compute_binary_class_weight_ratio(y_train)
    assert ratio == 3.0


def test_compute_binary_class_weight_ratio_raises_on_missing_class():
    with pytest.raises(ValueError, match="zero positive"):
        compute_binary_class_weight_ratio([0, 0, 0])
    with pytest.raises(ValueError, match="zero negative"):
        compute_binary_class_weight_ratio([1, 1, 1])


def test_required_model_registry_entries_instantiate():
    required = {"hddt", "bagged_hddt", "hddt_forest", "cart", "random_forest"}
    for model_id in required:
        model = make_model(model_id, seed=0)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


def test_hddt_forest_fit_predict_proba_and_determinism():
    config = SyntheticSkewConfig(skew_ratio=1, seed=3, separation=2.0, minority_count=40, test_size=0.5)
    X_train, X_test, y_train, _ = make_train_test_split(config)

    first = make_model("hddt_forest", seed=123)
    second = make_model("hddt_forest", seed=123)
    first.fit(X_train, y_train)
    second.fit(X_train, y_train)
    first_proba = first.predict_proba(X_test)[:, 1]
    second_proba = second.predict_proba(X_test)[:, 1]

    assert first_proba.shape[0] == X_test.shape[0]
    np.testing.assert_allclose(first_proba, second_proba)


def test_lightgbm_non_constant_probabilities_on_separable_data_when_available():
    if find_spec("lightgbm") is None:
        pytest.skip("lightgbm is not installed")

    config = SyntheticSkewConfig(
        skew_ratio=1,
        seed=17,
        separation=3.0,
        minority_count=50,
        n_features=6,
        noise=0.6,
        test_size=0.5,
    )
    X_train, X_test, y_train, _ = make_train_test_split(config)
    model = make_model("lightgbm", seed=17)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    assert probabilities.size > 0
    assert probabilities.std() > 0.0
    assert probabilities.min() < probabilities.max()
