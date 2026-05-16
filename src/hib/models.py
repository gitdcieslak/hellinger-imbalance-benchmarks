"""Model registry for benchmark baselines."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from hellinger_tree import HellingerDecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


ModelFactory = Callable[[int], Any]


class OptionalDependencyUnavailable(RuntimeError):
    """Raised when a configured optional model dependency is unavailable."""

    def __init__(self, model_id: str, package_name: str) -> None:
        super().__init__(
            f"model {model_id!r} requires optional package {package_name!r}; "
            "install the package or remove the model from the config"
        )
        self.model_id = model_id
        self.package_name = package_name


def _make_hddt(seed: int) -> HellingerDecisionTreeClassifier:
    return HellingerDecisionTreeClassifier(random_state=seed)


def _make_bagged_hddt(seed: int) -> BaggingClassifier:
    base = HellingerDecisionTreeClassifier(random_state=seed)
    kwargs: dict[str, Any] = {
        "n_estimators": 10,
        "random_state": seed,
        "n_jobs": 1,
    }
    try:
        return BaggingClassifier(estimator=base, **kwargs)
    except TypeError:
        return BaggingClassifier(base_estimator=base, **kwargs)


def _make_hddt_forest(seed: int) -> BaggingClassifier:
    base = HellingerDecisionTreeClassifier(random_state=seed)
    kwargs: dict[str, Any] = {
        "n_estimators": 50,
        "max_features": 1.0,
        "bootstrap": True,
        "bootstrap_features": False,
        "random_state": seed,
        "n_jobs": 1,
    }
    try:
        model = BaggingClassifier(estimator=base, **kwargs)
    except TypeError:
        model = BaggingClassifier(base_estimator=base, **kwargs)
    setattr(model, "_hib_max_features_spec", "sqrt")
    return model


def _make_cart(seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=seed)


def _make_cart_balanced(seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=seed, class_weight="balanced")


def _make_random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1)


def _make_random_forest_balanced(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=50,
        random_state=seed,
        n_jobs=1,
        class_weight="balanced_subsample",
    )


def _make_xgboost(seed: int) -> Any:
    try:
        xgboost = import_module("xgboost")
    except ImportError as exc:
        raise OptionalDependencyUnavailable("xgboost", "xgboost") from exc

    return xgboost.XGBClassifier(
        n_estimators=25,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def _make_xgboost_weighted(seed: int) -> Any:
    try:
        xgboost = import_module("xgboost")
    except ImportError as exc:
        raise OptionalDependencyUnavailable("xgboost_weighted", "xgboost") from exc

    return xgboost.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        scale_pos_weight=1.0,
    )


def _make_lightgbm(seed: int) -> Any:
    try:
        lightgbm = import_module("lightgbm")
    except ImportError as exc:
        raise OptionalDependencyUnavailable("lightgbm", "lightgbm") from exc

    return lightgbm.LGBMClassifier(
        n_estimators=50,
        max_depth=-1,
        learning_rate=0.1,
        objective="binary",
        min_child_samples=1,
        min_data_in_leaf=1,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        n_jobs=1,
        verbose=-1,
    )


def _make_lightgbm_unbalanced(seed: int) -> Any:
    try:
        lightgbm = import_module("lightgbm")
    except ImportError as exc:
        raise OptionalDependencyUnavailable("lightgbm_unbalanced", "lightgbm") from exc

    return lightgbm.LGBMClassifier(
        n_estimators=50,
        max_depth=-1,
        learning_rate=0.1,
        objective="binary",
        min_child_samples=1,
        min_data_in_leaf=1,
        is_unbalance=True,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        n_jobs=1,
        verbose=-1,
    )


def _make_lightgbm_weighted(seed: int) -> Any:
    try:
        lightgbm = import_module("lightgbm")
    except ImportError as exc:
        raise OptionalDependencyUnavailable("lightgbm_weighted", "lightgbm") from exc

    return lightgbm.LGBMClassifier(
        n_estimators=50,
        max_depth=-1,
        learning_rate=0.1,
        objective="binary",
        min_child_samples=1,
        min_data_in_leaf=1,
        scale_pos_weight=1.0,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        n_jobs=1,
        verbose=-1,
    )


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "hddt": _make_hddt,
    "bagged_hddt": _make_bagged_hddt,
    "hddt_forest": _make_hddt_forest,
    "random_subspace_hddt": _make_hddt_forest,
    "cart": _make_cart,
    "cart_balanced": _make_cart_balanced,
    "random_forest": _make_random_forest,
    "random_forest_balanced": _make_random_forest_balanced,
    "xgboost": _make_xgboost,
    "xgboost_weighted": _make_xgboost_weighted,
    "lightgbm": _make_lightgbm,
    "lightgbm_unbalanced": _make_lightgbm_unbalanced,
    "lightgbm_weighted": _make_lightgbm_weighted,
}


def available_model_ids() -> list[str]:
    """Return required model ids in deterministic order."""

    return list(MODEL_REGISTRY)


def make_model(model_id: str, seed: int) -> Any:
    """Instantiate a configured model baseline."""

    try:
        factory = MODEL_REGISTRY[model_id]
    except KeyError as exc:
        known = ", ".join(available_model_ids())
        raise ValueError(f"unknown model_id {model_id!r}; expected one of: {known}") from exc
    return factory(seed)
