"""Model registry for benchmark baselines."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from importlib import import_module
from typing import Any

import numpy as np
from hellinger_tree import HellingerDecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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


def _make_mlp(seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=seed,
    )


class _MlpOversampledAdapter:
    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self._model = _make_mlp(seed)
        self._fit_input_n = 0
        self._fit_resampled_n = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MlpOversampledAdapter":
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        self._fit_input_n = int(y_arr.size)
        classes, counts = np.unique(y_arr, return_counts=True)
        if classes.size != 2:
            self._fit_resampled_n = int(y_arr.size)
            self._model.fit(X_arr, y_arr)
            return self

        majority = int(np.max(counts))
        rng = np.random.default_rng(self._seed)
        sampled_idx: list[np.ndarray] = []
        for cls in classes:
            cls_idx = np.flatnonzero(y_arr == cls)
            replace = cls_idx.size < majority
            picked = rng.choice(cls_idx, size=majority, replace=replace)
            sampled_idx.append(picked)
        train_idx = np.concatenate(sampled_idx)
        rng.shuffle(train_idx)
        self._fit_resampled_n = int(train_idx.size)
        self._model.fit(X_arr[train_idx], y_arr[train_idx])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)


class _MlpWeightedAdapter:
    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self._model = _make_mlp(seed)
        self._uses_sample_weight = False
        self._fit_input_n = 0
        self._fit_resampled_n = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MlpWeightedAdapter":
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        self._fit_input_n = int(y_arr.size)
        self._fit_resampled_n = int(y_arr.size)
        classes, counts = np.unique(y_arr, return_counts=True)
        if classes.size != 2:
            self._model.fit(X_arr, y_arr)
            return self

        count_map = {int(cls): int(count) for cls, count in zip(classes, counts, strict=False)}
        n_total = int(y_arr.size)
        class_weights = {
            int(cls): n_total / (2.0 * float(count_map[int(cls)]))
            for cls in classes
        }
        sample_weight = np.asarray([class_weights[int(label)] for label in y_arr], dtype=float)

        signature = inspect.signature(self._model.fit)
        if "sample_weight" in signature.parameters:
            self._model.fit(X_arr, y_arr, sample_weight=sample_weight)
            self._uses_sample_weight = True
            return self

        self._uses_sample_weight = False
        majority = int(np.max(counts))
        rng = np.random.default_rng(self._seed)
        sampled_idx: list[np.ndarray] = []
        for cls in classes:
            cls_idx = np.flatnonzero(y_arr == cls)
            replace = cls_idx.size < majority
            picked = rng.choice(cls_idx, size=majority, replace=replace)
            sampled_idx.append(picked)
        train_idx = np.concatenate(sampled_idx)
        rng.shuffle(train_idx)
        self._fit_resampled_n = int(train_idx.size)
        self._model.fit(X_arr[train_idx], y_arr[train_idx])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)


def _make_mlp_bce(seed: int) -> MLPClassifier:
    return _make_mlp(seed)


def _make_mlp_oversampled(seed: int) -> _MlpOversampledAdapter:
    return _MlpOversampledAdapter(seed)


def _make_mlp_weighted(seed: int) -> _MlpWeightedAdapter:
    return _MlpWeightedAdapter(seed)


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
    "mlp": _make_mlp,
    "mlp_bce": _make_mlp_bce,
    "mlp_oversampled": _make_mlp_oversampled,
    "mlp_weighted": _make_mlp_weighted,
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
