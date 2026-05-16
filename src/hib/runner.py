"""Runner for small synthetic benchmark suites."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hib.metrics import evaluate_model
from hib.metrics import positive_class_scores
from hib.models import OptionalDependencyUnavailable, available_model_ids, make_model
from hib.separation import score_separation_metrics
from hib.scores import class_conditional_score_summaries
from hib.splits import generate_stratified_repeated_splits
from hib.synthetic import SyntheticSkewConfig, make_gaussian_skew_dataset
from hib.thresholds import DEFAULT_THRESHOLDS, sweep_thresholds


DEFAULT_EXPERIMENT_CONFIG_PATH = Path("configs/experiments/synthetic_smoke.yaml")
DEFAULT_MODEL_CONFIG_PATH = Path("configs/models/synthetic_smoke.yaml")
DEFAULT_METRIC_CONFIG_PATH = Path("configs/metrics/core_imbalance.yaml")
KNOWN_MODEL_FAMILIES = {
    "HDDT",
    "Bagged HDDT",
    "HDDTForest",
    "CART",
    "RandomForest",
    "XGBoost",
    "LightGBM",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_id": "synthetic_smoke",
    "skew_ratios": [1, 10, 100],
    "seeds": [0],
    "separation": 2.0,
    "minority_count": 10,
    "n_features": 6,
    "noise": 1.0,
    "test_size": 0.5,
    "n_repeats": 5,
    "split_seed": 0,
    "models": ["hddt", "bagged_hddt", "cart", "random_forest"],
    "thresholds": DEFAULT_THRESHOLDS,
    "output_path": "results/synthetic_smoke.jsonl",
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML runner config or return defaults."""

    if path is None:
        return dict(DEFAULT_CONFIG)
    loaded = _read_yaml_mapping(path)
    return {**DEFAULT_CONFIG, **loaded}


def load_runner_config(
    experiment_path: str | Path | None = None,
    model_paths: list[str | Path] | None = None,
    metric_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load experiment, model, and metric configs for a runner invocation."""

    config = load_config(experiment_path or DEFAULT_EXPERIMENT_CONFIG_PATH)
    model_configs = load_model_configs(model_paths or [DEFAULT_MODEL_CONFIG_PATH])
    config["models"] = list(model_configs)
    config["model_params"] = model_configs
    config["metrics"] = load_metric_ids(metric_path or DEFAULT_METRIC_CONFIG_PATH)
    return config


def load_model_ids(paths: list[str | Path]) -> list[str]:
    """Load selected model ids from one or more model config files."""

    return list(load_model_configs(paths))


def load_model_configs(paths: list[str | Path]) -> dict[str, dict[str, Any]]:
    """Load and validate model configs keyed by model id."""

    model_configs: dict[str, dict[str, Any]] = {}
    known_model_ids = set(available_model_ids())
    for path in paths:
        config = _read_yaml_mapping(path)
        models = config.get("models")
        if not isinstance(models, dict):
            raise ValueError(f"model config {path!s} must contain a 'models' mapping")
        for model_id, model_config in models.items():
            if model_id in model_configs:
                raise ValueError(f"duplicate model id {model_id!r} in model configs")
            if model_id not in known_model_ids:
                raise ValueError(f"unknown model id {model_id!r} in model config {path!s}")
            if not isinstance(model_config, dict):
                raise ValueError(f"model {model_id!r} config must be a mapping")
            family = model_config.get("family")
            if family not in KNOWN_MODEL_FAMILIES:
                known = ", ".join(sorted(KNOWN_MODEL_FAMILIES))
                raise ValueError(
                    f"unknown model family {family!r} for model {model_id!r}; "
                    f"expected one of: {known}"
                )
            model_configs[model_id] = dict(model_config)
    return model_configs


def load_metric_ids(path: str | Path) -> list[str]:
    """Load selected metric ids from a metric config file."""

    config = _read_yaml_mapping(path)
    metrics = config.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError(f"metric config {path!s} must contain a 'metrics' list")
    return [str(metric) for metric in metrics]


def _read_yaml_mapping(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"config path does not exist: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config {yaml_path!s} must be a YAML mapping")
    return loaded


def package_versions() -> dict[str, str | None]:
    """Capture package versions required for result records."""

    packages = ["numpy", "scikit-learn", "hellinger-tree"]
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def compute_binary_class_weight_ratio(y_train: np.ndarray, positive_label: int = 1) -> float:
    """Compute n_negative / n_positive for a binary training split."""

    y = np.asarray(y_train)
    positive_count = int(np.sum(y == positive_label))
    negative_count = int(np.sum(y != positive_label))
    if positive_count == 0:
        raise ValueError("training split has zero positive examples; cannot compute class ratio")
    if negative_count == 0:
        raise ValueError("training split has zero negative examples; cannot compute class ratio")
    return float(negative_count) / float(positive_count)


def apply_split_dependent_model_params(model: Any, model_id: str, y_train: np.ndarray) -> Any:
    """Set split-dependent model parameters before fitting."""

    if model_id in {"xgboost_weighted", "lightgbm_weighted"}:
        ratio = compute_binary_class_weight_ratio(y_train)
        model.set_params(scale_pos_weight=ratio)
    return model


def resolve_hddt_forest_max_features(
    max_features_spec: Any,
    n_features: int,
) -> int | float:
    """Resolve hddt_forest max_features spec to sklearn-compatible value."""

    if max_features_spec is None:
        return 1.0
    if isinstance(max_features_spec, str):
        key = max_features_spec.strip().lower()
        if key == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if key == "log2":
            return max(1, int(np.log2(max(2, n_features))))
        if key == "null":
            return 1.0
        raise ValueError(f"unsupported max_features spec {max_features_spec!r}")
    if isinstance(max_features_spec, (int, np.integer)):
        return max(1, min(int(max_features_spec), n_features))
    if isinstance(max_features_spec, (float, np.floating)):
        value = float(max_features_spec)
        if value <= 0:
            raise ValueError("max_features float must be > 0")
        return value
    raise ValueError(f"unsupported max_features type: {type(max_features_spec)!r}")


def apply_split_dependent_ensemble_params(model: Any, model_id: str, X_train: np.ndarray) -> Any:
    """Set split-dependent ensemble params before fit."""

    if model_id not in {"hddt_forest", "random_subspace_hddt"}:
        return model
    spec = getattr(model, "_hib_max_features_spec", "sqrt")
    resolved = resolve_hddt_forest_max_features(spec, int(X_train.shape[1]))
    model.set_params(max_features=resolved)
    return model


def apply_model_config_params(
    model: Any,
    model_id: str,
    model_params: dict[str, dict[str, Any]] | None,
) -> Any:
    """Apply selected config-level params to model instances."""

    if not model_params:
        return model
    params = model_params.get(model_id, {})
    if model_id in {"hddt_forest", "random_subspace_hddt"}:
        if "max_features" in params:
            setattr(model, "_hib_max_features_spec", params["max_features"])
        if "n_estimators" in params:
            model.set_params(n_estimators=int(params["n_estimators"]))
        if "bootstrap" in params:
            model.set_params(bootstrap=bool(params["bootstrap"]))
    return model


def fit_or_skip_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> bool:
    """Fit a model and return False for expected single-class bootstrap failures."""

    try:
        model.fit(X_train, y_train)
    except ValueError as exc:
        message = str(exc)
        if "supports binary classification only" in message:
            return False
        raise
    return True


def run_synthetic_suite(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the configured synthetic benchmark suite and return result records."""

    model_ids = list(config["models"])
    unknown_models = sorted(set(model_ids) - set(available_model_ids()))
    if unknown_models:
        raise ValueError(f"unknown model ids: {', '.join(unknown_models)}")

    records: list[dict[str, Any]] = []
    versions = package_versions()
    n_repeats = int(config.get("n_repeats", 5))
    split_seed = int(config.get("split_seed", 0))
    model_params = config.get("model_params")

    for skew_ratio in config["skew_ratios"]:
        for seed in config["seeds"]:
            dataset_config = SyntheticSkewConfig(
                skew_ratio=int(skew_ratio),
                seed=int(seed),
                separation=float(config["separation"]),
                minority_count=int(config["minority_count"]),
                n_features=int(config["n_features"]),
                noise=float(config["noise"]),
                test_size=float(config["test_size"]),
            )
            X, y = make_gaussian_skew_dataset(dataset_config)
            split_specs = generate_stratified_repeated_splits(
                y,
                n_repeats=n_repeats,
                test_size=float(config["test_size"]),
                random_seed=split_seed,
            )

            for split_spec in split_specs:
                X_train = X[split_spec.train_idx]
                y_train = y[split_spec.train_idx]
                X_test = X[split_spec.test_idx]
                y_test = y[split_spec.test_idx]

                for model_id in model_ids:
                    try:
                        model = make_model(model_id, seed=int(seed))
                    except OptionalDependencyUnavailable:
                        continue
                    model = apply_model_config_params(model, model_id, model_params)
                    model = apply_split_dependent_ensemble_params(model, model_id, X_train)
                    model = apply_split_dependent_model_params(model, model_id, y_train)
                    if not fit_or_skip_model(model, X_train, y_train):
                        continue
                    metrics = evaluate_model(model, X_test, y_test)
                    train_pos = int(np.sum(y_train == 1))
                    train_n = int(y_train.size)
                    test_pos = int(np.sum(y_test == 1))
                    test_n = int(y_test.size)
                    records.append(
                        {
                            "experiment_id": config["experiment_id"],
                            "dataset_id": dataset_config.dataset_id,
                            "model_id": model_id,
                            "seed": int(seed),
                            "repeat_id": split_spec.repeat_id,
                            "split_id": split_spec.split_id,
                            "split_seed": split_spec.split_seed,
                            "skew_ratio": f"{int(skew_ratio)}:1",
                            "synthetic_config": asdict(dataset_config),
                            "train_n": train_n,
                            "test_n": test_n,
                            "train_pos": train_pos,
                            "train_neg": train_n - train_pos,
                            "test_pos": test_pos,
                            "test_neg": test_n - test_pos,
                            "metrics": metrics,
                            "package_versions": versions,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
    return records


def run_threshold_sweep_suite(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run threshold sweeps over configured synthetic datasets and models."""

    model_ids = list(config["models"])
    unknown_models = sorted(set(model_ids) - set(available_model_ids()))
    if unknown_models:
        raise ValueError(f"unknown model ids: {', '.join(unknown_models)}")

    records: list[dict[str, Any]] = []
    versions = package_versions()
    n_repeats = int(config.get("n_repeats", 5))
    split_seed = int(config.get("split_seed", 0))
    thresholds = [float(threshold) for threshold in config.get("thresholds", DEFAULT_THRESHOLDS)]
    model_params = config.get("model_params")

    for skew_ratio in config["skew_ratios"]:
        for seed in config["seeds"]:
            dataset_config = SyntheticSkewConfig(
                skew_ratio=int(skew_ratio),
                seed=int(seed),
                separation=float(config["separation"]),
                minority_count=int(config["minority_count"]),
                n_features=int(config["n_features"]),
                noise=float(config["noise"]),
                test_size=float(config["test_size"]),
            )
            X, y = make_gaussian_skew_dataset(dataset_config)
            split_specs = generate_stratified_repeated_splits(
                y,
                n_repeats=n_repeats,
                test_size=float(config["test_size"]),
                random_seed=split_seed,
            )

            for split_spec in split_specs:
                X_train = X[split_spec.train_idx]
                y_train = y[split_spec.train_idx]
                X_test = X[split_spec.test_idx]
                y_test = y[split_spec.test_idx]

                for model_id in model_ids:
                    try:
                        model = make_model(model_id, seed=int(seed))
                    except OptionalDependencyUnavailable:
                        continue
                    model = apply_model_config_params(model, model_id, model_params)
                    model = apply_split_dependent_ensemble_params(model, model_id, X_train)
                    model = apply_split_dependent_model_params(model, model_id, y_train)
                    if not fit_or_skip_model(model, X_train, y_train):
                        continue
                    y_score = positive_class_scores(model, X_test)
                    train_pos = int(np.sum(y_train == 1))
                    train_n = int(y_train.size)
                    test_pos = int(np.sum(y_test == 1))
                    test_n = int(y_test.size)
                    for threshold_result in sweep_thresholds(y_test, y_score, thresholds):
                        records.append(
                            {
                                "experiment_id": f"{config['experiment_id']}_threshold_sweep",
                                "dataset_id": dataset_config.dataset_id,
                                "model_id": model_id,
                                "seed": int(seed),
                                "repeat_id": split_spec.repeat_id,
                                "split_id": split_spec.split_id,
                                "split_seed": split_spec.split_seed,
                                "skew_ratio": f"{int(skew_ratio)}:1",
                                "threshold": threshold_result["threshold"],
                                "synthetic_config": asdict(dataset_config),
                                "train_n": train_n,
                                "test_n": test_n,
                                "train_pos": train_pos,
                                "train_neg": train_n - train_pos,
                                "test_pos": test_pos,
                                "test_neg": test_n - test_pos,
                                "metrics": threshold_result["metrics"],
                                "package_versions": versions,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
    return records


def write_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write structured result records to JSONL."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def run_from_config(
    path: str | Path | None = None,
    model_paths: list[str | Path] | None = None,
    metric_path: str | Path | None = None,
) -> Path:
    """Load config, run the synthetic suite, and write JSONL results."""

    config = load_runner_config(path, model_paths, metric_path)
    records = run_synthetic_suite(config)
    return write_jsonl(records, config["output_path"])


def run_threshold_sweep_from_config(
    path: str | Path | None = None,
    model_paths: list[str | Path] | None = None,
    metric_path: str | Path | None = None,
    output_path: str | Path = "results/synthetic_threshold_sweep.jsonl",
) -> Path:
    """Load config, run threshold sweeps, and write JSONL results."""

    config = load_runner_config(path, model_paths, metric_path)
    config["output_path"] = str(output_path)
    records = run_threshold_sweep_suite(config)
    return write_jsonl(records, config["output_path"])


def run_score_analysis_suite(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run class-conditional score distribution analysis for synthetic data."""

    model_ids = list(config["models"])
    unknown_models = sorted(set(model_ids) - set(available_model_ids()))
    if unknown_models:
        raise ValueError(f"unknown model ids: {', '.join(unknown_models)}")

    records: list[dict[str, Any]] = []
    versions = package_versions()
    n_repeats = int(config.get("n_repeats", 5))
    split_seed = int(config.get("split_seed", 0))
    model_params = config.get("model_params")

    for skew_ratio in config["skew_ratios"]:
        for seed in config["seeds"]:
            dataset_config = SyntheticSkewConfig(
                skew_ratio=int(skew_ratio),
                seed=int(seed),
                separation=float(config["separation"]),
                minority_count=int(config["minority_count"]),
                n_features=int(config["n_features"]),
                noise=float(config["noise"]),
                test_size=float(config["test_size"]),
            )
            X, y = make_gaussian_skew_dataset(dataset_config)
            split_specs = generate_stratified_repeated_splits(
                y,
                n_repeats=n_repeats,
                test_size=float(config["test_size"]),
                random_seed=split_seed,
            )

            for split_spec in split_specs:
                X_train = X[split_spec.train_idx]
                y_train = y[split_spec.train_idx]
                X_test = X[split_spec.test_idx]
                y_test = y[split_spec.test_idx]
                train_pos = int(np.sum(y_train == 1))
                train_n = int(y_train.size)
                test_pos = int(np.sum(y_test == 1))
                test_n = int(y_test.size)

                for model_id in model_ids:
                    try:
                        model = make_model(model_id, seed=int(seed))
                    except OptionalDependencyUnavailable:
                        continue
                    model = apply_model_config_params(model, model_id, model_params)
                    model = apply_split_dependent_ensemble_params(model, model_id, X_train)
                    model = apply_split_dependent_model_params(model, model_id, y_train)
                    if not fit_or_skip_model(model, X_train, y_train):
                        continue
                    y_score = positive_class_scores(model, X_test)
                    class_summaries = class_conditional_score_summaries(y_test, y_score)
                    for summary in class_summaries:
                        histogram = summary.pop("histogram")
                        records.append(
                            {
                                "experiment_id": f"{config['experiment_id']}_score_analysis",
                                "dataset_id": dataset_config.dataset_id,
                                "model_id": model_id,
                                "seed": int(seed),
                                "repeat_id": split_spec.repeat_id,
                                "split_id": split_spec.split_id,
                                "split_seed": split_spec.split_seed,
                                "skew_ratio": f"{int(skew_ratio)}:1",
                                "class_label": summary.pop("class_label"),
                                "synthetic_config": asdict(dataset_config),
                                "train_n": train_n,
                                "test_n": test_n,
                                "train_pos": train_pos,
                                "train_neg": train_n - train_pos,
                                "test_pos": test_pos,
                                "test_neg": test_n - test_pos,
                                "score_summary": summary,
                                "histogram": histogram,
                                "package_versions": versions,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
    return records


def run_score_analysis_from_config(
    path: str | Path | None = None,
    model_paths: list[str | Path] | None = None,
    metric_path: str | Path | None = None,
    output_path: str | Path = "results/synthetic_score_analysis.jsonl",
) -> Path:
    """Load config, run score analysis, and write JSONL results."""

    config = load_runner_config(path, model_paths, metric_path)
    config["output_path"] = str(output_path)
    records = run_score_analysis_suite(config)
    return write_jsonl(records, config["output_path"])


def run_score_separation_suite(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run score separation analysis for synthetic data."""

    model_ids = list(config["models"])
    unknown_models = sorted(set(model_ids) - set(available_model_ids()))
    if unknown_models:
        raise ValueError(f"unknown model ids: {', '.join(unknown_models)}")

    records: list[dict[str, Any]] = []
    versions = package_versions()
    n_repeats = int(config.get("n_repeats", 5))
    split_seed = int(config.get("split_seed", 0))
    model_params = config.get("model_params")

    for skew_ratio in config["skew_ratios"]:
        for seed in config["seeds"]:
            dataset_config = SyntheticSkewConfig(
                skew_ratio=int(skew_ratio),
                seed=int(seed),
                separation=float(config["separation"]),
                minority_count=int(config["minority_count"]),
                n_features=int(config["n_features"]),
                noise=float(config["noise"]),
                test_size=float(config["test_size"]),
            )
            X, y = make_gaussian_skew_dataset(dataset_config)
            split_specs = generate_stratified_repeated_splits(
                y,
                n_repeats=n_repeats,
                test_size=float(config["test_size"]),
                random_seed=split_seed,
            )

            for split_spec in split_specs:
                X_train = X[split_spec.train_idx]
                y_train = y[split_spec.train_idx]
                X_test = X[split_spec.test_idx]
                y_test = y[split_spec.test_idx]
                train_pos = int(np.sum(y_train == 1))
                train_n = int(y_train.size)
                test_pos = int(np.sum(y_test == 1))
                test_n = int(y_test.size)

                for model_id in model_ids:
                    try:
                        model = make_model(model_id, seed=int(seed))
                    except OptionalDependencyUnavailable:
                        continue
                    model = apply_model_config_params(model, model_id, model_params)
                    model = apply_split_dependent_ensemble_params(model, model_id, X_train)
                    model = apply_split_dependent_model_params(model, model_id, y_train)
                    if not fit_or_skip_model(model, X_train, y_train):
                        continue
                    y_score = positive_class_scores(model, X_test)
                    positive_scores = y_score[y_test == 1]
                    negative_scores = y_score[y_test == 0]
                    if positive_scores.size == 0 or negative_scores.size == 0:
                        continue
                    metrics = score_separation_metrics(positive_scores, negative_scores)
                    records.append(
                        {
                            "experiment_id": f"{config['experiment_id']}_score_separation",
                            "dataset_id": dataset_config.dataset_id,
                            "model_id": model_id,
                            "seed": int(seed),
                            "repeat_id": split_spec.repeat_id,
                            "split_id": split_spec.split_id,
                            "split_seed": split_spec.split_seed,
                            "skew_ratio": f"{int(skew_ratio)}:1",
                            "synthetic_config": asdict(dataset_config),
                            "train_n": train_n,
                            "test_n": test_n,
                            "train_pos": train_pos,
                            "train_neg": train_n - train_pos,
                            "test_pos": test_pos,
                            "test_neg": test_n - test_pos,
                            "separation_metrics": metrics,
                            "package_versions": versions,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
    return records


def run_score_separation_from_config(
    path: str | Path | None = None,
    model_paths: list[str | Path] | None = None,
    metric_path: str | Path | None = None,
    output_path: str | Path = "results/synthetic_score_separation.jsonl",
) -> Path:
    """Load config, run score separation, and write JSONL results."""

    config = load_runner_config(path, model_paths, metric_path)
    config["output_path"] = str(output_path)
    records = run_score_separation_suite(config)
    return write_jsonl(records, config["output_path"])


def collect_score_distribution_data(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect class-conditional score arrays for plotting (in-memory only)."""

    model_ids = list(config["models"])
    unknown_models = sorted(set(model_ids) - set(available_model_ids()))
    if unknown_models:
        raise ValueError(f"unknown model ids: {', '.join(unknown_models)}")

    n_repeats = int(config.get("n_repeats", 5))
    split_seed = int(config.get("split_seed", 0))
    model_params = config.get("model_params")
    collected: dict[tuple[str, str], dict[str, list[float]]] = {}

    for skew_ratio in config["skew_ratios"]:
        skew_label = f"{int(skew_ratio)}:1"
        for seed in config["seeds"]:
            dataset_config = SyntheticSkewConfig(
                skew_ratio=int(skew_ratio),
                seed=int(seed),
                separation=float(config["separation"]),
                minority_count=int(config["minority_count"]),
                n_features=int(config["n_features"]),
                noise=float(config["noise"]),
                test_size=float(config["test_size"]),
            )
            X, y = make_gaussian_skew_dataset(dataset_config)
            split_specs = generate_stratified_repeated_splits(
                y,
                n_repeats=n_repeats,
                test_size=float(config["test_size"]),
                random_seed=split_seed,
            )

            for split_spec in split_specs:
                X_train = X[split_spec.train_idx]
                y_train = y[split_spec.train_idx]
                X_test = X[split_spec.test_idx]
                y_test = y[split_spec.test_idx]

                for model_id in model_ids:
                    try:
                        model = make_model(model_id, seed=int(seed))
                    except OptionalDependencyUnavailable:
                        continue
                    model = apply_model_config_params(model, model_id, model_params)
                    model = apply_split_dependent_ensemble_params(model, model_id, X_train)
                    model = apply_split_dependent_model_params(model, model_id, y_train)
                    if not fit_or_skip_model(model, X_train, y_train):
                        continue
                    y_score = positive_class_scores(model, X_test)
                    key = (skew_label, model_id)
                    if key not in collected:
                        collected[key] = {"positive_scores": [], "negative_scores": []}
                    collected[key]["positive_scores"].extend(
                        np.asarray(y_score[y_test == 1], dtype=float).tolist()
                    )
                    collected[key]["negative_scores"].extend(
                        np.asarray(y_score[y_test == 0], dtype=float).tolist()
                    )

    output: list[dict[str, Any]] = []
    for (skew_ratio, model_id), scores in sorted(collected.items()):
        output.append(
            {
                "skew_ratio": skew_ratio,
                "model_id": model_id,
                "positive_scores": np.asarray(scores["positive_scores"], dtype=float),
                "negative_scores": np.asarray(scores["negative_scores"], dtype=float),
            }
        )
    return output
