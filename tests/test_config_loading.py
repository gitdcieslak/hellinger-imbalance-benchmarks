import json

import pytest

from hib import runner
from hib.runner import load_model_ids, run_from_config


def test_model_config_override_works(tmp_path):
    model_config = tmp_path / "models.yaml"
    model_config.write_text(
        "models:\n"
        "  cart:\n"
        "    family: CART\n"
        "    package: scikit-learn\n",
        encoding="utf-8",
    )

    assert load_model_ids([model_config]) == ["cart"]


def test_boosted_models_appear_in_emitted_jsonl(monkeypatch, tmp_path):
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        "experiment_id: test_boosted\n"
        "skew_ratios: [1]\n"
        "seeds: [0]\n"
        "separation: 1.5\n"
        "minority_count: 12\n"
        "n_features: 4\n"
        "noise: 1.0\n"
        "test_size: 0.25\n"
        f"output_path: {tmp_path / 'boosted.jsonl'}\n",
        encoding="utf-8",
    )
    model_config = tmp_path / "boosted.yaml"
    model_config.write_text(
        "models:\n"
        "  xgboost:\n"
        "    family: XGBoost\n"
        "    package: xgboost\n"
        "    optional: true\n"
        "  lightgbm:\n"
        "    family: LightGBM\n"
        "    package: lightgbm\n"
        "    optional: true\n",
        encoding="utf-8",
    )
    metric_config = tmp_path / "metrics.yaml"
    metric_config.write_text("metrics: [auroc]\n", encoding="utf-8")

    def fake_make_model(model_id, seed):
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(random_state=seed)

    monkeypatch.setattr(runner, "make_model", fake_make_model)
    output_path = run_from_config(experiment_config, [model_config], metric_config)
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert {record["model_id"] for record in records} == {"xgboost", "lightgbm"}


def test_duplicate_model_ids_raise_error(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    content = (
        "models:\n"
        "  cart:\n"
        "    family: CART\n"
        "    package: scikit-learn\n"
    )
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate model id 'cart'"):
        load_model_ids([first, second])


def test_missing_config_path_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="config path does not exist"):
        load_model_ids([tmp_path / "missing.yaml"])


def test_unknown_model_family_raises_error(tmp_path):
    model_config = tmp_path / "models.yaml"
    model_config.write_text(
        "models:\n"
        "  cart:\n"
        "    family: MysteryForest\n"
        "    package: scikit-learn\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown model family 'MysteryForest'"):
        load_model_ids([model_config])


def test_no_duplicate_model_ids_when_merging_smoke_and_class_weighted_configs():
    model_ids = load_model_ids(
        [
            "configs/models/synthetic_smoke.yaml",
            "configs/models/class_weighted_trees.yaml",
        ]
    )
    assert len(model_ids) == len(set(model_ids))


def test_hddt_ensembles_config_loading_works():
    model_ids = load_model_ids(["configs/models/hddt_ensembles.yaml"])
    assert model_ids == ["hddt_forest"]


def test_neural_mlp_model_config_loading_works():
    model_ids = load_model_ids(["configs/models/neural_mlp_allocation_geometry.yaml"])
    assert "mlp" in model_ids


def test_neural_mlp_objective_perturbation_model_config_loading_works():
    model_ids = load_model_ids(["configs/models/neural_mlp_objective_perturbation.yaml"])
    assert {"mlp_bce", "mlp_oversampled", "mlp_weighted"}.issubset(set(model_ids))
