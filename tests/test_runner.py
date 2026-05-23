import json

from hib.runner import run_synthetic_suite, write_jsonl


def test_smoke_runner_produces_jsonl_records(tmp_path):
    config = {
        "experiment_id": "test_synthetic_smoke",
        "skew_ratios": [1],
        "seeds": [0],
        "separation": 1.5,
        "minority_count": 8,
        "n_features": 4,
        "noise": 1.0,
        "test_size": 0.5,
        "n_repeats": 5,
        "split_seed": 10,
        "models": ["hddt", "bagged_hddt", "cart", "random_forest"],
    }
    records = run_synthetic_suite(config)
    output_path = write_jsonl(records, tmp_path / "synthetic_smoke.jsonl")

    lines = output_path.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]

    assert len(parsed) == 20
    assert {record["model_id"] for record in parsed} == set(config["models"])
    assert all(record["dataset_id"] == "synthetic-gaussian-skew-1-to-1" for record in parsed)
    assert all(record["skew_ratio"] == "1:1" for record in parsed)
    assert all("package_versions" in record for record in parsed)
    assert all("repeat_id" in record for record in parsed)
    assert all("split_id" in record for record in parsed)
    assert all("split_seed" in record for record in parsed)
    assert all("train_n" in record for record in parsed)
    assert all("test_n" in record for record in parsed)
    assert all("train_pos" in record for record in parsed)
    assert all("train_neg" in record for record in parsed)
    assert all("test_pos" in record for record in parsed)
    assert all("test_neg" in record for record in parsed)


def test_runner_executes_weighted_sklearn_models_end_to_end():
    config = {
        "experiment_id": "test_weighted_models",
        "skew_ratios": [10],
        "seeds": [0],
        "separation": 1.5,
        "minority_count": 10,
        "n_features": 4,
        "noise": 1.0,
        "test_size": 0.5,
        "n_repeats": 2,
        "split_seed": 3,
        "models": ["cart_balanced", "random_forest_balanced"],
    }

    records = run_synthetic_suite(config)

    assert len(records) == 4
    assert {record["model_id"] for record in records} == {"cart_balanced", "random_forest_balanced"}


def test_runner_executes_hddt_forest_end_to_end():
    config = {
        "experiment_id": "test_hddt_forest",
        "skew_ratios": [1],
        "seeds": [0],
        "separation": 1.5,
        "minority_count": 20,
        "n_features": 6,
        "noise": 1.0,
        "test_size": 0.5,
        "n_repeats": 2,
        "split_seed": 4,
        "models": ["hddt_forest"],
    }

    records = run_synthetic_suite(config)

    assert len(records) == 2
    assert all(record["model_id"] == "hddt_forest" for record in records)


def test_runner_executes_mlp_end_to_end():
    config = {
        "experiment_id": "test_mlp",
        "skew_ratios": [10],
        "seeds": [0],
        "separation": 1.5,
        "minority_count": 12,
        "n_features": 6,
        "noise": 1.0,
        "test_size": 0.5,
        "n_repeats": 1,
        "split_seed": 5,
        "models": ["mlp"],
    }

    records = run_synthetic_suite(config)

    assert len(records) == 1
    assert records[0]["model_id"] == "mlp"


def test_runner_executes_mlp_oversampled_end_to_end():
    config = {
        "experiment_id": "test_mlp_oversampled",
        "skew_ratios": [10],
        "seeds": [0],
        "separation": 1.5,
        "minority_count": 12,
        "n_features": 6,
        "noise": 1.0,
        "test_size": 0.5,
        "n_repeats": 1,
        "split_seed": 5,
        "models": ["mlp_oversampled"],
    }

    records = run_synthetic_suite(config)

    assert len(records) == 1
    assert records[0]["model_id"] == "mlp_oversampled"
