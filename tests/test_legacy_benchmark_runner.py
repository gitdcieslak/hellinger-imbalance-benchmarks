import json

import numpy as np

from hib.datasets.legacy_loader import load_legacy_hddt_dataset
from hib.datasets.legacy_registry import LEGACY_HDDT_DATASET_REGISTRY
from hib.runner import run_legacy_hddt_benchmark, summarize_legacy_records, write_jsonl


def test_registry_contains_expected_dataset_ids():
    expected = {
        "breast-w",
        "pima",
        "sonar",
        "ion",
        "mushroom",
        "oil",
        "satimage",
        "sick",
        "boundary",
        "compustat",
        "phoneme",
        "page",
        "cam",
        "covtype",
    }
    assert expected.issubset(set(LEGACY_HDDT_DATASET_REGISTRY))


def test_loader_loads_data_and_maps_positive_class(tmp_path):
    extracted = tmp_path / "extracted"
    path = extracted / "sample.data"
    path.parent.mkdir(parents=True)
    path.write_text("1,red,yes\n2,blue,no\n3,red,yes\n", encoding="utf-8")
    entry = {
        "dataset_id": "sample",
        "relative_path": "sample.data",
        "target_column": "2",
        "positive_class": "yes",
        "source_group": "balanced",
    }
    X, y, metadata = load_legacy_hddt_dataset(extracted, entry)

    assert X.shape == (3, 2)
    assert y.tolist() == [1, 0, 1]
    assert metadata["target_column"] == "2"


def test_loader_categorical_encoding_is_deterministic(tmp_path):
    extracted = tmp_path / "extracted"
    path = extracted / "sample.data"
    path.parent.mkdir(parents=True)
    path.write_text("1,z,yes\n2,a,no\n3,z,yes\n", encoding="utf-8")
    entry = {
        "dataset_id": "sample",
        "relative_path": "sample.data",
        "target_column": "2",
        "positive_class": "yes",
        "source_group": "balanced",
    }
    X1, _, _ = load_legacy_hddt_dataset(extracted, entry)
    X2, _, _ = load_legacy_hddt_dataset(extracted, entry)
    assert np.array_equal(X1, X2)


def test_loader_handles_missing_values(tmp_path):
    extracted = tmp_path / "extracted"
    path = extracted / "sample.data"
    path.parent.mkdir(parents=True)
    path.write_text("1,?,yes\n,blue,no\n3,,yes\n", encoding="utf-8")
    entry = {
        "dataset_id": "sample",
        "relative_path": "sample.data",
        "target_column": "2",
        "positive_class": "yes",
        "source_group": "balanced",
    }
    X, _, _ = load_legacy_hddt_dataset(extracted, entry)
    assert np.isfinite(X).all()


def test_loader_fails_when_positive_class_missing(tmp_path):
    extracted = tmp_path / "extracted"
    path = extracted / "sample.data"
    path.parent.mkdir(parents=True)
    path.write_text("1,a,no\n2,b,no\n", encoding="utf-8")
    entry = {
        "dataset_id": "sample",
        "relative_path": "sample.data",
        "target_column": "2",
        "positive_class": "yes",
        "source_group": "balanced",
    }
    try:
        load_legacy_hddt_dataset(extracted, entry)
    except ValueError as exc:
        assert "positive class" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing positive class")


def test_legacy_runner_executes_tiny_fixture_end_to_end(tmp_path, monkeypatch):
    extracted = tmp_path / "extracted"
    path = extracted / "tiny.data"
    path.parent.mkdir(parents=True)
    path.write_text("1,a,yes\n2,b,no\n3,a,yes\n4,b,no\n", encoding="utf-8")

    tiny_registry = {
        "tiny": {
            "dataset_id": "tiny",
            "relative_path": "tiny.data",
            "target_column": "2",
            "positive_class": "yes",
            "task_type": "binary",
            "source_group": "balanced",
            "n_rows": 4,
            "n_columns": 3,
            "imbalance_ratio": 1.0,
            "notes": "",
        }
    }
    monkeypatch.setattr("hib.runner.CURATED_LEGACY_HDDT_DATASET_REGISTRY", tiny_registry)

    records = run_legacy_hddt_benchmark(
        dataset_ids=["tiny"],
        model_ids=["cart"],
        extracted_dir=extracted,
        n_repeats=1,
        test_size=0.5,
        split_seed=1,
        seed=1,
    )
    assert len(records) == 1
    assert records[0]["dataset_id"] == "tiny"
    assert "metrics" in records[0]

    jsonl_path = write_jsonl(records, tmp_path / "legacy.jsonl")
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["model_id"] == "cart"

    summary = summarize_legacy_records(records)
    assert not summary.empty
