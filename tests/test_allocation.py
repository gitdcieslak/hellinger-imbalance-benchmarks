import numpy as np

from hib.allocation import allocation_concentration_metrics, gini_coefficient
from hib.reporting import summarize_allocation_records
from hib.runner import run_allocation_concentration_legacy


def test_entropy_concentrated_vs_spread():
    concentrated = allocation_concentration_metrics(np.array([0.0] * 90 + [0.01] * 10))
    spread = allocation_concentration_metrics(np.linspace(0.0, 1.0, 100))
    assert concentrated["histogram_entropy"] < spread["histogram_entropy"]


def test_effective_support_matches_exp_entropy():
    metrics = allocation_concentration_metrics(np.linspace(0.0, 1.0, 100))
    assert np.isclose(metrics["effective_support_size"], np.exp(metrics["histogram_entropy"]))


def test_gini_bounded_zero_to_one():
    gini = gini_coefficient(np.array([0.0, 0.1, 0.9, 1.0]))
    assert 0.0 <= gini <= 1.0


def test_fraction_below_thresholds_are_correct():
    scores = np.array([0.0, 0.0005, 0.003, 0.009, 0.03, 0.7])
    metrics = allocation_concentration_metrics(scores)
    assert np.isclose(metrics["fraction_scores_below_0_001"], 2 / 6)
    assert np.isclose(metrics["fraction_scores_below_0_005"], 3 / 6)
    assert np.isclose(metrics["fraction_scores_below_0_01"], 4 / 6)
    assert np.isclose(metrics["fraction_scores_below_0_05"], 5 / 6)
    assert np.isclose(metrics["fraction_scores_above_0_50"], 1 / 6)


def test_allocation_records_include_split_metadata_and_summary_runs(monkeypatch, tmp_path):
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

    records = run_allocation_concentration_legacy(
        dataset_ids=["tiny"],
        model_ids=["cart"],
        extracted_dir=extracted,
        n_repeats=1,
        test_size=0.5,
        split_seed=1,
    )
    assert len(records) == 1
    assert records[0]["dataset_id"] == "tiny"
    assert "repeat_id" in records[0] and "split_id" in records[0] and "split_seed" in records[0]

    summary = summarize_allocation_records(records)
    assert int(summary.loc[0, "n_runs"]) == 1
