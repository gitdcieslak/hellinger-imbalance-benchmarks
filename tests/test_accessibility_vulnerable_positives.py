import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_vulnerable_positives.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_vulnerable_positives", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _scores():
    rows = []
    for task_id in ["d:1", "d:2"]:
        for seed in [0, 1, 2]:
            for model in ["cart", "rf"]:
                run_id = f"{task_id}|{model}|seed={seed}|split={seed}"
                for example_id, score in [(1, 0.1), (2, 0.1), (3, 0.55), (4, 0.85), (5, 1.0)]:
                    rows.append(
                        {
                            "run_id": run_id,
                            "dataset_id": "d",
                            "task_id": task_id,
                            "model_id": model,
                            "seed": seed,
                            "split_id": seed,
                            "example_id": example_id,
                            "positive_label": 1,
                            "score": score,
                            "positive_key": f"d|{task_id}|{example_id}",
                            "score_round": round(score, 6),
                            "minority_survival_auc": 0.5,
                            "minority_survival_cliffiness": 0.8 if task_id == "d:1" else 0.2,
                            "breadth": 0.2 if task_id == "d:1" else 0.8,
                            "elevation": 0.9 if task_id == "d:1" else 0.4,
                            "regime_id": 1,
                        }
                    )
    return pd.DataFrame(rows)


def test_construct_positive_table_adds_rank_and_driver_status():
    module = _load_module()

    table = module.construct_positive_table(_scores())

    assert {"posterior_rank", "posterior_rank_percentile", "cliff_driver", "drop_fraction"}.issubset(table.columns)
    assert table["posterior_rank_percentile"].between(0, 1).all()
    assert set(table[table["cliff_driver"]]["example_id"]) == {1, 2}
    assert table[table["example_id"] == 5]["cliff_driver"].sum() == 0


def test_driver_frequency_identifies_persistent_drivers():
    module = _load_module()
    table = module.construct_positive_table(_scores())

    freq = module.driver_frequency(table)
    summary = module.driver_frequency_summary(freq)

    assert freq[freq["positive_id"].isin([1, 2])]["driver_frequency"].min() == 1.0
    assert freq[freq["positive_id"] == 5]["driver_frequency"].max() == 0.0
    assert summary["persistent_driver_fraction"].iloc[0] > 0


def test_rank_controlled_rates_and_bin_variability_nonempty():
    module = _load_module()
    table = module.construct_positive_table(_scores())


    rates = module.rank_controlled_rates(table)
    variability = module.bin_level_vulnerability(table)

    assert not rates.empty
    assert {"rank_decile", "driver_rate"}.issubset(rates.columns)
    assert not variability.empty
    assert "high_frequency_positive_fraction" in variability.columns


def test_predictive_models_returns_rank_and_morphology_scores():
    module = _load_module()
    table = module.construct_positive_table(_scores())


    scores, calibration = module.predictive_models(table, max_rows=None)

    assert set(scores["model"]) == {"rank_only", "rank_plus_morphology"}
    assert scores["auroc"].between(0, 1).all()
    assert scores["log_loss"].notna().all()
    assert not calibration.empty


def test_write_reports_creates_expected_files(tmp_path):
    module = _load_module()
    scores_path = tmp_path / "scores.csv"
    _scores().to_csv(scores_path, index=False)

    outputs = module.write_reports(tmp_path, scores_path, max_model_rows=None)

    names = {p.name for p in outputs}
    assert "accessibility_vulnerable_positive_analysis.md" in names
    assert "accessibility_vulnerable_positive_table.csv" in names
    assert "accessibility_vulnerable_model_scores.csv" in names
    assert all(p.exists() for p in outputs)
