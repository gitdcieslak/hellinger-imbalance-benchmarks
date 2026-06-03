import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_run_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_weighted_dropout_sweep.py"
    spec = importlib.util.spec_from_file_location("run_weighted_dropout_sweep", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load run_weighted_dropout_sweep module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_report_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_weighted_dropout_sweep.py"
    spec = importlib.util.spec_from_file_location("report_weighted_dropout_sweep", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_weighted_dropout_sweep module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _df():
    rows = []
    for skew_ratio in [25, 100]:
        for rate in [0.0, 0.1, 0.3]:
            for seed in [0, 1]:
                rows.append(
                    {
                        "skew_ratio": skew_ratio,
                        "dropout_rate": rate,
                        "seed": seed,
                        "epoch": 100,
                        "breadth": 1.0 + rate,
                        "effective_breadth": 2.0 + rate,
                        "elevation": 0.8 - rate,
                        "peak_concentration": 0.8 - rate,
                        "minority_survival_auc": 0.5 + (0.2 if rate == 0.1 else 0.0),
                        "minority_survival_cliffiness": 0.6 - rate,
                        "minority_survival_max_drop": 0.2,
                        "minority_survival_effective_drop_count": 2.0,
                        "auroc": 0.7 + rate,
                        "average_precision": 0.1 + rate,
                    }
                )
    return pd.DataFrame(rows)


def test_dropout_rate_parsing():
    module = _load_run_module()

    assert module.parse_float_list("0.00,0.05,0.10") == [0.0, 0.05, 0.1]
    assert module.parse_int_list("25,100,500") == [25, 100, 500]


def test_output_row_construction_has_expected_columns():
    module = _load_run_module()
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.05, 0.2, 0.6, 0.9])

    row = module.build_output_row(100, 0.1, 3, 100, y, scores)

    assert set(row) == set(module.OUTPUT_COLUMNS)
    assert row["skew_ratio"] == 100
    assert row["dropout_rate"] == 0.1
    assert row["epoch"] == 100


def test_best_rate_identification():
    module = _load_report_module()
    best = module.best_dropout_rates(_df())

    auc_rate = float(best[best["criterion"] == "highest_mean_survival_auc"].iloc[0]["dropout_rate"])
    cliff_rate = float(best[best["criterion"] == "lowest_mean_cliffiness"].iloc[0]["dropout_rate"])

    assert auc_rate == 0.1
    assert cliff_rate == 0.3
    assert set(best["skew_ratio"]) == {25, 100}


def test_trend_correlations_by_skew():
    module = _load_report_module()
    corrs = module.tradeoff_correlations(_df())

    assert set(corrs["skew_ratio"]) == {25, 100}
    assert "dropout_rate vs breadth" in set(corrs["comparison"])


def test_weighted_dropout_report_generation():
    module = _load_report_module()
    report = module.build_weighted_dropout_sweep_report(_df())

    assert "## Dropout x Skew Means" in report
    assert "## Best Dropout By Skew" in report
    assert "## Trend Correlations By Skew" in report
    assert "## Stability Summary And Interpretation" in report
    assert "Does survival AUC increase monotonically" in report
