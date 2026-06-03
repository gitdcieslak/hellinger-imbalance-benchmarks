import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_run_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_weighted_oversampling_grid.py"
    spec = importlib.util.spec_from_file_location("run_weighted_oversampling_grid", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load run_weighted_oversampling_grid module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_report_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_weighted_oversampling_grid.py"
    spec = importlib.util.spec_from_file_location("report_weighted_oversampling_grid", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_weighted_oversampling_grid module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _df():
    rows = []
    for weight in [1, 2, 5, 10]:
        for oversampling in [1, 2, 5, 10]:
            for seed in [0, 1]:
                rows.append(
                    {
                        "class_weight_multiplier": weight,
                        "oversampling_multiplier": oversampling,
                        "seed": seed,
                        "epoch": 100,
                        "breadth": 1.0 + 0.01 * oversampling - 0.005 * weight,
                        "effective_breadth": 2.0 + 0.01 * oversampling,
                        "elevation": 0.2 + 0.01 * weight,
                        "peak_concentration": 0.3 + 0.01 * weight,
                        "minority_survival_auc": 0.4 + 0.01 * weight + 0.005 * oversampling,
                        "minority_survival_cliffiness": 0.7 - 0.01 * oversampling,
                        "minority_survival_max_drop": 0.2,
                        "minority_survival_effective_drop_count": 2.0,
                        "auroc": 0.6 + 0.002 * weight,
                        "average_precision": 0.2 + 0.003 * oversampling,
                    }
                )
    return pd.DataFrame(rows)


def test_grid_argument_parsing():
    module = _load_run_module()

    assert module.parse_int_list("1,2,5,10") == [1, 2, 5, 10]


def test_output_row_construction_has_expected_columns():
    module = _load_run_module()
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.05, 0.2, 0.6, 0.9])

    row = module.build_output_row(2, 5, 3, 100, y, scores)

    assert set(row) == set(module.OUTPUT_COLUMNS)
    assert row["class_weight_multiplier"] == 2
    assert row["oversampling_multiplier"] == 5
    assert row["epoch"] == 100


def test_best_cells_identify_requested_metrics():
    module = _load_report_module()
    best = module.best_cells(_df())

    assert set(best["criterion"]) == {
        "best_survival_auc",
        "lowest_cliffiness",
        "highest_breadth",
        "highest_elevation",
        "best_auroc",
        "best_ap",
    }
    assert int(best[best["criterion"] == "best_survival_auc"].iloc[0]["class_weight_multiplier"]) == 10
    assert int(best[best["criterion"] == "lowest_cliffiness"].iloc[0]["oversampling_multiplier"]) == 10


def test_pareto_frontier_contains_nondominated_cells():
    module = _load_report_module()
    frontier = module.pareto_frontier(_df())

    assert not frontier.empty
    assert {"breadth", "elevation"}.issubset(frontier.columns)


def test_weighted_oversampling_report_generation():
    module = _load_report_module()
    report = module.build_weighted_oversampling_grid_report(_df())

    assert "## Grid Cell Means" in report
    assert "## Best Cells" in report
    assert "## Breadth/Elevation Pareto Frontier" in report
    assert "Do weighting and oversampling move in different morphology directions?" in report
    assert "complementary" in report or "redundant" in report
