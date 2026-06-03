import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_run_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_allocation_trajectory.py"
    spec = importlib.util.spec_from_file_location("run_allocation_trajectory", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load run_allocation_trajectory module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_report_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_allocation_trajectory.py"
    spec = importlib.util.spec_from_file_location("report_allocation_trajectory", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_allocation_trajectory module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_checkpoint_row_has_requested_columns():
    module = _load_run_module()
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.05, 0.2, 0.6, 0.9])

    row = module.build_checkpoint_row("mlp_bce", 0, 5, y, scores)

    assert set(row) == set(module.OUTPUT_COLUMNS)
    assert row["objective"] == "mlp_bce"
    assert row["epoch"] == 5
    assert row["breadth"] >= 0.0
    assert row["elevation"] >= 0.0


def test_allocation_trajectory_report_contains_analysis_answers():
    module = _load_report_module()
    rows = []
    for objective, offset in [("mlp_bce", 0.0), ("mlp_weighted_bce", 0.2)]:
        for seed in [0, 1]:
            for epoch in [1, 2, 5]:
                rows.append(
                    {
                        "objective": objective,
                        "seed": seed,
                        "epoch": epoch,
                        "breadth": 0.1 * epoch + offset,
                        "elevation": 0.05 * epoch + offset,
                        "minority_survival_auc": 0.2 + 0.05 * epoch + offset,
                        "minority_survival_cliffiness": 0.8 - 0.05 * epoch,
                        "auroc": 0.5 + 0.02 * epoch,
                        "average_precision": 0.1 + 0.01 * epoch,
                    }
                )

    report = module.build_allocation_trajectory_report(pd.DataFrame(rows))

    assert "## Final Mean Metrics" in report
    assert "## Fastest High-Elevation Objectives" in report
    assert "## Emergence Lag Summary" in report
    assert "Do objectives follow different paths through morphology space?" in report
    assert "Is allocation morphology established before conventional metrics stabilize?" in report
    assert "Are breadth and elevation behaving as separable axes during training?" in report
