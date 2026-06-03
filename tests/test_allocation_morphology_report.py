import importlib.util
from pathlib import Path

import pandas as pd


def _load_report_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_allocation_morphology.py"
    spec = importlib.util.spec_from_file_location("report_allocation_morphology", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_allocation_morphology module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _df():
    rows = []
    values = [
        ("a", 0, 0.1, 0.1),
        ("a", 1, 0.9, 0.1),
        ("b", 0, 0.1, 0.9),
        ("b", 1, 0.9, 0.9),
    ]
    for objective, seed, breadth, elevation in values:
        rows.append(
            {
                "objective": objective,
                "seed": seed,
                "model_name": objective,
                "fit_failed": False,
                "positive_histogram_entropy": breadth,
                "positive_effective_score_bins": 1.0 + breadth,
                "positive_top_bin_mass": elevation,
                "positive_max_bin_mass": max(breadth, elevation),
                "minority_survival_auc": elevation,
                "minority_survival_cliffiness": breadth,
                "auroc": 0.5 + elevation,
                "average_precision": 0.1 + elevation,
            }
        )
    return pd.DataFrame(rows)


def test_quadrant_classification_uses_median_breadth_and_elevation():
    module = _load_report_module()
    run_level = module.classify_allocation_quadrants(module.build_run_level_dataset(_df()))

    quadrants = set(run_level["allocation_quadrant"])

    assert quadrants == {
        "low_breadth_low_elevation",
        "high_breadth_low_elevation",
        "low_breadth_high_elevation",
        "high_breadth_high_elevation",
    }


def test_allocation_morphology_report_contains_required_sections():
    module = _load_report_module()
    report = module.build_allocation_morphology_report(_df())

    assert "## Objective-Level Means" in report
    assert "## Breadth/Elevation Correlations" in report
    assert "## Objective-Level Quadrant Counts" in report
    assert "## Interpretation" in report
    assert "separable breadth and elevation axes" in report
