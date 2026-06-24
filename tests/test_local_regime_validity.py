import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_local_regime_validity.py"
    spec = importlib.util.spec_from_file_location("report_local_regime_validity", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_local_region_boundaries():
    module = _load_module()

    assert module.local_region(0.2, 0.9) == "high_elevation_low_breadth"
    assert module.local_region(0.7, 0.9) == "high_elevation_moderate_breadth"
    assert module.local_region(1.0, 0.5) == "mid_elevation"
    assert module.local_region(1.0, 0.2) == "low_elevation_high_breadth"
    assert module.local_region(0.2, 0.2) == "low_elevation_low_breadth"


def test_assign_regions_uses_quantile_fallback_when_sparse():
    module = _load_module()
    df = pd.DataFrame({"breadth": np.linspace(0.1, 1.0, 10), "elevation": np.linspace(0.1, 0.9, 10)})

    out, used_fallback = module.assign_regions(df, min_rows=5)

    assert used_fallback
    assert "region" in out.columns
    assert "fallback_region" in out.columns


def test_classify_delta_thresholds():
    module = _load_module()

    assert module.classify_delta(0.05) == "regime-valid region"
    assert module.classify_delta(0.02) == "weak-regime region"
    assert module.classify_delta(0.01) == "regime-invalid region"
    assert module.classify_delta(-0.1) == "regime-invalid region"


def test_regional_predictive_scores_schema():
    module = _load_module()
    rows = []
    for dataset in ["a", "b", "c"]:
        for i in range(12):
            breadth = 0.1 + 0.02 * i if dataset != "c" else 0.8 + 0.02 * i
            elevation = 0.9 if dataset != "c" else 0.5
            rows.append(
                {
                    "dataset_name": dataset,
                    "task_id": f"{dataset}:1",
                    "model_id": "m",
                    "breadth": breadth,
                    "elevation": elevation,
                    "minority_survival_auc": elevation,
                    "minority_survival_cliffiness": 0.2 + breadth,
                    "regime_id": i % 2,
                    "expanded_cluster": i % 3,
                }
            )
    df, _ = module.assign_regions(pd.DataFrame(rows), min_rows=1)

    scores = module.regional_predictive_scores(df)

    assert {"region", "model_spec", "grouped_cv_r2", "delta_vs_morphology", "valid", "skip_reason"}.issubset(scores.columns)
    assert "morphology_only" in set(scores["model_spec"])
