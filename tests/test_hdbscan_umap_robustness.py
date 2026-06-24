import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_hdbscan_umap_robustness.py"
    spec = importlib.util.spec_from_file_location("report_hdbscan_umap_robustness", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_hdbscan_umap_robustness module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overall_outcome_requires_all_claims_for_strong_success():
    module = _load_module()
    claims = pd.DataFrame(
        [
            {"classification": "Fully Replicated"},
            {"classification": "Fully Replicated"},
            {"classification": "Not Replicated"},
        ]
    )

    assert module.overall_outcome(claims) == "Acceptable Success"


def test_local_law_survival_compares_against_global_cv():
    module = _load_module()
    equation_summary = pd.DataFrame(
        [
            {"regime_id": "global", "ridge_cv_r2": 0.5},
            {"regime_id": "0", "ridge_cv_r2": 0.4},
            {"regime_id": "1", "ridge_cv_r2": 0.3},
        ]
    )

    assert module.local_law_survives(equation_summary) is False
