import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_frontier_morphology_geometry.py"
    spec = importlib.util.spec_from_file_location("report_frontier_morphology_geometry", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_frontier_morphology_geometry module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wedge_rows(n: int = 60):
    rows = []
    for idx in range(n):
        breadth = (idx % 20) / 19 * 1.5
        frac = (idx // 20 + 1) / 3
        upper = max(0.05, 1.0 - 0.45 * breadth)
        lower = 0.02 * breadth
        elevation = lower + frac * (upper - lower)
        rows.append(
            {
                "model_id": "m" + str(idx % 3),
                "seed": idx,
                "breadth": breadth,
                "elevation": elevation,
                "minority_survival_auc": elevation,
                "minority_survival_cliffiness": 0.2 + 0.5 * (1.0 - frac),
            }
        )
    return rows


def _classical_rows():
    rows = []
    for idx, row in enumerate(_wedge_rows(20)):
        rows.append(
            {
                "model_id": "cart" if idx % 2 else "random_forest",
                "seed": idx,
                "skew_ratio": 100,
                "minority_count": 50,
                "fit_failed": False,
                "failure_reason": "",
                **{key: row[key] for key in ["breadth", "elevation", "minority_survival_auc", "minority_survival_cliffiness"]},
            }
        )
    return rows


def test_frontier_fitting_on_synthetic_wedge():
    module = _load_module()
    df = pd.DataFrame(_wedge_rows())

    frontier = module.binwise_frontier(df, n_bins=10)

    assert not frontier.empty
    assert (frontier["upper"] >= frontier["lower"]).all()
    assert frontier["breadth"].is_monotonic_increasing


def test_vertical_slack_and_position_bounds():
    module = _load_module()
    df = pd.DataFrame(_wedge_rows())
    frontier = module.binwise_frontier(df, n_bins=10)

    enriched = module.attach_frontier_features(df, frontier)

    assert (enriched["vertical_slack"] >= -1e-12).all()
    assert enriched["normalized_position_between_envelopes"].between(0.0, 1.0).all()
    assert "frontier_curvature" in enriched.columns


def test_missing_optional_inputs_do_not_crash_report_generation(tmp_path):
    module = _load_module()
    prior = tmp_path / "density_threshold_sweep.csv"
    classical = tmp_path / "classical_allocator_morphology.csv"
    pd.DataFrame(_wedge_rows(30)).to_csv(prior, index=False)
    pd.DataFrame(_classical_rows()).to_csv(classical, index=False)
    missing = tmp_path / "missing.csv"
    output = tmp_path / "frontier_morphology_summary.md"

    outputs = module.write_report(output, prior_inputs=[prior, missing], classical_input=classical)
    report = output.read_text(encoding="utf-8")

    assert "# Frontier Morphology Geometry" in report
    assert "missing:" in report
    assert len(outputs) == 6
    for path in outputs:
        assert path.exists()


def test_hypothesis_label_uses_grouped_cv_rules():
    module = _load_module()
    results = pd.DataFrame(
        [
            {"validation_scheme": "group_experiment_family", "feature_set": "linear_morphology", "cv_r2": 0.1},
            {"validation_scheme": "group_experiment_family", "feature_set": "frontier_only", "cv_r2": 0.2},
            {"validation_scheme": "group_experiment_family", "feature_set": "frontier_morphology", "cv_r2": 0.45},
            {"validation_scheme": "group_experiment_family", "feature_set": "random_forest_reference", "cv_r2": 0.7},
            {"validation_scheme": "group_experiment_family", "feature_set": "rbf_reference", "cv_r2": 0.65},
        ]
    )

    label, bullets = module.hypothesis_label(results)

    assert label == "partially supported"
    assert any("Frontier + morphology" in bullet for bullet in bullets)
