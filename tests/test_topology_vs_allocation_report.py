import importlib.util
from pathlib import Path

import pandas as pd


def _load_report_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_topology_vs_allocation.py"
    spec = importlib.util.spec_from_file_location("report_topology_vs_allocation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_topology_vs_allocation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_identifies_stronger_family_section():
    module = _load_report_module()
    rows = []
    for objective in ["bce", "weighted_bce", "oversampled_bce"]:
        offset = {"bce": 0.0, "weighted_bce": 0.1, "oversampled_bce": -0.1}[objective]
        model_name = {
            "bce": "mlp_bce",
            "weighted_bce": "mlp_weighted",
            "oversampled_bce": "mlp_oversampled",
        }[objective]
        for seed, cliffiness_base in enumerate([0.1, 0.4, 0.8]):
            cliffiness = max(0.0, min(1.0, cliffiness_base + offset))
            rows.append(
                {
                    "objective": objective,
                    "seed": seed,
                    "model_name": model_name,
                    "fit_failed": False,
                    "topology_mode": "knn",
                    "k": 1,
                    "radius_quantile": "",
                    "minority_survival_auc": 1.0 - cliffiness,
                    "persistence": 1.0 - cliffiness,
                    "minority_survival_cliffiness": cliffiness,
                    "minority_survival_max_drop": cliffiness,
                    "minority_survival_total_variation": cliffiness,
                    "minority_survival_effective_drop_count": 1.0,
                    "positive_unique_score_ratio": 1.0 - cliffiness,
                    "positive_quantization_score": cliffiness,
                    "positive_histogram_entropy": 1.0 - cliffiness,
                    "positive_effective_score_bins": 2.0,
                    "positive_max_bin_mass": cliffiness,
                    "positive_top_bin_mass": cliffiness,
                    "positive_score_iqr": cliffiness,
                    "positive_score_q10_q90_width": cliffiness,
                    "positive_score_gini_or_concentration_index": cliffiness,
                    "n_components": 2 + seed,
                    "giant_component_fraction": 0.5,
                    "mean_component_size": 2.0,
                    "median_component_size": 2.0,
                    "component_entropy": 0.1 * seed,
                    "isolated_positive_fraction": 0.0,
                }
            )

    report = module.build_topology_vs_allocation_report(pd.DataFrame(rows), n_bootstrap=25)

    assert "## Stronger Explanatory Family" in report
    assert "## Bootstrap Confidence Intervals" in report
    assert "## Objective-Pair Contrasts" in report
    assert "## Run-Level Topology Correlations By Mode And Scale" in report
    assert "allocation" in report
    assert "weighted_bce - oversampled_bce" in report
    assert "positive_quantization_score" in report
