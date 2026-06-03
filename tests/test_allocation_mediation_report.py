import importlib.util
from pathlib import Path

import pandas as pd


def _load_report_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_allocation_mediation.py"
    spec = importlib.util.spec_from_file_location("report_allocation_mediation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_allocation_mediation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    rows = []
    for objective, offset in [("bce", 0.0), ("weighted_bce", 0.2), ("oversampled_bce", -0.1)]:
        for seed in range(3):
            auc = 0.4 + offset + 0.02 * seed
            cliffiness = 0.7 - offset - 0.02 * seed
            for mode, scale in [("knn", 1), ("radius", 0.1)]:
                rows.append(
                    {
                        "objective": objective,
                        "seed": seed,
                        "model_name": f"mlp_{objective}",
                        "fit_failed": False,
                        "topology_mode": mode,
                        "k": scale if mode == "knn" else "",
                        "radius_quantile": scale if mode == "radius" else "",
                        "minority_survival_auc": auc,
                        "minority_survival_cliffiness": cliffiness,
                        "positive_histogram_entropy": 1.0 - auc,
                        "positive_effective_score_bins": 2.0 - auc,
                        "positive_max_bin_mass": auc,
                        "positive_top_bin_mass": auc,
                        "positive_score_gini_or_concentration_index": cliffiness,
                        "n_components": 2 + seed,
                        "giant_component_fraction": 0.5 + offset,
                        "component_entropy": 0.1 + seed,
                        "isolated_positive_fraction": 0.0,
                    }
                )
    return rows


def test_mediation_report_builds_run_level_and_answers_questions():
    module = _load_report_module()
    df = pd.DataFrame(_rows())

    run_level = module.build_run_level_dataset(df)
    report = module.build_allocation_mediation_report(df)

    assert len(run_level) == 9
    assert "topology_n_components_mean" in run_level.columns
    assert "## Regression Family Performance" in report
    assert "Does objective explain survival?" in report
    assert "Does allocation explain survival?" in report
    assert "Does allocation explain away much of objective?" in report
    assert "Does topology add anything after allocation?" in report
