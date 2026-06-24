import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_capacity_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_capacity_hypothesis", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _curve_file(tmp_path: Path) -> Path:
    rows = []
    thresholds = np.linspace(0, 1, 11)
    for dataset in ["a", "b", "c", "d", "e"]:
        for model_id, capacity in [("cart", 2), ("rf", 8), ("gbm", 5)]:
            for seed in [0, 1, 2]:
                scores = np.linspace(0.1, 0.95, capacity)
                weights = np.ones(capacity) / capacity
                cliff = 1.0 / capacity
                run_id = f"{dataset}:1|{model_id}|seed={seed}|split={seed}"
                for t in thresholds:
                    rows.append(
                        {
                            "run_id": run_id,
                            "dataset_id": dataset,
                            "task_id": f"{dataset}:1",
                            "model_id": model_id,
                            "seed": seed,
                            "threshold": t,
                            "minority_reachability": float(weights[scores >= t].sum()),
                            "minority_survival_auc": float(scores.mean()),
                            "minority_survival_cliffiness": cliff,
                            "breadth": float(np.log(capacity)),
                            "elevation": float(1.0 / capacity),
                            "regime_id": 1,
                        }
                    )
    path = tmp_path / "curves.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_capacity_dataset_has_capacity_columns(tmp_path):
    module = _load_module()

    meta, morphology, capacity, thresholds = module.load_capacity_dataset([_curve_file(tmp_path)])

    assert len(meta) == len(capacity)
    assert "effective_support" in capacity.columns
    assert "positive_score_entropy" in capacity.columns
    assert len(thresholds) == 11


def test_capacity_relationship_returns_model_scores(tmp_path):
    module = _load_module()
    meta, _, capacity, _ = module.load_capacity_dataset([_curve_file(tmp_path)])

    scores = module.capacity_relationship(meta, capacity)

    assert {"capacity_ridge", "capacity_random_forest", "capacity_gradient_boosting"}.issubset(set(scores["model_spec"]))
    assert scores["grouped_cv_r2"].notna().all()


def test_family_conditioning_and_binning(tmp_path):
    module = _load_module()
    meta, _, capacity, _ = module.load_capacity_dataset([_curve_file(tmp_path)])

    family_scores, importance = module.family_conditioning(meta, capacity)
    bins = module.capacity_binning(meta, capacity)

    assert not family_scores.empty
    assert {"capacity", "family"}.issubset(set(importance["feature_block"]))
    assert not bins.empty
    assert "mean_cliffiness" in bins.columns


def test_interventions_modify_capacity_metrics(tmp_path):
    module = _load_module()
    from report_definitive_reachability_topology import load_curve_files, pivot_curves

    curves = load_curve_files([_curve_file(tmp_path)])
    meta, matrix, thresholds = pivot_curves(curves)

    interventions = module.capacity_interventions(meta, matrix, thresholds, max_runs=5)

    assert {"collapse", "expansion"}.issubset(set(interventions["intervention"]))
    assert interventions["effective_support"].notna().all()
    assert interventions["minority_survival_cliffiness"].between(0, 1).all()


def test_write_report_creates_outputs(tmp_path):
    module = _load_module()
    output_dir = tmp_path / "out"

    outputs = module.write_report([_curve_file(tmp_path)], output_dir)

    names = {p.name for p in outputs}
    assert "accessibility_capacity_hypothesis.md" in names
    assert "accessibility_capacity_summary.json" in names
    assert "capacity_intervention_results.csv" in names
    assert "capacity_vs_cliffiness.png" in names
    assert all(p.exists() for p in outputs)
