import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_morphology_decomposition.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_morphology_decomposition", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _curve_file(tmp_path: Path) -> Path:
    rows = []
    thresholds = np.linspace(0, 1, 11)
    for dataset in ["a", "b", "c", "d", "e"]:
        for cap in [2, 4, 8]:
            for skew in [0.0, 0.6]:
                for seed in [0, 1, 2]:
                    scores = np.linspace(0.1, 0.95, cap)
                    if skew:
                        weights = np.linspace(1.0, 3.0, cap)
                    else:
                        weights = np.ones(cap)
                    weights = weights / weights.sum()
                    curve = np.asarray([weights[scores >= t].sum() for t in thresholds])
                    drops = np.maximum(0, curve[:-1] - curve[1:])
                    cliff = float(drops.max() / drops.sum()) if drops.sum() else 0.0
                    run_id = f"{dataset}:1|m{cap}|seed={seed}|skew={skew}"
                    for t, reach in zip(thresholds, curve):
                        rows.append({
                            "run_id": run_id,
                            "dataset_id": dataset,
                            "task_id": f"{dataset}:1",
                            "model_id": f"m{cap}",
                            "seed": seed,
                            "threshold": t,
                            "minority_reachability": float(reach),
                            "minority_survival_auc": float(np.trapezoid(curve, thresholds)),
                            "minority_survival_cliffiness": cliff,
                            "breadth": float(np.log(cap)),
                            "elevation": float(weights.max()),
                            "regime_id": 1,
                        })
    path = tmp_path / "curves.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_decomposition_dataset_blocks(tmp_path):
    module = _load_module()
    meta, capacity, arrangement, full = module.load_decomposition_dataset([_curve_file(tmp_path)])
    assert len(meta) == len(capacity) == len(arrangement) == len(full)
    assert "effective_support" in capacity.columns
    assert any(c in arrangement.columns for c in ["positive_score_gini", "top_1pct_mass"])


def test_model_scores_and_variance_partition(tmp_path):
    module = _load_module()
    meta, capacity, arrangement, full = module.load_decomposition_dataset([_curve_file(tmp_path)])
    scores = module.model_scores(meta, capacity, arrangement, full)
    partition = module.variance_partition(scores)
    assert {"capacity", "arrangement", "capacity_plus_arrangement", "full_morphology"}.issubset(set(scores["model_spec"]))
    assert {"capacity_unique", "arrangement_unique", "shared_capacity_arrangement", "unexplained"}.issubset(set(partition["component"]))


def test_frontier_and_controlled_effects(tmp_path):
    module = _load_module()
    meta, capacity, arrangement, _ = module.load_decomposition_dataset([_curve_file(tmp_path)])
    front = module.frontier(meta, capacity)
    cap_ctrl, arr_ctrl = module.controlled_effects(meta, capacity, arrangement)
    assert not front.empty
    assert "p90_cliffiness" in front.columns
    assert not cap_ctrl.empty
    assert not arr_ctrl.empty


def test_write_report_creates_outputs(tmp_path):
    module = _load_module()
    outputs = module.write_report([_curve_file(tmp_path)], tmp_path / "out")
    names = {p.name for p in outputs}
    assert "accessibility_morphology_decomposition.md" in names
    assert "accessibility_morphology_decomposition_summary.json" in names
    assert "morphology_decomposition_model_scores.csv" in names
    assert "capacity_arrangement_scatter.png" in names
    assert all(p.exists() for p in outputs)
