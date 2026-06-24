import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_arrangement_mechanism.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_arrangement_mechanism", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _curve_file(tmp_path: Path) -> Path:
    rows = []
    thresholds = np.linspace(0, 1, 11)
    for dataset in ["a", "b", "c", "d", "e"]:
        for cap in [2, 4, 8]:
            for skew in [0.0, 0.7]:
                for seed in [0, 1, 2]:
                    scores = np.linspace(0.1, 0.95, cap)
                    weights = np.linspace(1, 4, cap) if skew else np.ones(cap)
                    weights = weights / weights.sum()
                    curve = np.asarray([weights[scores >= t].sum() for t in thresholds])
                    drops = np.maximum(0, curve[:-1] - curve[1:])
                    cliff = float(drops.max() / drops.sum()) if drops.sum() else 0.0
                    run_id = f"{dataset}:1|m{cap}|seed={seed}|skew={skew}"
                    for t, reach in zip(thresholds, curve):
                        rows.append({"run_id": run_id, "dataset_id": dataset, "task_id": f"{dataset}:1", "model_id": f"m{cap}", "seed": seed, "threshold": t, "minority_reachability": float(reach), "minority_survival_auc": float(np.trapezoid(curve, thresholds)), "minority_survival_cliffiness": cliff, "breadth": float(np.log(cap)), "elevation": float(weights.max()), "regime_id": 1})
    path = tmp_path / "curves.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_arrangement_features_have_blocks(tmp_path):
    module = _load_module()
    from report_definitive_reachability_topology import load_curve_files, pivot_curves
    _, matrix, thresholds = pivot_curves(load_curve_files([_curve_file(tmp_path)]))
    features, blocks = module.arrangement_features_from_curves(matrix, thresholds)
    assert len(features) == len(matrix)
    assert {"concentration", "spacing", "occupancy", "drop_structure"}.issubset(blocks)
    assert "drop_entropy" in features.columns


def test_model_scores_and_partition(tmp_path):
    module = _load_module()
    from report_accessibility_morphology_decomposition import load_decomposition_dataset
    from report_definitive_reachability_topology import load_curve_files, pivot_curves
    path = _curve_file(tmp_path)
    meta, capacity, _, full = load_decomposition_dataset([path])
    _, matrix, thresholds = pivot_curves(load_curve_files([path]))
    arrangement, _ = module.arrangement_features_from_curves(matrix, thresholds)
    scores = module.model_scores(meta, capacity, arrangement, full)
    part = module.partition_from_scores(scores)
    assert "arrangement_mechanism" in set(scores["model_spec"])
    assert not part.empty


def test_importance_and_report_outputs(tmp_path):
    module = _load_module()
    outputs = module.write_report([_curve_file(tmp_path)], tmp_path / "out")
    names = {p.name for p in outputs}
    assert "accessibility_arrangement_mechanism.md" in names
    assert "arrangement_feature_importance.csv" in names
    assert "arrangement_umap.png" in names
    assert "arrangement_mechanism_summary.json" in names
    assert all(p.exists() for p in outputs)
