import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_reachability_shape_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_reachability_shape_hypothesis", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _curve_file(tmp_path: Path) -> Path:
    rows = []
    thresholds = np.linspace(0, 1, 11)
    for dataset in ["a", "b", "c", "d", "e"]:
        for mode in ["single", "smooth", "late"]:
            for seed in [0, 1, 2, 3]:
                if mode == "single":
                    curve = np.asarray([1, 1, .2, .2, .2, .2, .2, .2, .2, .1, 0])
                elif mode == "late":
                    curve = np.asarray([1, 1, 1, .95, .9, .8, .6, .4, .2, .1, 0])
                else:
                    curve = np.linspace(1, 0, len(thresholds))
                drops = np.maximum(0, curve[:-1] - curve[1:])
                cliff = float(drops.max() / drops.sum()) if drops.sum() else 0.0
                run_id = f"{dataset}:1|{mode}|seed={seed}|split={seed}"
                for t, reach in zip(thresholds, curve):
                    rows.append({"run_id": run_id, "dataset_id": dataset, "task_id": f"{dataset}:1", "model_id": mode, "seed": seed, "threshold": t, "minority_reachability": float(reach), "minority_survival_auc": float(np.trapezoid(curve, thresholds)), "minority_survival_cliffiness": cliff, "breadth": float(1 - cliff), "elevation": float(cliff), "regime_id": 1})
    path = tmp_path / "curves.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_shape_dataset(tmp_path):
    module = _load_module()
    meta, capacity, matrix, full, thresholds = module.load_shape_dataset([_curve_file(tmp_path)])
    assert len(meta) == len(matrix)
    assert matrix.shape[1] == len(thresholds)
    assert "effective_support" in capacity.columns
    assert not full.empty


def test_model_scores_and_partition(tmp_path):
    module = _load_module()
    meta, capacity, matrix, full, _ = module.load_shape_dataset([_curve_file(tmp_path)])
    scores = module.model_scores(meta, capacity, matrix, full)
    part = module.variance_partition(scores)
    assert "trajectory_shape" in set(scores["model_spec"])
    assert "capacity_plus_trajectory" in set(scores["model_spec"])
    assert not part.empty


def test_pca_archetypes_distance(tmp_path):
    module = _load_module()
    meta, _, matrix, _, thresholds = module.load_shape_dataset([_curve_file(tmp_path)])
    components, scores, pca = module.pca_components(matrix, thresholds)
    arche, labels = module.archetypes(meta, matrix, n_clusters=3)
    dist = module.distance_analysis(meta, matrix, max_pairs=200)
    assert not components.empty
    assert scores.shape[0] == len(meta)
    assert len(labels) == len(meta)
    assert not arche.empty
    assert not dist.empty


def test_write_report_outputs(tmp_path):
    module = _load_module()
    outputs = module.write_report([_curve_file(tmp_path)], tmp_path / "out")
    names = {p.name for p in outputs}
    assert "reachability_shape_hypothesis.md" in names
    assert "reachability_shape_summary.json" in names
    assert "reachability_shape_model_scores.csv" in names
    assert "reachability_umap.png" in names
    assert all(p.exists() for p in outputs)
