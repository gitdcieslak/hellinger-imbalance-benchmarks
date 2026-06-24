import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_transition_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_transition_hypothesis", script)
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
                    curve = np.asarray([1, 1, 1, .2, .2, .2, .2, .2, .2, .2, 0])
                elif mode == "late":
                    curve = np.asarray([1, 1, 1, 1, 1, .9, .8, .6, .3, .1, 0])
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


def test_transition_features_include_blocks(tmp_path):
    module = _load_module()
    from report_definitive_reachability_topology import load_curve_files, pivot_curves
    _, matrix, thresholds = pivot_curves(load_curve_files([_curve_file(tmp_path)]))
    features, blocks = module.transition_features_from_curves(matrix, thresholds)
    assert len(features) == len(matrix)
    assert {"drop_spectrum", "persistence", "roughness", "concentration", "graph"}.issubset(blocks)
    assert "largest_drop_fraction" in features.columns


def test_model_scores_partition_and_archetypes(tmp_path):
    module = _load_module()
    from report_accessibility_morphology_decomposition import load_decomposition_dataset
    from report_definitive_reachability_topology import load_curve_files, pivot_curves
    path = _curve_file(tmp_path)
    meta, capacity, _, full = load_decomposition_dataset([path])
    _, matrix, thresholds = pivot_curves(load_curve_files([path]))
    transition, _ = module.transition_features_from_curves(matrix, thresholds)
    scores = module.model_scores(meta, capacity, transition, full)
    part = module.variance_partition(scores)
    profiles, labels = module.archetypes(meta, transition, matrix, n_clusters=3)
    assert "transition" in set(scores["model_spec"])
    assert not part.empty
    assert len(labels) == len(meta)
    assert not profiles.empty


def test_write_report_outputs(tmp_path):
    module = _load_module()
    outputs = module.write_report([_curve_file(tmp_path)], tmp_path / "out")
    names = {p.name for p in outputs}
    assert "accessibility_transition_hypothesis.md" in names
    assert "transition_model_scores.csv" in names
    assert "transition_umap.png" in names
    assert "accessibility_transition_summary.json" in names
    assert all(p.exists() for p in outputs)
