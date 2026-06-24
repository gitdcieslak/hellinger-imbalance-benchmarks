import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_event_structure_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_event_structure_hypothesis", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _curve_file(tmp_path: Path) -> Path:
    rows = []
    thresholds = np.linspace(0, 1, 11)
    for dataset in ["a", "b", "c", "d", "e"]:
        for mode in ["single", "stair", "tail"]:
            for seed in [0, 1, 2, 3]:
                if mode == "single":
                    curve = np.asarray([1, 1, 1, .1, .1, .1, .1, .1, .1, .1, 0])
                elif mode == "tail":
                    curve = np.asarray([1, 1, 1, 1, 1, 1, 1, .9, .8, .1, 0])
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


def test_event_features_have_expected_blocks(tmp_path):
    module = _load_module()
    from report_definitive_reachability_topology import load_curve_files, pivot_curves
    _, matrix, thresholds = pivot_curves(load_curve_files([_curve_file(tmp_path)]))
    events, blocks = module.event_features_from_curves(matrix, thresholds)
    assert len(events) == len(matrix)
    assert {"magnitude", "count", "timing", "hierarchy", "spacing", "tail"}.issubset(blocks)
    assert "top1_fraction" in events.columns


def test_scores_partition_archetypes_frontier(tmp_path):
    module = _load_module()
    from report_accessibility_morphology_decomposition import load_decomposition_dataset
    from report_definitive_reachability_topology import load_curve_files, pivot_curves
    path = _curve_file(tmp_path)
    meta, capacity, _, full = load_decomposition_dataset([path])
    _, matrix, thresholds = pivot_curves(load_curve_files([path]))
    events, _ = module.event_features_from_curves(matrix, thresholds)
    scores = module.model_scores(meta, capacity, events, full)
    partition = module.variance_partition(scores)
    archetypes, labels = module.event_archetypes(meta, events)
    frontier = module.event_frontier(meta, capacity, events)
    assert "capacity_plus_event_structure" in set(scores["model_spec"])
    assert "event_unique" in set(partition["component"])
    assert len(labels) == len(meta)
    assert not archetypes.empty
    assert not frontier.empty


def test_write_report_outputs(tmp_path):
    module = _load_module()
    outputs = module.write_report([_curve_file(tmp_path)], tmp_path / "out")
    names = {p.name for p in outputs}
    assert "accessibility_event_structure_hypothesis.md" in names
    assert "accessibility_event_structure_summary.json" in names
    assert "event_structure_model_scores.csv" in names
    assert "event_structure_frontier.png" in names
    assert all(p.exists() for p in outputs)
