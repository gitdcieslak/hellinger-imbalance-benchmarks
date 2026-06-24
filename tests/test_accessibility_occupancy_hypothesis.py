import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_occupancy_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_occupancy_hypothesis", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _scores():
    rows = []
    patterns = {
        "cart": [0.2] * 16 + [0.8] * 4,
        "random_forest": [0.1, 0.2, 0.3, 0.4, 0.5] * 4,
        "xgboost": [i / 20 for i in range(1, 21)],
    }
    for dataset in ["a", "b", "c", "d", "e"]:
        for model, score_set in patterns.items():
            for seed in [0, 1, 2, 3]:
                run_id = f"{dataset}:1|{model}|seed={seed}|split={seed}"
                cliff = 0.8 if model == "cart" else 0.35 if model == "random_forest" else 0.15
                for i, score in enumerate(score_set):
                    rows.append({"run_id": run_id, "dataset_id": dataset, "task_id": f"{dataset}:1", "model_id": model, "seed": seed, "split_id": seed, "example_id": i, "positive_label": 1, "score": score, "minority_survival_auc": 0.5, "minority_survival_cliffiness": cliff, "breadth": 1 - cliff, "elevation": cliff, "regime_id": 1})
    return pd.DataFrame(rows)


def test_occupancy_core_analyses():
    module = _load_module()
    data, coords = module.build_coordinates(_scores())
    neigh = module.neighborhood_consistency(data, coords, k=5)
    overlap = module.family_overlap(data, coords, k=5)
    dist = module.distance_correlation(data, coords, max_pairs=2000)
    cond = module.conditional_prediction(data, coords)
    assert "family_mixing" in neigh.columns
    assert "other_family_fraction" in overlap.columns
    assert "spearman_r" in dist.columns
    assert set(cond["model_spec"]) == {"coordinates", "family", "coordinates_plus_family"}


def test_local_family_effect():
    module = _load_module()
    data, coords = module.build_coordinates(_scores())
    neigh = module.neighborhood_consistency(data, coords, k=5)
    local = module.local_family_effect(data, neigh)
    assert "mixing_quintile" in local.columns
    assert len(local) == 5


def test_write_report_outputs(tmp_path):
    module = _load_module()
    scores_path = tmp_path / "scores.csv"
    _scores().to_csv(scores_path, index=False)
    outputs = module.write_report(scores_path, tmp_path / "out")
    names = {p.name for p in outputs}
    assert "accessibility_occupancy_hypothesis.md" in names
    assert "accessibility_occupancy_summary.json" in names
    assert "accessibility_occupancy_conditional_prediction.csv" in names
    assert "accessibility_occupancy_family_map.png" in names
    assert all(p.exists() for p in outputs)
