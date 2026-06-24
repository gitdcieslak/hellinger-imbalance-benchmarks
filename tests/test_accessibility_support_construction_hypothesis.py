import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_support_construction_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_support_construction_hypothesis", script)
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


def test_build_feature_matrix():
    module = _load_module()
    features = module.build_feature_matrix(_scores())
    assert {"broad_allocator_index", "concentrated_allocator_index", "quantized_allocator_index", "fragmented_allocator_index"} <= set(features.columns)
    assert len(features) == 60


def test_predictive_lofo_and_mediation(tmp_path):
    module = _load_module()
    features = module.build_feature_matrix(_scores())
    baseline = tmp_path / "topology.csv"
    pd.DataFrame({"held_out_family": ["cart", "random_forest", "xgboost"], "support_topology_r2": [-1.0, -1.0, -1.0]}).to_csv(baseline, index=False)
    scores = module.predictive_scores(features)
    lofo, predictions = module.lofo_scores(features, baseline)
    mediation = module.mediation_scores(features)
    assert set(scores["model_spec"]) == {"capacity", "geometry", "construction", "full"}
    assert set(lofo["held_out_family"]) == {"cart", "random_forest", "xgboost"}
    assert not predictions.empty
    assert "construction_to_geometry" in set(mediation["path"])


def test_write_report_outputs(tmp_path):
    module = _load_module()
    scores_path = tmp_path / "scores.csv"
    _scores().to_csv(scores_path, index=False)
    outputs = module.write_report(scores_path, tmp_path / "out")
    names = {p.name for p in outputs}
    assert "accessibility_support_construction_hypothesis.md" in names
    assert "support_construction_model_scores.csv" in names
    assert "support_construction_family_holdout.csv" in names
    assert "support_construction_mediation.csv" in names
    assert all(p.exists() for p in outputs)
