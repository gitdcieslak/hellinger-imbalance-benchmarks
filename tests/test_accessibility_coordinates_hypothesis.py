import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_coordinates_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_coordinates_hypothesis", script)
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


def test_coordinate_learning_and_predictiveness():
    module = _load_module()
    features, X = module.coordinate_feature_matrix(_scores())
    coords, loadings, explained, _ = module.learn_coordinates(X, n_components=3)
    pred = module.predictiveness(features, X, coords)
    assert {"coord_pc1", "coord_pc2", "coord_umap1", "coord_spectral1"} <= set(coords.columns)
    assert not loadings.empty
    assert len(explained) == 3
    assert "coordinates_3d" in set(pred["model_spec"])


def test_neighbors_mixing_and_vector_fields():
    module = _load_module()
    features, X = module.coordinate_feature_matrix(_scores())
    coords, _, _, _ = module.learn_coordinates(X, n_components=3)
    neighbors, mixing = module.neighbor_analysis(features, coords, k=5)
    vectors = module.vector_fields(features, coords)
    assert "mean_abs_delta_minority_survival_cliffiness" in neighbors.columns
    assert "family_mixing_score" in mixing.columns
    assert vectors.iloc[0]["intervention"] == "unavailable"


def test_write_report_outputs(tmp_path):
    module = _load_module()
    scores_path = tmp_path / "scores.csv"
    _scores().to_csv(scores_path, index=False)
    outputs = module.write_report(scores_path, tmp_path / "out")
    names = {p.name for p in outputs}
    assert "accessibility_coordinates_hypothesis.md" in names
    assert "accessibility_coordinate_loadings.csv" in names
    assert "accessibility_coordinate_predictiveness.csv" in names
    assert "family_mixing_map.png" in names
    assert all(p.exists() for p in outputs)
