import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_regime_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_regime_hypothesis", script)
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


def test_add_regimes_and_scores():
    module = _load_module()
    construction = __import__("report_accessibility_support_construction_hypothesis")
    features = module.add_regimes(construction.build_feature_matrix(_scores()))
    assert set(features["simple_regime"]) == {"quantized", "continuous"}
    scores = module.model_scores(features)
    assert {"simple_regime", "capacity", "topology_geometry", "construction", "full"} <= set(scores["model_spec"])


def test_absorption_partition_and_residuals():
    module = _load_module()
    construction = __import__("report_accessibility_support_construction_hypothesis")
    features = module.add_regimes(construction.build_feature_matrix(_scores()))
    scores = module.model_scores(features)
    absorption = module.absorption_analysis(scores)
    partition = module.variance_partition(scores)
    residuals = module.regime_residuals(features)
    assert set(absorption["feature_block"]) == {"topology", "construction"}
    assert "regime_effect" in set(partition["component"])
    assert "regime_residual" in residuals.columns


def test_write_report_outputs(tmp_path):
    module = _load_module()
    scores_path = tmp_path / "scores.csv"
    _scores().to_csv(scores_path, index=False)
    outputs = module.write_report(scores_path, tmp_path / "out")
    names = {p.name for p in outputs}
    assert "accessibility_regime_hypothesis.md" in names
    assert "accessibility_regime_summary.json" in names
    assert "regime_variance_partition.csv" in names
    assert "regime_umap.png" in names
    assert all(p.exists() for p in outputs)
