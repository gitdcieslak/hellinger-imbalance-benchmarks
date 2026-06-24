import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_support_topology_family_holdout.py"
    spec = importlib.util.spec_from_file_location("report_support_topology_family_holdout", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _scores():
    rows = []
    patterns = {"single": [0.2] * 20, "split": [0.2] * 10 + [0.8] * 10, "spread": [i / 20 for i in range(1, 21)]}
    family_names = {"single": "cart", "split": "random_forest", "spread": "xgboost"}
    for dataset in ["a", "b", "c", "d", "e"]:
        for pattern, score_set in patterns.items():
            model = family_names[pattern]
            for seed in [0, 1, 2, 3]:
                run_id = f"{dataset}:1|{model}|seed={seed}|split={seed}"
                cliff = 1.0 if pattern == "single" else 0.5 if pattern == "split" else 0.1
                for i, score in enumerate(score_set):
                    rows.append({"run_id": run_id, "dataset_id": dataset, "task_id": f"{dataset}:1", "model_id": model, "seed": seed, "split_id": seed, "example_id": i, "positive_label": 1, "score": score, "minority_survival_auc": 0.5, "minority_survival_cliffiness": cliff, "breadth": 1 - cliff, "elevation": cliff, "regime_id": 1})
    return pd.DataFrame(rows)


def test_lofo_scores_and_summary():
    module = _load_module()
    support_module = __import__("report_accessibility_support_topology_hypothesis")
    support = support_module.support_topology_features(_scores())
    meta = support[["run_id", "dataset_id", "task_id", "model_id", module.TARGET, "minority_survival_auc", "breadth", "elevation"]]
    full = support[["breadth", "elevation", "minority_survival_auc", "n_score_groups", "group_mass_entropy", "top1_group_mass", "effective_support_groups"]]
    blocks = module.feature_blocks(support, full)
    lofo, predictions = module.lofo_scores(meta, blocks)
    summary = module.score_summary(lofo)
    assert set(lofo["held_out_family"]) == {"cart", "random_forest", "xgboost"}
    assert not predictions.empty
    assert "support_topology_r2" in set(summary["metric"])


def test_regime_aware_lofo_scores():
    module = _load_module()
    support_module = __import__("report_accessibility_support_topology_hypothesis")
    support = support_module.support_topology_features(_scores())
    meta = support[["run_id", "dataset_id", "task_id", "model_id", module.TARGET, "minority_survival_auc", "breadth", "elevation"]]
    full = support[["breadth", "elevation", "minority_survival_auc", "n_score_groups", "group_mass_entropy", "top1_group_mass", "effective_support_groups"]]
    blocks = module.feature_blocks(support, full)
    lofo, predictions = module.regime_aware_lofo_scores(meta, blocks)
    summary = module.regime_score_summary(lofo)
    assert set(lofo["allocator_regime"]) == {"quantized_allocator"}
    assert set(lofo["held_out_family"]) == {"cart", "random_forest"}
    assert not predictions.empty
    assert "support_topology_r2" in set(summary["metric"])


def test_family_classification():
    module = _load_module()
    support_module = __import__("report_accessibility_support_topology_hypothesis")
    support = support_module.support_topology_features(_scores())
    full = support[["breadth", "elevation", "minority_survival_auc", "n_score_groups", "group_mass_entropy", "top1_group_mass", "effective_support_groups"]]
    blocks = module.feature_blocks(support, full)
    result = module.family_classification(support, blocks["support_topology"])
    assert result["n_families"] == 3
    assert 0 <= result["family_accuracy"] <= 1


def test_write_report_outputs(tmp_path):
    module = _load_module()
    scores_path = tmp_path / "scores.csv"
    _scores().to_csv(scores_path, index=False)
    outputs = module.write_report(scores_path, tmp_path / "out")
    names = {p.name for p in outputs}
    assert "support_topology_family_holdout.md" in names
    assert "support_topology_family_holdout_scores.csv" in names
    assert "support_topology_regime_aware_holdout_scores.csv" in names
    assert "support_topology_regime_aware_holdout_performance.png" in names
    assert "support_topology_family_holdout_performance.png" in names
    assert all(p.exists() for p in outputs)
