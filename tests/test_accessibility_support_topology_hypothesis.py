import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_support_topology_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_support_topology_hypothesis", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _scores():
    rows = []
    for dataset in ["a", "b", "c", "d", "e"]:
        for model, score_set in {"single": [0.2] * 20, "split": [0.2] * 10 + [0.8] * 10, "spread": [i / 20 for i in range(1, 21)]}.items():
            for seed in [0, 1, 2]:
                run_id = f"{dataset}:1|{model}|seed={seed}|split={seed}"
                cliff = 1.0 if model == "single" else 0.5 if model == "split" else 0.1
                for i, score in enumerate(score_set):
                    rows.append({"run_id": run_id, "dataset_id": dataset, "task_id": f"{dataset}:1", "model_id": model, "seed": seed, "split_id": seed, "example_id": i, "positive_label": 1, "score": score, "minority_survival_auc": 0.5, "minority_survival_cliffiness": cliff, "breadth": 1 - cliff, "elevation": cliff, "regime_id": 1})
    return pd.DataFrame(rows)


def test_support_topology_features_nonempty():
    module = _load_module()
    features = module.support_topology_features(_scores())
    assert len(features) == 45
    assert {"n_score_groups", "group_mass_entropy", "n_support_islands", "stable_to_fragile_ratio"}.issubset(features.columns)
    assert features["effective_support_groups"].notna().all()


def test_model_scores_and_partition():
    module = _load_module()
    support = module.support_topology_features(_scores())
    meta = support[["run_id", "dataset_id", "task_id", "model_id", module.TARGET, "minority_survival_auc", "breadth", "elevation"]]
    full = support[["n_score_groups", "group_mass_entropy", "top1_group_mass", "breadth", "elevation"]]
    scores = module.model_scores(meta, support, full)
    partition = module.variance_partition(scores)
    assert "capacity_plus_support_topology" in set(scores["model_spec"])
    assert "support_topology_unique" in set(partition["component"])


def test_write_report_outputs(tmp_path):
    module = _load_module()
    scores_path = tmp_path / "positive_scores.csv"
    _scores().to_csv(scores_path, index=False)
    outputs = module.write_report(scores_path, tmp_path / "out")
    names = {p.name for p in outputs}
    assert "accessibility_support_topology_hypothesis.md" in names
    assert "accessibility_support_topology_summary.json" in names
    assert "support_topology_model_scores.csv" in names
    assert "support_topology_umap.png" in names
    assert all(p.exists() for p in outputs)
