import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_support_granularity_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_support_granularity_hypothesis", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _scores():
    rows = []
    patterns = {
        "cart": [0.2] * 16 + [0.8] * 4,
        "random_forest": [0.1, 0.2, 0.3, 0.4, 0.5] * 4,
    }
    for dataset in ["a", "b", "c", "d", "e"]:
        for model, score_set in patterns.items():
            for seed in [0, 1, 2, 3, 4, 5]:
                run_id = f"{dataset}:1|{model}|seed={seed}|split={seed}"
                cliff = 0.8 if model == "cart" else 0.2
                for i, score in enumerate(score_set):
                    rows.append({"run_id": run_id, "dataset_id": dataset, "task_id": f"{dataset}:1", "model_id": model, "seed": seed, "split_id": seed, "example_id": i, "positive_label": 1, "score": score, "minority_survival_auc": 0.5, "minority_survival_cliffiness": cliff, "breadth": 1 - cliff, "elevation": cliff, "regime_id": 1})
    return pd.DataFrame(rows)


def test_granularity_features():
    module = _load_module()
    features = module.granularity_features(_scores())
    assert {"n_support_groups", "support_redundancy", "mass_outside_top3", "support_fragility_index"} <= set(features.columns)
    cart = features[features["model_id"] == "cart"].iloc[0]
    assert cart["n_support_groups"] == 2
    assert round(cart["top1_group_mass"], 6) == 0.8
    assert round(cart["support_redundancy"], 6) == 0.2


def test_scores_replacement_and_transfer(tmp_path):
    module = _load_module()
    qmodule = __import__("report_accessibility_quantization_hypothesis")
    scores = _scores()
    features = module.granularity_features(scores)
    quant = qmodule.quantization_features(scores).merge(features[["run_id", *module.CAPACITY_COLS]], on="run_id")
    model_scores = module.explanatory_scores(features, module.QUANTIZED_FAMILIES, "quantized")
    replacement = module.replacement_test(features, quant)
    baseline = tmp_path / "baseline.csv"
    pd.DataFrame({"train_family": ["random_forest", "cart"], "test_family": ["cart", "random_forest"], "quantization_r2": [-1.0, -1.0]}).to_csv(baseline, index=False)
    transfer = module.cart_rf_transfer(features, baseline)
    assert set(model_scores["model_spec"]) == {"capacity", "granularity", "capacity_plus_granularity"}
    assert set(replacement["model_spec"]) == {"capacity_plus_quantization", "capacity_plus_granularity"}
    assert set(transfer["test_family"]) == {"cart", "random_forest"}


def test_write_report_outputs(tmp_path):
    module = _load_module()
    scores_path = tmp_path / "scores.csv"
    _scores().to_csv(scores_path, index=False)
    outputs = module.write_report(scores_path, tmp_path / "out")
    names = {p.name for p in outputs}
    assert "accessibility_support_granularity_hypothesis.md" in names
    assert "support_granularity_model_scores.csv" in names
    assert "support_granularity_mediation.csv" in names
    assert "support_granularity_frontier.png" in names
    assert all(p.exists() for p in outputs)
