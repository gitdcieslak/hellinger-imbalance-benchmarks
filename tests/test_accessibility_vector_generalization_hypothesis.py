import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_vector_generalization_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_vector_generalization_hypothesis", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _paired():
    rows = []
    for dataset in ["a", "b", "c", "d", "e"]:
        for seed in range(6):
            for intervention, step in [("bagging", 1.0), ("dropout", 0.5)]:
                pair_id = f"{dataset}-{seed}-{intervention}"
                for stage, strength, offset in [("before", 0.0, 0.0), ("after", 1.0, step)]:
                    rows.append({"pair_id": pair_id, "dataset": dataset, "dataset_id": dataset, "task_id": f"{dataset}:1", "split_id": seed, "seed": seed, "model_family": "test", "base_model_id": "base", "model_id": stage, "intervention": intervention, "stage": stage, "strength": strength, "coord_1": seed + offset, "coord_2": offset * 0.5, "coord_3": 0.0, "coord_4": 0.0, "coord_5": 0.0, "minority_survival_cliffiness": 0.8 - 0.2 * offset, "persistence": 0.2 + 0.1 * offset, "minority_survival_auc": 0.4 + 0.1 * offset, "breadth": 0.3 + 0.2 * offset, "elevation": 0.5 + 0.05 * offset})
    return pd.DataFrame(rows)


def test_generalization_analyses():
    module = _load_module()
    disp = module.paired_displacements(_paired())
    consistency = module.intra_intervention_consistency(disp)
    holdout = module.dataset_holdout_consistency(disp)
    arithmetic = module.vector_arithmetic(disp)
    counterfactual = module.arithmetic_counterfactual_error(disp)
    assert set(disp["intervention"]) == {"bagging", "dropout"}
    assert consistency[consistency["available"]].shape[0] == 2
    assert not holdout.empty
    assert "delta_minority_survival_cliffiness" in set(arithmetic["target"])
    assert set(counterfactual["intervention"]) == {"bagging", "dropout"}


def test_write_report_outputs(tmp_path):
    module = _load_module()
    input_path = tmp_path / "paired.csv"
    _paired().to_csv(input_path, index=False)
    outputs = module.write_report(input_path, tmp_path / "out")
    names = {p.name for p in outputs}
    assert "accessibility_vector_generalization_hypothesis.md" in names
    assert "accessibility_vector_generalization_consistency.csv" in names
    assert "accessibility_vector_generalization_dataset_holdout.csv" in names
    assert "accessibility_vector_generalization_arithmetic.csv" in names
    assert all(p.exists() for p in outputs)
