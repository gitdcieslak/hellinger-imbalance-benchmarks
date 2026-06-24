import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_intervention_vector_hypothesis.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_intervention_vector_hypothesis", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _paired_rows():
    rows = []
    for dataset in ["a", "b", "c", "d", "e"]:
        for i in range(8):
            pair_id = f"{dataset}-{i}"
            base = {"pair_id": pair_id, "intervention": "dropout", "dataset_id": dataset, "model_id": "mlp", "seed": i, "split_id": i}
            rows.append({**base, "stage": "before", "coord_pc1": float(i), "coord_pc2": 0.0, "coord_pc3": 0.0, "minority_survival_cliffiness": 0.8, "persistence_weighted_mass": 0.2, "minority_survival_auc": 0.4, "breadth": 0.3, "elevation": 0.5})
            rows.append({**base, "stage": "after", "coord_pc1": float(i) + 1.0, "coord_pc2": 0.2, "coord_pc3": 0.0, "minority_survival_cliffiness": 0.6, "persistence_weighted_mass": 0.4, "minority_survival_auc": 0.6, "breadth": 0.5, "elevation": 0.6})
    return pd.DataFrame(rows)


def test_displacements_and_consistency():
    module = _load_module()
    disp = module.displacement_vectors(_paired_rows())
    consistency = module.vector_consistency(disp)
    assert len(disp) == 40
    assert set(disp["intervention"]) == {"dropout"}
    assert consistency.iloc[0]["mean_cosine_similarity"] > 0.99


def test_prediction_and_trajectory_unavailable():
    module = _load_module()
    disp = module.displacement_vectors(_paired_rows())
    pred = module.morphology_prediction(disp)
    traj = module.trajectory_analysis(_paired_rows())
    assert f"delta_{module.TARGETS[0]}" in set(pred["target"])
    assert traj.iloc[0]["intervention"] == "unavailable"


def test_write_report_outputs(tmp_path):
    module = _load_module()
    input_path = tmp_path / "coords.csv"
    _paired_rows().to_csv(input_path, index=False)
    outputs = module.write_report(input_path, tmp_path / "out")
    names = {p.name for p in outputs}
    assert "accessibility_intervention_vector_hypothesis.md" in names
    assert "accessibility_intervention_displacements.csv" in names
    assert "accessibility_intervention_vector_consistency.csv" in names
    assert "vector_field_dropout.png" in names
    assert all(p.exists() for p in outputs)


def test_unpaired_report_is_blocked(tmp_path):
    module = _load_module()
    input_path = tmp_path / "coords.csv"
    pd.DataFrame({"run_id": ["r1"], "coord_pc1": [0.0], "coord_pc2": [1.0]}).to_csv(input_path, index=False)
    outputs = module.write_report(input_path, tmp_path / "out")
    summary = [p for p in outputs if p.name == "accessibility_intervention_vector_summary.json"][0].read_text(encoding="utf-8")
    assert '"outcome": "blocked"' in summary
