import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "capture_accessibility_paired_interventions.py"
    spec = importlib.util.spec_from_file_location("capture_accessibility_paired_interventions", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    coord_rows = []
    feat_rows = []
    for dataset in ["a", "b"]:
        for seed in [0, 1]:
            for split in [0, 1]:
                for model, cliff, c1 in [("cart", 0.9, 1.0), ("random_forest", 0.4, 2.0), ("xgboost", 0.3, 3.0)]:
                    run_id = f"{dataset}:1|{model}|seed={seed}|split={split}"
                    coord_rows.append({"run_id": run_id, "dataset_id": dataset, "task_id": f"{dataset}:1", "model_id": model, "simple_regime": "quantized", "minority_survival_cliffiness": cliff, "minority_survival_auc": 0.7, "breadth": 0.2, "elevation": 0.8, "coord_pc1": c1, "coord_pc2": 0.2, "coord_pc3": 0.3, "coord_pc4": 0.4, "coord_pc5": 0.5})
                    feat_rows.append({"run_id": run_id, "n_score_groups": 2, "support_unique_score_ratio": 0.1, "group_mass_entropy": 0.2, "group_mass_gini": 0.3, "group_mass_hhi": 0.4, "top1_group_mass": 0.6, "top3_group_mass": 0.8, "top5_group_mass": 0.9, "effective_support_groups": 3.0, "n_support_islands": 1, "largest_island_mass": 0.7, "island_entropy": 0.2, "island_gini": 0.1, "small_island_fraction": 0.0, "largest_to_median_group_ratio": 2.0, "persistence_weighted_mass": 0.5, "fragile_support_mass": 0.1, "stable_support_mass": 0.6, "n_unique_positive_posteriors": 2, "effective_posterior_alphabet": 3.0, "posterior_hhi": 0.4, "largest_posterior_mass": 0.6, "posterior_entropy": 0.2, "mean_posterior_gap": 0.1, "max_posterior_gap": 0.2, "mass_at_largest_gap": 0.3, "mass_above_largest_gap": 0.4, "support_redundancy": 0.4, "mass_outside_top3": 0.2, "mass_outside_top5": 0.1, "support_fragility_index": 0.2, "broad_allocator_index": 0.3, "concentrated_allocator_index": 0.4, "quantized_allocator_index": 0.5, "continuous_allocator_index": 0.1, "fragmented_allocator_index": 0.2, "redundant_allocator_index": 0.3, "hierarchical_allocator_index": 0.4})
    coords = tmp_path / "coords.csv"
    feats = tmp_path / "features.csv"
    pd.DataFrame(coord_rows).to_csv(coords, index=False)
    pd.DataFrame(feat_rows).to_csv(feats, index=False)
    return coords, feats


def test_capture_bagging_pairs_and_validation(tmp_path):
    module = _load_module()
    coords, feats = _fixtures(tmp_path)
    data = module.load_source(coords, feats)
    paired = module.capture_bagging_pairs(data)
    issues, summary = module.validate_pairs(paired)
    assert summary["n_paired_vectors"] == 8
    assert summary["minimum_success"] is True
    assert summary["validation_passed"] is True
    assert set(paired["stage"]) == {"before", "after"}
    assert set([f"coord_{i}" for i in range(1, 6)]).issubset(paired.columns)
    assert issues[issues["status"] == "fail"].empty


def test_write_capture_outputs(tmp_path):
    module = _load_module()
    coords, feats = _fixtures(tmp_path)
    outputs = module.write_capture(coords, feats, tmp_path / "out")
    names = {p.name for p in outputs}
    assert "accessibility_paired_interventions.csv" in names
    assert "accessibility_paired_intervention_summary.json" in names
    assert "accessibility_paired_intervention_validation.md" in names
    csv_path = [p for p in outputs if p.name == "accessibility_paired_interventions.csv"][0]
    df = pd.read_csv(csv_path)
    assert len(df) == 16
    assert df["pair_id"].nunique() == 8
