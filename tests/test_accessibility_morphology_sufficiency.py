import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_accessibility_morphology_sufficiency.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_morphology_sufficiency", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _curve_file(tmp_path: Path) -> Path:
    rows = []
    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
    for dataset in ["a", "b", "c", "d", "e"]:
        for model_id in ["cart", "rf"]:
            for seed in [0, 1, 2, 3]:
                cliff = 0.15 + 0.1 * seed if model_id == "rf" else 0.65 + 0.06 * seed
                survival = 0.85 - 0.2 * cliff
                breadth = 0.7 - 0.4 * cliff
                elevation = 0.2 + 0.7 * cliff
                run_id = f"{dataset}:1|{model_id}|seed={seed}|split={seed}"
                for threshold in thresholds:
                    if threshold == 0.0:
                        reachability = 1.0
                    elif model_id == "cart" and threshold >= 0.5:
                        reachability = max(0.0, 1.0 - cliff)
                    else:
                        reachability = max(0.0, 1.0 - cliff * threshold)
                    rows.append(
                        {
                            "run_id": run_id,
                            "dataset_id": dataset,
                            "task_id": f"{dataset}:1",
                            "model_id": model_id,
                            "seed": seed,
                            "threshold": threshold,
                            "minority_reachability": reachability,
                            "minority_survival_auc": survival,
                            "minority_survival_cliffiness": cliff,
                            "breadth": breadth,
                            "elevation": elevation,
                            "regime_id": 1,
                        }
                    )
    path = tmp_path / "curves.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_dataset_builds_feature_blocks(tmp_path):
    module = _load_module()
    path = _curve_file(tmp_path)

    meta, morphology, topology, thresholds = module.load_dataset([path])

    assert len(meta) == 40
    assert len(morphology) == len(meta)
    assert len(topology) == len(meta)
    assert len(thresholds) == 5
    assert {"breadth", "elevation", "effective_support"}.issubset(morphology.columns)
    assert {"giant_component_fraction", "component_entropy", "curve_roughness"}.issubset(topology.columns)


def test_model_scores_and_partition_are_valid(tmp_path):
    module = _load_module()
    meta, morphology, topology, _ = module.load_dataset([_curve_file(tmp_path)])

    scores, predictions, _ = module.model_scores(meta, morphology, topology)
    partition = module.variance_partition(scores)

    assert set(scores["model_spec"]) == {"topology_only", "morphology_only", "topology_plus_morphology"}
    assert predictions[["topology_only", "morphology_only", "topology_plus_morphology"]].notna().any().all()
    assert set(partition["component"]) == {"topology_unique", "morphology_unique", "shared_topology_morphology", "unexplained"}
    assert partition["variance_fraction"].between(0, 1).all()


def test_counterfactual_pairs_nonempty(tmp_path):
    module = _load_module()
    meta, morphology, topology, _ = module.load_dataset([_curve_file(tmp_path)])

    pairs = module.counterfactual_pairs(meta, morphology, topology, max_pairs=20)
    summary = module.counterfactual_summary(pairs)

    assert not pairs.empty
    assert {"pair_type", "cliffiness_abs_delta"}.issubset(pairs.columns)
    assert not summary.empty


def test_write_report_creates_deliverables(tmp_path):
    module = _load_module()
    output_dir = tmp_path / "out"

    outputs = module.write_report([_curve_file(tmp_path)], output_dir)

    names = {p.name for p in outputs}
    assert "accessibility_morphology_sufficiency.md" in names
    assert "morphology_model_scores.csv" in names
    assert "morphology_variance_partition.csv" in names
    assert "accessibility_morphology_sufficiency_summary.json" in names
    assert all(p.exists() for p in outputs)
