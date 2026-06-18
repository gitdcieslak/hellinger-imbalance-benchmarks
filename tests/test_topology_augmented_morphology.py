import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_topology_augmented_morphology.py"
    spec = importlib.util.spec_from_file_location("report_topology_augmented_morphology", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_topology_augmented_morphology module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(offset=0.0, topology=False):
    rows = []
    for seed in range(6):
        breadth = 0.1 + 0.08 * seed + offset
        elevation = 0.9 - 0.06 * seed + offset
        row = {
            "model_id": "m0" if seed % 2 == 0 else "m1",
            "seed": seed,
            "breadth": breadth,
            "elevation": elevation,
            "minority_survival_auc": 0.4 + 0.5 * elevation,
            "minority_survival_cliffiness": 0.2 + 0.3 * breadth,
            "auroc": 0.7 + 0.01 * seed,
            "average_precision": 0.2 + 0.01 * seed,
        }
        if topology:
            row.update({"n_islands": seed + 1, "minority_count": 100, "positives_per_cluster": 100 / (seed + 1)})
        rows.append(row)
    return rows


def test_input_normalization_aliases(tmp_path):
    module = _load_module()
    path = tmp_path / "weighted_dropout_sweep_dense.csv"
    pd.DataFrame(_rows()).drop(columns=["model_id"]).rename(columns={"breadth": "positive_histogram_entropy", "elevation": "positive_top_bin_mass"}).assign(dropout_rate=0.1).to_csv(path, index=False)

    df, messages = module.normalize_result_file(path)

    assert df is not None
    assert "breadth" in df.columns
    assert "elevation" in df.columns
    assert df["model_id"].iloc[0].startswith("dropout_")
    assert any("used:" in message for message in messages)


def test_fragmentation_topology_derivation(tmp_path):
    module = _load_module()
    path = tmp_path / "fragmentation_sweep.csv"
    pd.DataFrame(_rows(topology=True)).to_csv(path, index=False)

    df, _ = module.normalize_result_file(path)

    assert df is not None
    assert "largest_island_fraction" in df.columns
    assert "island_entropy" in df.columns
    assert "island_gini_or_concentration" in df.columns
    assert df["largest_island_fraction"].notna().all()


def test_missing_topology_features_survive_pipeline(tmp_path):
    module = _load_module()
    path = tmp_path / "density_threshold_sweep.csv"
    pd.DataFrame(_rows()).to_csv(path, index=False)
    df, _ = module.load_inputs([path])

    perf, _ = module.fit_models(df)

    assert not perf.empty
    assert "topology" in set(perf["feature_set"])


def test_feature_set_construction_and_improvement_classification():
    module = _load_module()

    assert "morphology_topology" in module.FEATURE_SETS
    assert "n_islands" in module.FEATURE_SETS["topology"][0]
    assert module.improvement_classification(0.05) == "weak support"
    assert module.improvement_classification(0.15) == "moderate support"
    assert module.improvement_classification(0.35) == "strong support"


def test_report_generation_on_tiny_synthetic_csvs(tmp_path):
    module = _load_module()
    paths = []
    for filename, offset, topology in [
        ("density_threshold_sweep.csv", 0.0, False),
        ("fragmentation_sweep.csv", 0.1, True),
        ("mlp_objectives_topology.csv", 0.2, False),
    ]:
        path = tmp_path / filename
        df = pd.DataFrame(_rows(offset, topology))
        if filename == "mlp_objectives_topology.csv":
            df = pd.concat([df.assign(k=3), df.assign(k=5)], ignore_index=True).rename(columns={"model_id": "objective"})
        df.to_csv(path, index=False)
        paths.append(path)
    output_md = tmp_path / "topology_augmented_morphology_summary.md"

    outputs = module.write_report(paths, output_md, bootstrap_samples=5)
    report = output_md.read_text(encoding="utf-8")

    assert "# Topology-Augmented Morphology" in report
    assert "## Topology Feature Coverage" in report
    assert "## Cliffiness Improvement Test" in report
    assert "Does topology improve prediction of cliffiness" in report
    assert len(outputs) == 5
    for path in outputs:
        assert path.exists()
