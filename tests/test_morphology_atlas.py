import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_morphology_atlas.py"
    spec = importlib.util.spec_from_file_location("report_morphology_atlas", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_morphology_atlas module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    rows = []
    specs = [("cart", 0.15, 0.12, 0.05), ("random_forest", 1.1, 0.1, 0.5), ("lightgbm", 0.55, 0.65, 0.3)]
    seed = 0
    for model, b0, e0, c0 in specs:
        for i in range(18):
            breadth = b0 + 0.02 * (i % 6)
            elevation = e0 + 0.02 * (i // 6)
            rows.append(
                {
                    "model_id": model,
                    "seed": seed,
                    "skew_ratio": 100,
                    "minority_count": 50,
                    "fit_failed": False,
                    "failure_reason": "",
                    "breadth": breadth,
                    "elevation": elevation,
                    "minority_survival_auc": 0.2 + elevation,
                    "minority_survival_cliffiness": c0 + 0.05 * (i % 3),
                }
            )
            seed += 1
    return rows


def test_feature_space_preparation(tmp_path):
    module = _load_module()
    path = tmp_path / "classical.csv"
    pd.DataFrame(_rows()).to_csv(path, index=False)
    df = module.load_classical_atlas(path)

    enriched, features, x_raw, x_pca, x_umap, notes = module.prepare_feature_space(df)

    assert not enriched.empty
    assert "vertical_slack" in enriched.columns
    assert "density_separability" in features
    assert x_raw.shape[0] == len(df)
    assert x_pca.shape[0] == len(df)
    assert x_umap.shape[1] == 2
    assert notes


def test_clustering_and_stability_outputs(tmp_path):
    module = _load_module()
    path = tmp_path / "classical.csv"
    pd.DataFrame(_rows()).to_csv(path, index=False)
    df = module.load_classical_atlas(path)
    _, _, _, x_pca, _, _ = module.prepare_feature_space(df)

    labels, notes = module.run_clustering(x_pca, n_clusters=3)
    stability = module.stability_report(labels, x_pca)

    assert {"hdbscan", "gmm", "kmeans", "spectral"}.issubset(labels.columns)
    assert not stability.empty
    assert notes


def test_atlas_artifact_generation(tmp_path):
    module = _load_module()
    path = tmp_path / "classical_allocator_morphology.csv"
    pd.DataFrame(_rows()).to_csv(path, index=False)

    outputs = module.write_atlas(path, tmp_path)

    names = {p.name for p in outputs}
    assert "morphology_atlas_clusters.csv" in names
    assert "cluster_summary_table.md" in names
    assert "cluster_stability_report.md" in names
    assert "within_cluster_equation_scores.json" in names
    assert "regime_phase_diagram.png" in names
    assert "cluster_cliffiness_distribution.png" in names
    for path in outputs:
        assert path.exists()
