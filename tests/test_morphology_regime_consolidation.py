import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_morphology_regime_consolidation.py"
    spec = importlib.util.spec_from_file_location("report_morphology_regime_consolidation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_morphology_regime_consolidation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atlas_rows():
    rows = []
    cluster_specs = [(0, "cart", 0.2, 0.05, 0.0), (1, "rf", 1.2, 0.1, 0.5), (2, "lgbm", 0.7, 0.6, 0.25), (3, "xgb", 1.5, 0.25, 0.4)]
    seed = 0
    for cluster, family, b0, e0, c0 in cluster_specs:
        for i in range(12):
            rows.append(
                {
                    "hdbscan_cluster": cluster,
                    "allocation_family": family,
                    "breadth": b0 + 0.01 * (i % 4),
                    "elevation": e0 + 0.01 * (i // 4),
                    "minority_survival_cliffiness": c0 + 0.02 * (i % 3),
                    "minority_survival_auc": e0 + 0.1,
                    "vertical_slack": 0.3 - e0,
                    "normalized_position_between_envelopes": min(1.0, e0 + 0.2),
                }
            )
            seed += 1
    return rows


def test_centroids_and_similarity():
    module = _load_module()
    df = pd.DataFrame(_atlas_rows())

    centroids = module.cluster_centroids(df)
    matrices = module.similarity_matrices(centroids)

    assert len(centroids) == 4
    assert "euclidean" in matrices
    assert matrices["euclidean"].shape == (4, 4)


def test_evaluate_k_and_compression():
    module = _load_module()
    df = pd.DataFrame(_atlas_rows())
    centroids = module.cluster_centroids(df)

    evaluation, assignments = module.evaluate_k_range(df, centroids, k_values=range(2, 4))
    compression = module.compression_curve(df, assignments)

    assert set(evaluation["k"]) == {2, 3}
    assert "cv_accuracy" in compression.columns
    assert "cluster_id" in set(compression["representation"])


def test_report_generation(tmp_path):
    module = _load_module()
    atlas = tmp_path / "morphology_atlas_clusters.csv"
    pd.DataFrame(_atlas_rows()).to_csv(atlas, index=False)

    outputs = module.write_report(tmp_path, atlas_csv=atlas)

    names = {path.name for path in outputs}
    assert "morphology_regime_consolidation.md" in names
    assert "cluster_distance_heatmap.png" in names
    assert "cluster_dendrogram.png" in names
    assert "regime_compression_curve.png" in names
    for path in outputs:
        assert path.exists()
