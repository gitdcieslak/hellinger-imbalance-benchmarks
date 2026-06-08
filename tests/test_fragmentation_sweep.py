import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_run_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_fragmentation_sweep.py"
    spec = importlib.util.spec_from_file_location("run_fragmentation_sweep", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load run_fragmentation_sweep module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_report_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_fragmentation_sweep.py"
    spec = importlib.util.spec_from_file_location("report_fragmentation_sweep", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_fragmentation_sweep module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _df():
    rows = []
    for n_islands in [1, 10, 100]:
        for model_id, offset in [("mlp_weighted_bce", 0.0), ("mlp_weighted_bce_dropout_0_1", -0.05)]:
            for seed in [0, 1]:
                rows.append(
                    {
                        "n_islands": n_islands,
                        "model_id": model_id,
                        "registry_model_id": model_id,
                        "seed": seed,
                        "skew_ratio": 100,
                        "minority_count": 100,
                        "n_features": 10,
                        "island_cov": 0.16,
                        "centroid_radius": 3.0,
                        "auroc": 0.9 - 0.001 * n_islands,
                        "average_precision": 0.5 - 0.002 * n_islands,
                        "minority_survival_auc": 0.9 - 0.003 * n_islands + offset,
                        "minority_survival_cliffiness": 0.2 + 0.002 * n_islands,
                        "minority_survival_max_drop": 0.2,
                        "minority_survival_effective_drop_count": 2.0,
                        "breadth": 0.3 + 0.003 * n_islands,
                        "effective_breadth": 1.3 + 0.003 * n_islands,
                        "elevation": 0.9 - 0.002 * n_islands,
                        "peak_concentration": 0.9,
                        "mean_positive_knn_distance": 0.5 + 0.001 * n_islands,
                        "positive_density_proxy": 1.0 / (0.5 + 0.001 * n_islands),
                        "local_positive_ratio_mean": 0.8,
                        "local_label_entropy_mean": 0.2,
                        "minority_cluster_count": n_islands,
                        "positives_per_cluster": 100.0 / n_islands,
                    }
                )
    return pd.DataFrame(rows)


def test_positive_distribution_evenly_splits_remainder():
    module = _load_run_module()

    counts = module.positives_per_island_counts(10, 4)

    assert counts.tolist() == [3, 3, 2, 2]
    assert int(counts.sum()) == 10


def test_fragmentation_dataset_shape():
    module = _load_run_module()

    dataset = module.make_fragmentation_dataset(
        n_islands=5,
        seed=0,
        skew_ratio=10,
        minority_count=20,
        n_features=4,
    )

    assert dataset.X.shape == (220, 4)
    assert dataset.y.shape == (220,)
    assert int(dataset.y.sum()) == 20
    assert dataset.positives_per_cluster == 4.0


def test_output_row_has_expected_columns():
    module = _load_run_module()
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.05, 0.2, 0.6, 0.9])

    row = module.build_output_row(
        n_islands=5,
        model_id="mlp_weighted_bce",
        registry_model_id="mlp_weighted",
        seed=0,
        skew_ratio=100,
        minority_count=100,
        n_features=10,
        island_cov=0.16,
        centroid_radius=3.0,
        y_true=y,
        y_score=scores,
        density_metrics={
            "mean_positive_knn_distance": 0.5,
            "positive_density_proxy": 2.0,
            "local_positive_ratio_mean": 0.5,
            "local_label_entropy_mean": 0.5,
        },
    )

    assert set(row) == set(module.OUTPUT_COLUMNS)
    assert row["minority_cluster_count"] == 5
    assert row["positives_per_cluster"] == 20.0


def test_report_contains_requested_answers():
    module = _load_report_module()

    report = module.build_fragmentation_report(_df())

    assert "## Density-Adjusted Correlations" in report
    assert "## Fragmentation Threshold Diagnostics" in report
    assert "Does fragmentation reduce survival AUC" in report
    assert "Does fragmentation increase cliffiness independently of density" in report
    assert "Does dropout help fragmented support or worsen survival" in report


def test_report_generation_on_tiny_csv(tmp_path):
    module = _load_report_module()
    input_path = tmp_path / "fragmentation.csv"
    output_md = tmp_path / "summary.md"
    output_survival = tmp_path / "survival.png"
    output_cliff = tmp_path / "cliff.png"
    output_morph = tmp_path / "morph.png"
    _df().to_csv(input_path, index=False)

    paths = module.write_report(input_path, output_md, output_survival, output_cliff, output_morph)

    assert len(paths) == 4
    for path in paths:
        assert path.exists()
