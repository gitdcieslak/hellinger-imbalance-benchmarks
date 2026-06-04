import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_run_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_information_density_sweep.py"
    spec = importlib.util.spec_from_file_location("run_information_density_sweep", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load run_information_density_sweep module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_report_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_information_density_sweep.py"
    spec = importlib.util.spec_from_file_location("report_information_density_sweep", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_information_density_sweep module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _df():
    rows = []
    for regime, density, entropy, clusters in [
        ("compact_dense", 2.0, 0.1, 1),
        ("fragmented_islands_20", 0.5, 0.2, 20),
        ("boundary_mixed", 1.5, 0.6, 1),
    ]:
        for model_id, offset in [("mlp_weighted_bce", 0.0), ("mlp_weighted_bce_dropout_0_1", 0.05)]:
            for seed in [0, 1]:
                rows.append(
                    {
                        "support_regime": regime,
                        "model_id": model_id,
                        "registry_model_id": model_id,
                        "seed": seed,
                        "skew_ratio": 100,
                        "minority_count": 100,
                        "auroc": 0.7 + offset,
                        "average_precision": 0.2 + offset,
                        "minority_survival_auc": 0.4 + 0.1 * density + offset,
                        "minority_survival_cliffiness": 0.2 + 0.02 * clusters + entropy,
                        "minority_survival_max_drop": 0.2,
                        "minority_survival_effective_drop_count": 2.0,
                        "breadth": 0.3 + 0.02 * clusters,
                        "effective_breadth": 1.3 + 0.02 * clusters,
                        "elevation": 0.5 + 0.1 * density + offset,
                        "peak_concentration": 0.6,
                        "mean_positive_knn_distance": 1.0 / density,
                        "median_positive_knn_distance": 1.0 / density,
                        "positive_density_proxy": density,
                        "local_positive_ratio_mean": 1.0 - entropy,
                        "local_label_entropy_mean": entropy,
                        "minority_cluster_count": clusters,
                        "positives_per_cluster": 100.0 / clusters,
                    }
                )
    return pd.DataFrame(rows)


def test_support_regime_generation_separates_density_and_ambiguity():
    module = _load_run_module()
    compact = module.make_support_dataset("compact_dense", seed=0)
    boundary = module.make_support_dataset("boundary_mixed", seed=0)
    fragmented = module.make_support_dataset("fragmented_islands_20", seed=0)

    assert compact.minority_cluster_count == 1
    assert boundary.minority_cluster_count == 1
    assert fragmented.minority_cluster_count == 20
    assert fragmented.positives_per_cluster == 5.0


def test_information_density_metrics_have_expected_keys():
    module = _load_run_module()
    X = np.array([[0.0], [0.1], [0.2], [2.0], [2.1], [2.2]])
    y = np.array([0, 0, 0, 1, 1, 1])

    metrics = module.information_density_metrics(
        X,
        y,
        minority_cluster_count=1,
        positives_per_cluster=3,
        positive_k=2,
        local_k=2,
    )

    assert metrics["mean_positive_knn_distance"] > 0.0
    assert metrics["positive_density_proxy"] > 0.0
    assert metrics["local_positive_ratio_mean"] >= 0.0
    assert metrics["local_label_entropy_mean"] >= 0.0


def test_output_row_construction_has_expected_columns():
    module = _load_run_module()
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.05, 0.2, 0.6, 0.9])
    density = {
        "mean_positive_knn_distance": 1.0,
        "median_positive_knn_distance": 1.0,
        "positive_density_proxy": 1.0,
        "local_positive_ratio_mean": 0.5,
        "local_label_entropy_mean": 0.5,
        "minority_cluster_count": 1,
        "positives_per_cluster": 50.0,
    }

    row = module.build_output_row(
        support_regime="compact_dense",
        model_id="mlp_weighted_bce",
        registry_model_id="mlp_weighted",
        seed=3,
        skew_ratio=100,
        minority_count=100,
        y_true=y,
        y_score=scores,
        density=density,
    )

    assert set(row) == set(module.OUTPUT_COLUMNS)
    assert row["support_regime"] == "compact_dense"
    assert row["registry_model_id"] == "mlp_weighted"


def test_information_density_report_contains_requested_answers():
    module = _load_report_module()
    report = module.build_information_density_report(_df())

    assert "## Regime Means" in report
    assert "## Density And Ambiguity Correlations" in report
    assert "Does lower local minority density reduce elevation?" in report
    assert "Does fragmented support increase cliffiness?" in report
    assert "Does global skew matter less than local information density?" in report
    assert "Are breadth/elevation better explained by density metrics than by model family?" in report
