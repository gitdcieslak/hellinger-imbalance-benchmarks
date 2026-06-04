import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_run_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_density_separability_factorial.py"
    spec = importlib.util.spec_from_file_location("run_density_separability_factorial", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load run_density_separability_factorial module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_report_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_density_separability_factorial.py"
    spec = importlib.util.spec_from_file_location("report_density_separability_factorial", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_density_separability_factorial module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_df():
    rows = []
    for density_level, minority_cov in [("high_density", 0.1), ("low_density", 1.0)]:
        for separability_level, centroid_distance in [("high_separability", 4.0), ("low_separability", 1.0)]:
            for model_id, dropout_offset in [("mlp_weighted_bce", 0.0), ("mlp_weighted_bce_dropout_0_1", -0.05)]:
                for seed in [0, 1]:
                    rows.append(
                        {
                            "density_level": density_level,
                            "separability_level": separability_level,
                            "density_label": f"cov_{minority_cov:.2f}".replace(".", "_"),
                            "separability_label": f"dist_{centroid_distance:.1f}".replace(".", "_"),
                            "minority_cov": minority_cov,
                            "centroid_distance": centroid_distance,
                            "model_id": model_id,
                            "registry_model_id": model_id,
                            "seed": seed,
                            "skew_ratio": 100,
                            "minority_count": 100,
                            "auroc": 0.8 + 0.02 * centroid_distance,
                            "average_precision": 0.2 + 0.02 * centroid_distance,
                            "minority_survival_auc": 0.6 + 0.05 * centroid_distance - 0.05 * minority_cov + dropout_offset,
                            "minority_survival_cliffiness": 0.2 + 0.2 * minority_cov - 0.02 * centroid_distance,
                            "minority_survival_max_drop": 0.2,
                            "minority_survival_effective_drop_count": 2.0,
                            "breadth": 0.4 + 0.1 * minority_cov,
                            "effective_breadth": 1.4 + 0.1 * minority_cov,
                            "elevation": 0.5 + 0.1 * centroid_distance - 0.1 * minority_cov,
                            "peak_concentration": 0.6,
                            "mean_positive_knn_distance": minority_cov,
                            "positive_density_proxy": 1.0 / minority_cov,
                            "local_positive_ratio_mean": 0.8,
                            "local_label_entropy_mean": 0.2,
                        }
                    )
    return pd.DataFrame(rows)


def test_grid_construction():
    module = _load_run_module()

    grid = module.construct_grid(
        ["high_density", "low_density"],
        ["high_separability", "low_separability"],
        ["mlp_weighted_bce", "mlp_weighted_bce_dropout_0_1"],
        [0, 1],
    )

    assert len(grid) == 16
    assert grid[0]["minority_cov"] == 0.1
    assert grid[0]["centroid_distance"] == 4.0
    assert grid[0]["density_label"] == "cov_0_10"
    assert grid[0]["separability_label"] == "dist_4_0"


def test_numeric_grid_construction_and_parsing():
    module = _load_run_module()

    grid = module.construct_grid(
        None,
        None,
        ["mlp_weighted_bce"],
        module.parse_int_list("0-1"),
        minority_covs=module.parse_float_list("0.05,0.10"),
        centroid_distances=module.parse_float_list("4.0,0.5"),
    )

    assert len(grid) == 8
    assert module.parse_int_list("0-2,5") == [0, 1, 2, 5]
    assert module.numeric_label("cov", 0.05) == "cov_0_05"
    assert module.numeric_label("cov", 0.10) == "cov_0_10"
    assert module.numeric_label("dist", 0.5) == "dist_0_5"
    assert module.numeric_label("dist", 4.0) == "dist_4_0"
    assert {row["density_label"] for row in grid} == {"cov_0_05", "cov_0_10"}
    assert {row["separability_label"] for row in grid} == {"dist_4_0", "dist_0_5"}


def test_categorical_grid_backward_compatibility():
    module = _load_run_module()

    grid = module.construct_grid(
        ["high_density"],
        ["low_separability"],
        ["mlp_weighted_bce"],
        [0],
    )

    assert grid[0]["density_level"] == "high_density"
    assert grid[0]["separability_level"] == "low_separability"
    assert grid[0]["minority_cov"] == 0.1
    assert grid[0]["centroid_distance"] == 1.0


def test_synthetic_dataset_generation_shape():
    module = _load_run_module()

    dataset = module.make_factorial_dataset(
        minority_cov=0.1,
        centroid_distance=4.0,
        seed=0,
        skew_ratio=10,
        minority_count=5,
        n_features=3,
    )

    assert dataset.X.shape == (55, 3)
    assert dataset.y.shape == (55,)
    assert int(dataset.y.sum()) == 5


def test_output_row_construction():
    module = _load_run_module()
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.05, 0.2, 0.6, 0.9])

    row = module.build_output_row(
        density_level="high_density",
        separability_level="high_separability",
        density_label="cov_0_10",
        separability_label="dist_4_0",
        minority_cov=0.1,
        centroid_distance=4.0,
        model_id="mlp_weighted_bce",
        registry_model_id="mlp_weighted",
        seed=0,
        skew_ratio=100,
        minority_count=100,
        y_true=y,
        y_score=scores,
        density_metrics={
            "mean_positive_knn_distance": 0.1,
            "positive_density_proxy": 10.0,
            "local_positive_ratio_mean": 0.5,
            "local_label_entropy_mean": 0.5,
        },
    )

    assert set(row) == set(module.OUTPUT_COLUMNS)
    assert row["density_level"] == "high_density"
    assert row["density_label"] == "cov_0_10"
    assert row["registry_model_id"] == "mlp_weighted"


def test_factor_interaction_feature_construction():
    module = _load_report_module()

    featured = module.factor_features(_tiny_df())

    assert "density_numeric" in featured.columns
    assert "separability_numeric" in featured.columns
    assert "density_x_separability" in featured.columns
    assert "is_dropout" in featured.columns
    assert featured["density_x_separability"].iloc[0] == featured["minority_cov"].iloc[0] * featured["centroid_distance"].iloc[0]


def test_report_generation_on_tiny_csv(tmp_path):
    module = _load_report_module()
    input_path = tmp_path / "tiny.csv"
    output_md = tmp_path / "summary.md"
    _tiny_df().to_csv(input_path, index=False)

    paths = module.write_report(input_path, output_md)
    report = output_md.read_text(encoding="utf-8")

    assert "## Factor Effects" in report
    assert "## Continuous Factor Correlations" in report
    assert "Does density independently affect elevation?" in report
    assert len(paths) == 8
    for path in paths:
        assert path.exists()


def test_report_generation_with_numeric_grid(tmp_path):
    module = _load_report_module()
    input_path = tmp_path / "numeric.csv"
    output_md = tmp_path / "numeric_summary.md"
    df = _tiny_df().copy()
    df["density_level"] = df["density_label"]
    df["separability_level"] = df["separability_label"]
    df.to_csv(input_path, index=False)

    paths = module.write_report(input_path, output_md)
    report = output_md.read_text(encoding="utf-8")

    assert "minority_cov" in report
    assert "centroid_distance" in report
    assert len(paths) == 8
