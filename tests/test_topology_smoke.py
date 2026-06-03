import importlib.util
from pathlib import Path


def _load_script_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_topology_smoke.py"
    spec = importlib.util.spec_from_file_location("run_topology_smoke", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load run_topology_smoke module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_topology_rows_shapes_output():
    module = _load_script_module()
    assert "effective_k" in module.OUTPUT_COLUMNS

    rows = module.build_topology_rows(
        dataset_id="synthetic-gaussian-skew-100-to-1",
        model_name="mlp_bce",
        objective="bce",
        seed=0,
        split_id=0,
        auroc=0.8,
        average_precision=0.3,
        persistence=0.5,
        positive_score_entropy=0.7,
        requested_k_values=[5, 12],
        topology_by_k={
            5: {
                "n_positive": 10,
                "n_components": 2,
                "giant_component_fraction": 0.8,
                "mean_component_size": 5.0,
                "median_component_size": 5.0,
                "component_entropy": 0.2,
                "isolated_positive_fraction": 0.1,
            },
            12: {
                "n_positive": 10,
                "n_components": 1,
                "giant_component_fraction": 1.0,
                "mean_component_size": 10.0,
                "median_component_size": 10.0,
                "component_entropy": 0.0,
                "isolated_positive_fraction": 0.0,
            },
        },
    )

    assert len(rows) == 2
    row = rows[0]
    assert row["dataset_id"] == "synthetic-gaussian-skew-100-to-1"
    assert row["model_name"] == "mlp_bce"
    assert row["objective"] == "bce"
    assert row["k"] == 5
    assert row["effective_k"] == 5
    assert row["k_capped"] is False
    assert row["auroc"] == 0.8
    assert row["average_precision"] == 0.3
    assert row["persistence"] == 0.5
    assert row["positive_score_entropy"] == 0.7
    assert row["n_positive"] == 10
    assert row["n_components"] == 2
    assert row["giant_component_fraction"] == 0.8
    assert row["mean_component_size"] == 5.0
    assert row["median_component_size"] == 5.0
    assert row["component_entropy"] == 0.2
    assert row["isolated_positive_fraction"] == 0.1

    capped = rows[1]
    assert capped["k"] == 12
    assert capped["effective_k"] == 9
    assert capped["k_capped"] is True
