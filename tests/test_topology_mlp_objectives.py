import importlib.util
from pathlib import Path


def _load_script_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_topology_mlp_objectives.py"
    spec = importlib.util.spec_from_file_location("run_topology_mlp_objectives", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load run_topology_mlp_objectives module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_objective_model_ids_uses_aliases(monkeypatch):
    module = _load_script_module()
    monkeypatch.setattr(module, "available_model_ids", lambda: ["mlp_bce", "mlp_weighted"])

    resolved = module.resolve_objective_model_ids()

    assert resolved == [
        ("mlp_bce", "bce"),
        ("mlp_weighted", "weighted_bce"),
    ]


def test_failure_rows_have_k_and_effective_k():
    module = _load_script_module()
    assert "minority_survival_auc" in module.OUTPUT_COLUMNS
    assert "minority_survival_cliffiness" in module.OUTPUT_COLUMNS
    assert "topology_mode" in module.OUTPUT_COLUMNS
    assert "radius_quantile" in module.OUTPUT_COLUMNS
    assert "radius" in module.OUTPUT_COLUMNS
    assert "positive_unique_score_ratio" in module.OUTPUT_COLUMNS
    assert "positive_score_gini_or_concentration_index" in module.OUTPUT_COLUMNS

    rows = module._failure_rows(
        dataset_id="synthetic-gaussian-skew-100-to-1",
        model_name="mlp_bce",
        objective="bce",
        seed=0,
        requested_k_values=[1, 9],
        radius_quantiles=[0.1],
        topology_mode="both",
        n_positive=5,
        reason="fit broke",
    )

    assert len(rows) == 3
    assert rows[0]["topology_mode"] == "knn"
    assert rows[0]["k"] == 1
    assert rows[0]["effective_k"] == 1
    assert rows[0]["fit_failed"] is True
    assert rows[1]["k"] == 9
    assert rows[1]["effective_k"] == 4
    assert rows[1]["k_capped"] is True
    assert rows[2]["topology_mode"] == "radius"
    assert rows[2]["k"] == ""
    assert rows[2]["effective_k"] == ""
    assert rows[2]["radius_quantile"] == 0.1
    assert str(rows[0]["minority_survival_auc"]) == "nan"
    assert str(rows[0]["minority_survival_cliffiness"]) == "nan"


def test_build_objective_topology_rows_supports_both_modes():
    module = _load_script_module()
    rows = module.build_objective_topology_rows(
        dataset_id="synthetic-gaussian-skew-100-to-1",
        model_name="mlp_bce",
        objective="bce",
        seed=0,
        split_id=0,
        auroc=0.8,
        average_precision=0.3,
        persistence=0.5,
        positive_score_entropy=0.7,
        k_values=[1],
        topology_by_k={
            1: {
                "n_positive": 4,
                "n_components": 2,
                "giant_component_fraction": 0.5,
                "mean_component_size": 2.0,
                "median_component_size": 2.0,
                "component_entropy": 0.69,
                "isolated_positive_fraction": 0.0,
            }
        },
        radius_by_quantile={0.1: 0.25},
        topology_by_radius={
            0.25: {
                "n_positive": 4,
                "n_components": 3,
                "giant_component_fraction": 0.5,
                "mean_component_size": 4.0 / 3.0,
                "median_component_size": 1.0,
                "component_entropy": 1.04,
                "isolated_positive_fraction": 0.5,
            }
        },
        topology_mode="both",
    )

    assert len(rows) == 2
    assert rows[0]["topology_mode"] == "knn"
    assert rows[0]["k"] == 1
    assert rows[0]["radius"] == ""
    assert rows[1]["topology_mode"] == "radius"
    assert rows[1]["k"] == ""
    assert rows[1]["radius_quantile"] == 0.1
    assert rows[1]["radius"] == 0.25
