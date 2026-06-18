import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_kernel_morphology_robustness.py"
    spec = importlib.util.spec_from_file_location("report_kernel_morphology_robustness", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_kernel_morphology_robustness module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prior_rows(offset: float = 0.0, n: int = 8):
    rows = []
    for seed in range(n):
        breadth = 0.1 + 0.08 * seed + offset
        elevation = 0.95 - 0.06 * seed + offset * 0.2
        rows.append(
            {
                "model_id": "mlp_bce" if seed % 2 == 0 else "mlp_weighted_bce",
                "seed": seed,
                "breadth": breadth,
                "elevation": elevation,
                "minority_survival_auc": 0.2 + 0.6 * elevation - 0.05 * breadth,
                "minority_survival_cliffiness": 0.1 + (breadth - 0.45) ** 2 + 0.2 * (1.0 - elevation),
            }
        )
    return rows


def _classical_rows(n: int = 8):
    rows = []
    for seed in range(n):
        breadth = 0.2 + 0.07 * seed
        elevation = 0.15 + 0.03 * seed
        rows.append(
            {
                "model_id": "cart" if seed % 2 == 0 else "random_forest",
                "seed": seed,
                "skew_ratio": 100,
                "minority_count": 50,
                "fit_failed": False,
                "failure_reason": "",
                "breadth": breadth,
                "elevation": elevation,
                "minority_survival_auc": 0.2 + 0.4 * elevation,
                "minority_survival_cliffiness": 0.2 + (breadth - 0.4) ** 2,
            }
        )
    return rows


def test_load_pooled_dataset_includes_classical(tmp_path):
    module = _load_module()
    prior = tmp_path / "density_threshold_sweep.csv"
    classical = tmp_path / "classical_allocator_morphology.csv"
    pd.DataFrame(_prior_rows()).to_csv(prior, index=False)
    pd.DataFrame(_classical_rows()).to_csv(classical, index=False)

    df, messages = module.load_pooled_dataset([prior], classical)

    assert not df.empty
    assert "classical_allocator" in set(df["experiment_family"])
    assert any("used classical input" in message for message in messages)


def test_evaluate_models_small_configuration(tmp_path):
    module = _load_module()
    prior = tmp_path / "density_threshold_sweep.csv"
    classical = tmp_path / "classical_allocator_morphology.csv"
    pd.DataFrame(_prior_rows(0.0, n=10)).to_csv(prior, index=False)
    pd.DataFrame(_classical_rows(n=10)).to_csv(classical, index=False)
    df, _ = module.load_pooled_dataset([prior], classical)

    results = module.evaluate_models(df, gammas=[0.1, 1.0], k_values=[3], schemes=["random_kfold", "group_experiment_family"])

    assert {"linear_regression", "ridge", "poly_degree_2", "rbf_nystroem_ridge", "knn_regressor"}.issubset(set(results["model_name"]))
    assert {"minority_survival_auc", "minority_survival_cliffiness"}.issubset(set(results["target"]))
    assert "train_test_gap" in results.columns


def test_robustness_interpretation_returns_label():
    module = _load_module()
    results = pd.DataFrame(
        [
            {"target": "minority_survival_cliffiness", "validation_scheme": "group_experiment_family", "model_family": "linear", "model_name": "linear_regression", "cv_r2": -0.1},
            {"target": "minority_survival_cliffiness", "validation_scheme": "group_experiment_family", "model_family": "rbf_kernel", "model_name": "rbf_nystroem_ridge", "cv_r2": 0.5},
            {"target": "minority_survival_cliffiness", "validation_scheme": "group_experiment_family", "model_family": "rbf_kernel", "model_name": "rbf_nystroem_ridge", "cv_r2": 0.55},
            {"target": "minority_survival_cliffiness", "validation_scheme": "group_experiment_family", "model_family": "polynomial", "model_name": "poly_degree_2", "cv_r2": 0.3},
            {"target": "minority_survival_cliffiness", "validation_scheme": "random_kfold", "model_family": "rbf_kernel", "model_name": "rbf_nystroem_ridge", "cv_r2": 0.7},
        ]
    )

    label, summary, bullets = module.robustness_interpretation(results)

    assert label in {"robust", "partially robust", "fragile"}
    assert not summary.empty
    assert any("Cliffiness nonlinear morphology" in bullet for bullet in bullets)


def test_report_generation_creates_required_plots(tmp_path):
    module = _load_module()
    prior = tmp_path / "density_threshold_sweep.csv"
    classical = tmp_path / "classical_allocator_morphology.csv"
    pd.DataFrame(_prior_rows(0.0, n=10)).to_csv(prior, index=False)
    pd.DataFrame(_classical_rows(n=10)).to_csv(classical, index=False)
    output = tmp_path / "kernel_morphology_robustness_summary.md"

    outputs = module.write_report(output, prior_inputs=[prior], classical_input=classical, gammas=[0.1, 1.0], k_values=[3], schemes=["random_kfold", "group_experiment_family"])
    report = output.read_text(encoding="utf-8")

    assert "# Kernel Morphology Robustness Check" in report
    assert "## RBF Bandwidth Sweep" in report
    assert "## Paper 2 Recommendation" in report
    assert len(outputs) == 5
    for path in outputs:
        assert path.exists()
