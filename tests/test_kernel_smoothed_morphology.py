import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_kernel_smoothed_morphology.py"
    spec = importlib.util.spec_from_file_location("report_kernel_smoothed_morphology", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_kernel_smoothed_morphology module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _family_rows(model_prefix: str, offset: float):
    rows = []
    for seed in range(8):
        breadth = 0.1 + 0.1 * seed + offset
        elevation = 0.9 - 0.08 * seed + offset
        cliff = 0.2 + (breadth - 0.45) ** 2 + 0.2 * (1.0 - elevation)
        rows.append(
            {
                "model_id": f"{model_prefix}_{seed % 2}",
                "seed": seed,
                "breadth": breadth,
                "elevation": elevation,
                "minority_survival_auc": 0.3 + 0.6 * elevation - 0.05 * breadth,
                "minority_survival_cliffiness": cliff,
                "auroc": 0.7 + 0.01 * seed,
                "average_precision": 0.2 + 0.01 * seed,
            }
        )
    return rows


def test_kernel_smoother_evaluation_on_tiny_data(tmp_path):
    module = _load_module()
    paths = []
    for filename, offset in [("density_threshold_sweep.csv", 0.0), ("fragmentation_sweep.csv", 0.05), ("allocation_trajectory.csv", 0.10)]:
        path = tmp_path / filename
        df = pd.DataFrame(_family_rows(filename, offset))
        if filename == "allocation_trajectory.csv":
            df = df.rename(columns={"model_id": "objective"})
        df.to_csv(path, index=False)
        paths.append(path)
    df, _ = module.load_inputs(paths)

    perf, predictions = module.evaluate_smoothers(df, gamma=0.5)

    assert {"linear_morphology", "kernel_morphology"}.issubset(set(perf["model"]))
    assert "minority_survival_cliffiness_kernel_morphology" in predictions


def test_field_grid_contains_local_quantities(tmp_path):
    module = _load_module()
    path = tmp_path / "density_threshold_sweep.csv"
    pd.DataFrame(_family_rows("m", 0.0)).to_csv(path, index=False)
    df, _ = module.load_inputs([path])

    field, _, _, _ = module.build_field_grid(df, "minority_survival_cliffiness", gamma=0.5, grid_size=12)

    assert "predicted_minority_survival_cliffiness" in field.columns
    assert "gradient_minority_survival_cliffiness" in field.columns
    assert "laplacian_minority_survival_cliffiness" in field.columns
    assert "local_sample_density" in field.columns
    assert "local_residual_variance_minority_survival_cliffiness" in field.columns


def test_kernel_report_generation(tmp_path):
    module = _load_module()
    paths = []
    for filename, offset in [("density_threshold_sweep.csv", 0.0), ("fragmentation_sweep.csv", 0.05), ("weighted_dropout_sweep_dense.csv", 0.10)]:
        path = tmp_path / filename
        df = pd.DataFrame(_family_rows(filename, offset))
        if filename == "weighted_dropout_sweep_dense.csv":
            df = df.drop(columns=["model_id"]).assign(dropout_rate=0.1)
        df.to_csv(path, index=False)
        paths.append(path)
    output = tmp_path / "kernel_summary.md"

    outputs = module.write_report(paths, output, gamma=0.5, grid_size=12)
    report = output.read_text(encoding="utf-8")

    assert "# Kernel-Smoothed Morphology Field" in report
    assert "## Linear vs Kernel Performance" in report
    assert "## Local Field Diagnostics" in report
    assert "Is cliffiness nonlinear in morphology space?" in report
    assert len(outputs) == 4
    for path in outputs:
        assert path.exists()
