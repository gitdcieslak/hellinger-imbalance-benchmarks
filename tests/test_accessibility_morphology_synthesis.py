import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_accessibility_morphology_synthesis.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_morphology_synthesis", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_accessibility_morphology_synthesis module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(model: str, offset: float = 0.0, n: int = 8):
    rows = []
    for seed in range(n):
        breadth = 0.15 + 0.08 * seed + offset
        elevation = 0.90 - 0.06 * seed + 0.4 * offset
        rows.append(
            {
                "model_id": model if seed % 2 == 0 else "mlp_weighted_bce",
                "seed": seed,
                "breadth": breadth,
                "elevation": elevation,
                "minority_survival_auc": 0.30 + 0.50 * elevation - 0.05 * breadth,
                "minority_survival_cliffiness": 0.15 + (breadth - 0.45) ** 2 + 0.15 * (1.0 - elevation),
                "auroc": 0.7 + 0.01 * seed,
                "average_precision": 0.2 + 0.01 * seed,
            }
        )
    return rows


def test_morphology_power_and_claim_logic():
    module = _load_module()
    perf = pd.DataFrame(
        [
            {"target": "minority_survival_auc", "feature_set": "breadth_elevation", "cv_r2": 0.8},
            {"target": "minority_survival_auc", "feature_set": "pc1_pc2", "cv_r2": 0.82},
            {"target": "minority_survival_auc", "feature_set": "kernel_breadth_elevation", "cv_r2": 0.9},
            {"target": "minority_survival_cliffiness", "feature_set": "breadth_elevation", "cv_r2": -0.1},
            {"target": "minority_survival_cliffiness", "feature_set": "pc1_pc2", "cv_r2": -0.08},
            {"target": "minority_survival_cliffiness", "feature_set": "kernel_breadth_elevation", "cv_r2": 0.85},
        ]
    )

    power = module.morphology_explanatory_power(perf)
    claim = module.synthesis_claim(power)

    assert set(power["Target"]) == {"Survival AUC", "Cliffiness"}
    assert "Accessibility level is largely explained" in claim


def test_family_deviation_table_marks_large_residuals():
    module = _load_module()
    arcs = pd.DataFrame(
        [
            {"experiment_family": "density_threshold", "pc2_std": 0.1},
            {"experiment_family": "fragmentation", "pc2_std": 0.1},
            {"experiment_family": "mlp_objectives_topology", "pc2_std": 1.0},
        ]
    )
    residuals = pd.DataFrame(
        [
            {"target": "minority_survival_cliffiness", "experiment_family": "density_threshold", "rmse": 0.05, "mae": 0.04},
            {"target": "minority_survival_cliffiness", "experiment_family": "fragmentation", "rmse": 0.06, "mae": 0.05},
            {"target": "minority_survival_cliffiness", "experiment_family": "mlp_objectives_topology", "rmse": 0.8, "mae": 0.7},
        ]
    )

    table = module.family_deviation_table(pd.DataFrame(), residuals, arcs)

    assert {"Family", "Level Fit", "Dynamics Fit", "Notes"}.issubset(table.columns)
    assert "large kernel residual" in table.loc[table["Family"] == "mlp_objectives_topology", "Notes"].iat[0]


def test_allocator_state_table_keeps_unobserved_named_allocators():
    module = _load_module()
    df = pd.DataFrame(_rows("cart") + _rows("xgboost", 0.1)).assign(experiment_family="f")

    table = module.allocator_state_table(df)

    assert "CART" in set(table["Allocator"])
    assert "LightGBM" in set(table["Allocator"])
    assert table.loc[table["Allocator"] == "CART", "Observed?"].iat[0] == "yes"
    assert table.loc[table["Allocator"] == "LightGBM", "Observed?"].iat[0] == "no"


def test_synthesis_report_generation_creates_figures(tmp_path):
    module = _load_module()
    paths = []
    for filename, model, offset in [
        ("density_threshold_sweep.csv", "cart", 0.00),
        ("fragmentation_sweep.csv", "xgboost", 0.06),
        ("allocation_trajectory.csv", "mlp_bce", 0.12),
    ]:
        path = tmp_path / filename
        df = pd.DataFrame(_rows(model, offset, n=8))
        if filename == "allocation_trajectory.csv":
            df = df.rename(columns={"model_id": "objective"})
        df.to_csv(path, index=False)
        paths.append(path)
    output = tmp_path / "accessibility_morphology_synthesis_summary.md"

    outputs = module.write_report(paths, output, gamma=0.5, grid_size=12)
    report = output.read_text(encoding="utf-8")

    assert "# Accessibility Morphology Synthesis Report" in report
    assert "## RQ1: Minimum Morphology Representation" in report
    assert "### Table 1: Morphology Explanatory Power" in report
    assert "### Table 2: Family Deviations" in report
    assert "## Synthesis Claim" in report
    assert len(outputs) == 5
    for path in outputs:
        assert path.exists()
