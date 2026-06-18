import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_morphology_ridge_extraction.py"
    spec = importlib.util.spec_from_file_location("report_morphology_ridge_extraction", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_morphology_ridge_extraction module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(family_offset: float = 0.0):
    rows = []
    for seed in range(6):
        breadth = 0.1 + 0.1 * seed + family_offset
        elevation = 0.9 - 0.08 * seed + family_offset
        rows.append(
            {
                "model_id": "mlp_weighted_bce" if seed % 2 == 0 else "mlp_weighted_bce_dropout_0_1",
                "seed": seed,
                "breadth": breadth,
                "elevation": elevation,
                "minority_survival_auc": 0.4 + 0.5 * elevation - 0.1 * breadth,
                "minority_survival_cliffiness": 0.2 + 0.5 * breadth - 0.1 * elevation,
                "auroc": 0.7 + 0.01 * seed,
                "average_precision": 0.2 + 0.01 * seed,
            }
        )
    return rows


def test_normalize_alias_columns(tmp_path):
    module = _load_module()
    path = tmp_path / "weighted_dropout_sweep_dense.csv"
    df = pd.DataFrame(_rows()).rename(columns={"breadth": "positive_histogram_entropy", "elevation": "positive_top_bin_mass"})
    df.to_csv(path, index=False)

    normalized, message = module.normalize_result_file(path)

    assert normalized is not None
    assert "used:" in message
    assert {"experiment_family", "model", "breadth", "elevation"}.issubset(normalized.columns)
    assert normalized["experiment_family"].iloc[0] == "weighted_dropout_dense"


def test_missing_and_incomplete_inputs_are_reported(tmp_path):
    module = _load_module()
    missing = tmp_path / "missing.csv"
    incomplete = tmp_path / "fragmentation_sweep.csv"
    pd.DataFrame({"breadth": [0.1]}).to_csv(incomplete, index=False)

    df, messages = module.load_inputs([missing, incomplete])

    assert df.empty
    assert any("missing:" in message for message in messages)
    assert any("skipped incomplete:" in message for message in messages)


def test_topology_file_deduplicates_model_seed(tmp_path):
    module = _load_module()
    path = tmp_path / "mlp_objectives_topology.csv"
    rows = _rows()[:2]
    duplicated = []
    for row in rows:
        for k in [3, 5]:
            item = dict(row)
            item["objective"] = item.pop("model_id")
            item["k"] = k
            duplicated.append(item)
    pd.DataFrame(duplicated).to_csv(path, index=False)

    normalized, _ = module.normalize_result_file(path)

    assert normalized is not None
    assert len(normalized) == 2


def test_report_generation_creates_required_sections_and_plots(tmp_path):
    module = _load_module()
    paths = []
    for filename, offset in [
        ("density_threshold_sweep.csv", 0.00),
        ("fragmentation_sweep.csv", 0.05),
        ("allocation_trajectory.csv", 0.10),
    ]:
        path = tmp_path / filename
        df = pd.DataFrame(_rows(offset))
        if filename == "allocation_trajectory.csv":
            df = df.rename(columns={"model_id": "objective"})
        df.to_csv(path, index=False)
        paths.append(path)
    output_md = tmp_path / "morphology_ridge_extraction_summary.md"

    outputs = module.write_report(paths, output_md)
    report = output_md.read_text(encoding="utf-8")

    assert "# Morphology Ridge Extraction" in report
    assert "## Inputs Used" in report
    assert "## Predictive Performance" in report
    assert "## Morphology Coefficients" in report
    assert "## Sign Stability" in report
    assert "## Morphology Ridge Diagnostics" in report
    assert "Can breadth/elevation predict survival AUC" in report
    assert len(outputs) == 5
    for path in outputs:
        assert path.exists()
