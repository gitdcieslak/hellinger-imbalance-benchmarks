import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_script(name: str):
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_availability_resolution_handles_missing():
    runner = _load_script("run_classical_allocator_morphology")

    available, missing = runner.resolve_model_availability(["cart", "not_a_model"])

    assert "cart" in available
    assert missing["not_a_model"] == "not in model registry"


def test_result_and_failed_row_construction():
    runner = _load_script("run_classical_allocator_morphology")
    y_true = np.asarray([0, 0, 1, 1, 0, 1])
    y_score = np.asarray([0.1, 0.2, 0.6, 0.9, 0.3, 0.8])

    row = runner.build_result_row("cart", 0, 100, 50, y_true, y_score)
    failed = runner.failed_result_row("cart", 0, 100, 50, "boom")

    assert row["model_id"] == "cart"
    assert row["fit_failed"] is False
    assert "minority_survival_auc" in row
    assert "positive_quantization_score" in row
    assert failed["fit_failed"] is True
    assert np.isnan(failed["auroc"])


def test_regime_classification_rules():
    report = _load_script("report_classical_allocator_morphology")

    assert report.classify_regime(pd.Series({"positive_quantization_score": 0.95, "positive_unique_score_ratio": 0.5, "breadth": 1.2, "elevation": 0.2})) == "quantized allocator"
    assert report.classify_regime(pd.Series({"positive_quantization_score": 0.2, "positive_unique_score_ratio": 0.5, "breadth": 0.4, "elevation": 0.9})) == "concentrated allocator"
    assert report.classify_regime(pd.Series({"positive_quantization_score": 0.2, "positive_unique_score_ratio": 0.5, "breadth": 1.2, "elevation": 0.6})) == "broad allocator"
    assert report.classify_regime(pd.Series({"positive_quantization_score": 0.2, "positive_unique_score_ratio": 0.5, "breadth": 0.8, "elevation": 0.7})) == "mixed morphology"


def _tiny_classical_rows():
    rows = []
    for model_id, breadth, elevation, quant in [("cart", 0.0, 1.0, 0.98), ("random_forest", 0.7, 0.85, 0.2)]:
        for seed in [0, 1]:
            rows.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "skew_ratio": 100,
                    "minority_count": 50,
                    "fit_failed": False,
                    "failure_reason": "",
                    "auroc": 0.8 + 0.01 * seed,
                    "average_precision": 0.2 + 0.01 * seed,
                    "minority_survival_auc": 0.7 + 0.01 * seed,
                    "minority_survival_cliffiness": 0.3 + 0.01 * seed,
                    "minority_survival_max_drop": 0.1,
                    "minority_survival_effective_drop_count": 2.0,
                    "breadth": breadth,
                    "effective_breadth": 1.0,
                    "elevation": elevation,
                    "peak_concentration": elevation,
                    "positive_unique_score_ratio": 1.0 - quant,
                    "positive_quantization_score": quant,
                    "positive_histogram_entropy": breadth,
                    "positive_effective_score_bins": 1.0,
                    "positive_max_bin_mass": elevation,
                    "positive_top_bin_mass": elevation,
                    "positive_score_iqr": 0.1,
                    "positive_score_q10_q90_width": 0.2,
                }
            )
    return rows


def test_report_generation_with_missing_prior_overlay(tmp_path):
    report = _load_script("report_classical_allocator_morphology")
    input_csv = tmp_path / "classical.csv"
    pd.DataFrame(_tiny_classical_rows()).to_csv(input_csv, index=False)
    output_md = tmp_path / "classical_summary.md"

    outputs = report.write_report(
        input_csv,
        output_md,
        expected_models=["cart", "random_forest", "lightgbm"],
        prior_inputs=[tmp_path / "missing_prior.csv"],
    )
    text = output_md.read_text(encoding="utf-8")

    assert "# Classical Allocator Morphology" in text
    assert "## 1. Inputs and Availability" in text
    assert "lightgbm" in text
    assert "not present in input CSV" in text
    assert "## 7. Paper 2 Implications" in text
    assert len(outputs) == 5
    for path in outputs:
        assert path.exists()


def test_failed_fit_rows_are_reported(tmp_path):
    report = _load_script("report_classical_allocator_morphology")
    rows = _tiny_classical_rows()
    rows.append({**rows[0], "model_id": "hddt", "fit_failed": True, "failure_reason": "fit failed", "auroc": np.nan})
    df = pd.DataFrame(rows)
    input_csv = tmp_path / "classical.csv"
    df.to_csv(input_csv, index=False)

    loaded = report.add_regimes(report.load_classical_results(input_csv))
    availability = report.availability_table(loaded, ["cart", "hddt"])

    assert availability.loc[availability["model_id"] == "hddt", "status"].iat[0] == "failed"
