import pandas as pd

from hib.synthesis import build_paper_synthesis_table, paper_synthesis_to_markdown, write_paper_synthesis_report


def test_synthesis_table_and_markdown(tmp_path):
    occupancy = pd.DataFrame(
        [
            {
                "dataset_id": "boundary",
                "model_id": "cart",
                "occupancy_entropy_mean": 0.2,
                "posterior_sparsity_index_mean": 0.8,
                "threshold_occupancy_persistence_mean": 0.9,
                "quantization_score_mean": 0.1,
            },
            {
                "dataset_id": "boundary",
                "model_id": "xgboost",
                "occupancy_entropy_mean": 1.1,
                "posterior_sparsity_index_mean": 0.3,
                "threshold_occupancy_persistence_mean": 0.4,
                "quantization_score_mean": 0.7,
            },
        ]
    )
    elasticity = pd.DataFrame(
        [
            {
                "dataset_id": "boundary",
                "model_id": "cart",
                "mean_abs_recall_elasticity": 0.1,
                "operational_smoothness_index": 0.9,
                "max_recall_jump": 0.1,
            },
            {
                "dataset_id": "boundary",
                "model_id": "xgboost",
                "mean_abs_recall_elasticity": 1.2,
                "operational_smoothness_index": 0.3,
                "max_recall_jump": 0.6,
            },
        ]
    )
    regimes = pd.DataFrame(
        [
            {
                "model_id": "cart",
                "inferred_regime": "quantized_allocator",
                "mean_effective_support_size": 1.1,
                "mean_histogram_entropy": 0.2,
            },
            {
                "model_id": "xgboost",
                "inferred_regime": "cliff_allocator",
                "mean_effective_support_size": 3.1,
                "mean_histogram_entropy": 1.2,
            },
        ]
    )
    calibration = pd.DataFrame(
        [
            {"model_id": "cart", "calibration_method": "raw", "mean_ece": 0.20, "mean_brier": 0.30, "mean_smoothness": 0.7},
            {"model_id": "cart", "calibration_method": "platt", "mean_ece": 0.15, "mean_brier": 0.27, "mean_smoothness": 0.8},
            {"model_id": "xgboost", "calibration_method": "raw", "mean_ece": 0.12, "mean_brier": 0.22, "mean_smoothness": 0.5},
            {"model_id": "xgboost", "calibration_method": "isotonic", "mean_ece": 0.09, "mean_brier": 0.20, "mean_smoothness": 0.6},
        ]
    )
    persistence = pd.DataFrame(
        [
            {"model_id": "cart", "calibration_method": "raw", "inferred_regime": "quantized_allocator"},
            {"model_id": "cart", "calibration_method": "platt", "inferred_regime": "quantized_allocator"},
            {"model_id": "xgboost", "calibration_method": "raw", "inferred_regime": "cliff_allocator"},
            {"model_id": "xgboost", "calibration_method": "isotonic", "inferred_regime": "smooth_allocator"},
        ]
    )

    synthesis = build_paper_synthesis_table(occupancy, elasticity, regimes, calibration, persistence)
    assert list(synthesis["model_id"]) == ["cart", "xgboost"]
    assert synthesis.loc[synthesis["model_id"] == "cart", "best_calibration_method"].iat[0] == "platt"

    markdown = paper_synthesis_to_markdown(synthesis)
    assert "Operational Synthesis Report" in markdown
    assert "quantized_allocator" in markdown
    assert "xgboost" in markdown


def test_write_paper_synthesis_report(tmp_path):
    occupancy = tmp_path / "occupancy.csv"
    elasticity = tmp_path / "elasticity.csv"
    regimes = tmp_path / "regimes.csv"
    calibration = tmp_path / "calibration.csv"
    persistence = tmp_path / "persistence.csv"

    pd.DataFrame(
        [
            {
                "dataset_id": "boundary",
                "model_id": "cart",
                "occupancy_entropy_mean": 0.2,
                "posterior_sparsity_index_mean": 0.8,
                "threshold_occupancy_persistence_mean": 0.9,
                "quantization_score_mean": 0.1,
            }
        ]
    ).to_csv(occupancy, index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": "boundary",
                "model_id": "cart",
                "mean_abs_recall_elasticity": 0.1,
                "operational_smoothness_index": 0.9,
                "max_recall_jump": 0.1,
            }
        ]
    ).to_csv(elasticity, index=False)
    pd.DataFrame(
        [
            {
                "model_id": "cart",
                "inferred_regime": "quantized_allocator",
                "mean_effective_support_size": 1.1,
                "mean_histogram_entropy": 0.2,
            }
        ]
    ).to_csv(regimes, index=False)
    pd.DataFrame(
        [
            {"model_id": "cart", "calibration_method": "raw", "mean_ece": 0.20, "mean_brier": 0.30, "mean_smoothness": 0.7},
            {"model_id": "cart", "calibration_method": "platt", "mean_ece": 0.15, "mean_brier": 0.27, "mean_smoothness": 0.8},
        ]
    ).to_csv(calibration, index=False)
    pd.DataFrame(
        [
            {"model_id": "cart", "calibration_method": "raw", "inferred_regime": "quantized_allocator"},
            {"model_id": "cart", "calibration_method": "platt", "inferred_regime": "quantized_allocator"},
        ]
    ).to_csv(persistence, index=False)

    output = tmp_path / "paper.md"
    path = write_paper_synthesis_report(occupancy, elasticity, regimes, calibration, persistence, output)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Integrated Model Table" in text
