import numpy as np

from hib.occupancy_figures import (
    compute_ecdf,
    compute_reachability_curve,
    generate_posterior_occupancy_figure_family,
    summarize_unique_posterior_support,
)


def test_ecdf_monotonicity_and_shape():
    values = np.array([0.2, 0.1, 0.4, 0.4, 0.9], dtype=float)
    x, y = compute_ecdf(values)
    assert x.shape == y.shape
    assert x.size == values.size
    assert np.all(np.diff(x) >= 0)
    assert np.all(np.diff(y) >= 0)
    assert y[-1] == 1.0


def test_reachability_curve_monotonicity_and_threshold_values():
    scores = np.array([0.0, 0.2, 0.5, 0.7, 1.0], dtype=float)
    grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    reach = compute_reachability_curve(scores, grid)
    assert reach.shape == grid.shape
    assert np.all(np.diff(reach) <= 1e-12)
    assert np.isclose(reach[0], 1.0)
    assert np.isclose(reach[2], 0.6)
    assert np.isclose(reach[-1], 0.2)


def test_unique_posterior_support_summary_correctness():
    y = np.array([0, 0, 1, 1, 1], dtype=int)
    s = np.array([0.01, 0.01, 0.2, 0.2, 0.8], dtype=float)
    summary = summarize_unique_posterior_support(y, s, [0.50, 0.25, 0.10, 0.05, 0.01], round_decimals=6)
    assert summary["unique_score_count"] == 3
    assert np.isclose(summary["largest_mass_fraction"], 0.4)
    assert summary["threshold_crossing_count"] == 5
    assert np.isclose(summary["minority_reachability_at_0_50"], 1.0 / 3.0)
    assert np.isclose(summary["majority_reachability_at_0_50"], 0.0)


def test_tiny_end_to_end_figure_family_generation(tmp_path):
    records = [
        {
            "dataset_id": "tiny",
            "model_id": "cart",
            "split_id": "r0s0",
            "y_true": np.array([0, 0, 1, 1, 1, 0], dtype=int),
            "y_score": np.array([0.01, 0.02, 0.1, 0.2, 0.8, 0.03], dtype=float),
        },
        {
            "dataset_id": "tiny",
            "model_id": "xgboost",
            "split_id": "r0s0",
            "y_true": np.array([0, 0, 1, 1, 1, 0], dtype=int),
            "y_score": np.array([0.001, 0.002, 0.02, 0.08, 0.6, 0.01], dtype=float),
        },
    ]
    outputs = generate_posterior_occupancy_figure_family(
        posterior_records=records,
        output_dir=tmp_path,
        thresholds=[0.50, 0.25, 0.10, 0.05, 0.01],
        grid_size=101,
        round_decimals=6,
    )

    assert len(outputs["ecdf_paths"]) >= 4
    assert len(outputs["reachability_paths"]) >= 4
    assert len(outputs["support_paths"]) >= 4
    assert outputs["summary_csv"].exists()
    assert outputs["summary_md"].exists()
    assert (tmp_path / "ecdf").exists()
    assert (tmp_path / "reachability").exists()
    assert (tmp_path / "support").exists()
