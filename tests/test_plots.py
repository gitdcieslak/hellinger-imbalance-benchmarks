import numpy as np

from hib.plots import (
    deterministic_plot_paths,
    generate_score_distribution_plots,
    normalized_histogram,
    plot_score_ecdf,
    plot_score_histogram,
)


def test_deterministic_output_paths(tmp_path):
    paths = deterministic_plot_paths("xgboost", "100:1", tmp_path)

    assert paths["histogram_full"].name == "score_histogram_full_xgboost_skew_100.png"
    assert paths["histogram_zoom_010"].name == "score_histogram_zoom_010_xgboost_skew_100.png"
    assert paths["histogram_zoom_025"].name == "score_histogram_zoom_025_xgboost_skew_100.png"
    assert paths["ecdf_full"].name == "score_ecdf_full_xgboost_skew_100.png"
    assert paths["ecdf_zoom_010"].name == "score_ecdf_zoom_010_xgboost_skew_100.png"


def test_plot_generation(tmp_path):
    score_data = [
        {
            "skew_ratio": "10:1",
            "model_id": "cart",
            "positive_scores": np.array([0.6, 0.7, 0.8]),
            "negative_scores": np.array([0.1, 0.2, 0.3]),
        }
    ]
    created_paths = generate_score_distribution_plots(
        score_data,
        output_dir=tmp_path,
        normalization="class_fraction",
        include_medium_zoom=True,
    )

    assert len(created_paths) == 5
    assert all(path.exists() for path in created_paths)


def test_empty_score_handling(tmp_path):
    with np.testing.assert_raises(ValueError):
        plot_score_histogram(np.array([]), np.array([0.1, 0.2]), tmp_path / "x.png")


def test_histogram_normalization_safety():
    hist = normalized_histogram(np.array([]), normalization="class_fraction")
    assert np.all(hist == 0.0)

    hist_nonempty = normalized_histogram(np.array([0.0, 0.5, 1.0]), normalization="class_fraction")
    assert np.isclose(float(hist_nonempty.sum()), 1.0)


def test_count_normalization_preserves_counts():
    hist = normalized_histogram(np.array([0.0, 0.0, 0.1, 0.9]), normalization="count")
    assert np.isclose(float(hist.sum()), 4.0)


def test_zoomed_ecdf_generation(tmp_path):
    path = plot_score_ecdf(
        np.array([0.6, 0.7, 0.8]),
        np.array([0.1, 0.2, 0.3]),
        tmp_path / "ecdf_zoom.png",
        x_range=(0.0, 0.10),
        model_id="cart",
        skew_ratio="10:1",
    )
    assert path.exists()


def test_invalid_normalization_raises_error():
    with np.testing.assert_raises(ValueError):
        normalized_histogram(np.array([0.1, 0.2]), normalization="bad_mode")
