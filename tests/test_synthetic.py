import numpy as np

from hib.synthetic import SyntheticSkewConfig, make_gaussian_skew_dataset


def test_synthetic_generator_is_deterministic():
    config = SyntheticSkewConfig(skew_ratio=10, seed=123, minority_count=5)

    X_first, y_first = make_gaussian_skew_dataset(config)
    X_second, y_second = make_gaussian_skew_dataset(config)

    np.testing.assert_allclose(X_first, X_second)
    np.testing.assert_array_equal(y_first, y_second)
    assert X_first.shape == (55, 6)
    assert int(y_first.sum()) == 5
