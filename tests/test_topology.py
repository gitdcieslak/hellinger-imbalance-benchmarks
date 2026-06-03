import math

import numpy as np

from hib.topology import (
    compute_minority_support_radius_topology,
    compute_minority_support_topology,
    compute_positive_distance_quantile_radii,
)


def test_all_positives_connected():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
        ]
    )
    labels = np.array([1, 1, 1, 1])

    metrics = compute_minority_support_topology(embeddings, labels, k_values=[2])[2]

    assert metrics["n_positive"] == 4
    assert metrics["n_components"] == 1
    assert metrics["giant_component_fraction"] == 1.0
    assert metrics["mean_component_size"] == 4.0
    assert metrics["median_component_size"] == 4.0
    assert metrics["component_entropy"] == 0.0
    assert metrics["isolated_positive_fraction"] == 0.0


def test_all_positives_isolated():
    embeddings = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    labels = np.array([1, 1, 1])

    metrics = compute_minority_support_topology(embeddings, labels, k_values=[0])[0]

    assert metrics["n_positive"] == 3
    assert metrics["n_components"] == 3
    assert metrics["giant_component_fraction"] == 1.0 / 3.0
    assert metrics["mean_component_size"] == 1.0
    assert metrics["median_component_size"] == 1.0
    assert math.isclose(metrics["component_entropy"], math.log(3.0))
    assert metrics["isolated_positive_fraction"] == 1.0


def test_two_positive_components():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
        ]
    )
    labels = np.array([1, 1, 1, 1])

    metrics = compute_minority_support_topology(embeddings, labels, k_values=[1])[1]

    assert metrics["n_positive"] == 4
    assert metrics["n_components"] == 2
    assert metrics["giant_component_fraction"] == 0.5
    assert metrics["mean_component_size"] == 2.0
    assert metrics["median_component_size"] == 2.0
    assert math.isclose(metrics["component_entropy"], math.log(2.0))
    assert metrics["isolated_positive_fraction"] == 0.0


def test_k_larger_than_n_positive():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [100.0, 100.0],
            [200.0, 200.0],
        ]
    )
    labels = np.array([1, 1, 1, 0, 0])

    metrics = compute_minority_support_topology(embeddings, labels, k_values=[10])[10]

    assert metrics["n_positive"] == 3
    assert metrics["n_components"] == 1
    assert metrics["giant_component_fraction"] == 1.0
    assert metrics["isolated_positive_fraction"] == 0.0


def test_zero_or_one_positive_examples():
    embeddings_zero = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels_zero = np.array([0, 0])
    metrics_zero = compute_minority_support_topology(embeddings_zero, labels_zero, k_values=[1])[1]

    assert metrics_zero["n_positive"] == 0
    assert metrics_zero["n_components"] == 0
    assert metrics_zero["giant_component_fraction"] == 0.0
    assert metrics_zero["mean_component_size"] == 0.0
    assert metrics_zero["median_component_size"] == 0.0
    assert metrics_zero["component_entropy"] == 0.0
    assert metrics_zero["isolated_positive_fraction"] == 0.0

    embeddings_one = np.array([[0.0, 0.0], [10.0, 10.0]])
    labels_one = np.array([1, 0])
    metrics_one = compute_minority_support_topology(embeddings_one, labels_one, k_values=[3])[3]

    assert metrics_one["n_positive"] == 1
    assert metrics_one["n_components"] == 1
    assert metrics_one["giant_component_fraction"] == 1.0
    assert metrics_one["mean_component_size"] == 1.0
    assert metrics_one["median_component_size"] == 1.0
    assert metrics_one["component_entropy"] == 0.0
    assert metrics_one["isolated_positive_fraction"] == 1.0


def test_tiny_radius_isolates_positives():
    embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    labels = np.array([1, 1, 1])

    metrics = compute_minority_support_radius_topology(embeddings, labels, radii=[0.01])[0.01]

    assert metrics["n_positive"] == 3
    assert metrics["n_components"] == 3
    assert metrics["isolated_positive_fraction"] == 1.0


def test_large_radius_connects_all_positives():
    embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    labels = np.array([1, 1, 1])

    metrics = compute_minority_support_radius_topology(embeddings, labels, radii=[3.0])[3.0]

    assert metrics["n_positive"] == 3
    assert metrics["n_components"] == 1
    assert metrics["giant_component_fraction"] == 1.0


def test_radius_two_clusters_remain_separate_at_intermediate_radius():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
        ]
    )
    labels = np.array([1, 1, 1, 1])

    metrics = compute_minority_support_radius_topology(embeddings, labels, radii=[0.2])[0.2]

    assert metrics["n_components"] == 2
    assert metrics["giant_component_fraction"] == 0.5
    assert metrics["isolated_positive_fraction"] == 0.0


def test_positive_distance_quantile_radii_are_nondecreasing():
    embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
    labels = np.array([1, 1, 1, 0])

    radii = compute_positive_distance_quantile_radii(
        embeddings,
        labels,
        quantiles=[0.05, 0.1, 0.5, 0.9],
    )

    values = list(radii.values())
    assert values == sorted(values)
