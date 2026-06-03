"""Topology utilities for positive-class support in embedding spaces."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _positive_embeddings(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.asarray(embeddings, dtype=float)[np.asarray(labels) == 1]


def _pairwise_distances(embeddings: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    if metric != "euclidean":
        raise ValueError("only euclidean distance is currently supported")
    return np.linalg.norm(
        embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :],
        axis=2,
    )


def build_positive_knn_graph(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> list[set[int]]:
    """Build an undirected k-NN graph over positive examples only.

    Node indices are local to the positive subset, ranging from 0 to n_positive-1.
    """

    if k < 0:
        raise ValueError("k must be non-negative")

    positive_embeddings = _positive_embeddings(embeddings, labels)
    n_positive = int(positive_embeddings.shape[0])

    graph: list[set[int]] = [set() for _ in range(n_positive)]
    if n_positive <= 1 or k == 0:
        return graph

    effective_k = min(k, n_positive - 1)
    distances = _pairwise_distances(positive_embeddings)

    for node_index in range(n_positive):
        neighbor_order = np.argsort(distances[node_index])
        neighbors = [idx for idx in neighbor_order if idx != node_index][:effective_k]
        for neighbor in neighbors:
            graph[node_index].add(neighbor)
            graph[neighbor].add(node_index)

    return graph


def build_positive_radius_graph(
    embeddings: np.ndarray,
    labels: np.ndarray,
    radius: float,
    metric: str = "euclidean",
) -> list[set[int]]:
    """Build an undirected radius graph over positive examples only."""

    if radius < 0.0:
        raise ValueError("radius must be non-negative")

    positive_embeddings = _positive_embeddings(embeddings, labels)
    n_positive = int(positive_embeddings.shape[0])
    graph: list[set[int]] = [set() for _ in range(n_positive)]
    if n_positive <= 1:
        return graph

    distances = _pairwise_distances(positive_embeddings, metric=metric)
    for left in range(n_positive):
        for right in range(left + 1, n_positive):
            if float(distances[left, right]) <= float(radius):
                graph[left].add(right)
                graph[right].add(left)

    return graph


def connected_component_sizes(graph: list[set[int]]) -> list[int]:
    """Return sizes of connected components for an undirected graph."""

    n_nodes = len(graph)
    if n_nodes == 0:
        return []

    visited = [False] * n_nodes
    component_sizes: list[int] = []

    for start_node in range(n_nodes):
        if visited[start_node]:
            continue
        stack = [start_node]
        visited[start_node] = True
        size = 0

        while stack:
            node = stack.pop()
            size += 1
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        component_sizes.append(size)

    return component_sizes


def compute_topology_metrics_from_component_sizes(
    component_sizes: list[int],
    n_positive: int,
) -> dict[str, float | int]:
    """Compute topology summary metrics from connected component sizes."""

    if n_positive == 0:
        return {
            "n_positive": 0,
            "n_components": 0,
            "giant_component_fraction": 0.0,
            "mean_component_size": 0.0,
            "median_component_size": 0.0,
            "component_entropy": 0.0,
            "isolated_positive_fraction": 0.0,
        }

    sizes = np.asarray(component_sizes, dtype=float)
    probabilities = sizes / float(n_positive)
    entropy = float(max(0.0, -np.sum(probabilities * np.log(probabilities))))

    return {
        "n_positive": int(n_positive),
        "n_components": int(len(component_sizes)),
        "giant_component_fraction": float(np.max(sizes) / float(n_positive)),
        "mean_component_size": float(np.mean(sizes)),
        "median_component_size": float(np.median(sizes)),
        "component_entropy": entropy,
        "isolated_positive_fraction": float(np.sum(sizes == 1.0) / float(n_positive)),
    }


def compute_minority_support_topology(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: Iterable[int],
) -> dict[int, dict[str, float | int]]:
    """Compute positive-support topology metrics for multiple k values."""

    metrics_by_k: dict[int, dict[str, float | int]] = {}
    positive_count = int(np.sum(labels == 1))

    for k in k_values:
        graph = build_positive_knn_graph(embeddings=embeddings, labels=labels, k=int(k))
        component_sizes = connected_component_sizes(graph)
        metrics_by_k[int(k)] = compute_topology_metrics_from_component_sizes(
            component_sizes=component_sizes,
            n_positive=positive_count,
        )

    return metrics_by_k


def compute_minority_support_radius_topology(
    embeddings: np.ndarray,
    labels: np.ndarray,
    radii: Iterable[float],
    metric: str = "euclidean",
) -> dict[float, dict[str, float | int]]:
    """Compute positive-support topology metrics for multiple radii."""

    metrics_by_radius: dict[float, dict[str, float | int]] = {}
    positive_count = int(np.sum(np.asarray(labels) == 1))

    for radius in radii:
        radius_value = float(radius)
        graph = build_positive_radius_graph(
            embeddings=embeddings,
            labels=labels,
            radius=radius_value,
            metric=metric,
        )
        component_sizes = connected_component_sizes(graph)
        metrics_by_radius[radius_value] = compute_topology_metrics_from_component_sizes(
            component_sizes=component_sizes,
            n_positive=positive_count,
        )

    return metrics_by_radius


def compute_positive_distance_quantile_radii(
    embeddings: np.ndarray,
    labels: np.ndarray,
    quantiles: Iterable[float] = (0.05, 0.10, 0.20, 0.30, 0.50),
    metric: str = "euclidean",
) -> dict[float, float]:
    """Return positive-positive pairwise distance radii at requested quantiles."""

    positive_embeddings = _positive_embeddings(embeddings, labels)
    n_positive = int(positive_embeddings.shape[0])
    requested = [float(quantile) for quantile in quantiles]
    for quantile in requested:
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError("quantiles must be between 0 and 1")
    if n_positive <= 1:
        return {quantile: 0.0 for quantile in requested}

    distances = _pairwise_distances(positive_embeddings, metric=metric)
    pairwise = distances[np.triu_indices(n_positive, k=1)]
    return {quantile: float(np.quantile(pairwise, quantile)) for quantile in requested}
