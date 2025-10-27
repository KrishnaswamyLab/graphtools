from __future__ import print_function

from load_tests import build_graph
from load_tests import data
from load_tests import graphtools
from load_tests import np

import pytest


def test_connected_graph():
    """Test that a normal graph is connected"""
    G = build_graph(data, n_pca=20, decay=10, knn=5)

    # Check that the graph is connected
    assert G.is_connected, "Expected graph to be connected"
    assert G.n_connected_components == 1, "Expected exactly 1 connected component"

    # Check component labels
    labels = G.component_labels
    assert labels.shape[0] == data.shape[0], "Component labels should match data size"
    assert np.all(labels == 0), "All nodes should be in component 0"


def test_disconnected_graph():
    """Test a graph that is intentionally disconnected"""
    # Create two separate clusters of data that won't connect
    cluster1 = np.random.randn(50, 10)
    cluster2 = np.random.randn(50, 10) + 100  # Far away from cluster1
    disconnected_data = np.vstack([cluster1, cluster2])

    # Build graph with small knn to ensure disconnection
    G = build_graph(disconnected_data, n_pca=None, decay=10, knn=3, thresh=1e-4)

    # Check that the graph is disconnected
    assert not G.is_connected, "Expected graph to be disconnected"
    assert G.n_connected_components >= 2, "Expected at least 2 connected components"

    # Check component labels
    labels = G.component_labels
    assert labels.shape[0] == disconnected_data.shape[0], "Component labels should match data size"
    assert len(np.unique(labels)) >= 2, "Should have at least 2 unique component labels"


def test_component_labels_consistency():
    """Test that component labels are consistent across calls"""
    # Create disconnected data
    cluster1 = np.random.randn(30, 5)
    cluster2 = np.random.randn(30, 5) + 50
    disconnected_data = np.vstack([cluster1, cluster2])

    G = build_graph(disconnected_data, n_pca=None, decay=10, knn=2)

    # Get labels multiple times - should be cached and identical
    labels1 = G.component_labels
    labels2 = G.component_labels
    n_comp1 = G.n_connected_components
    n_comp2 = G.n_connected_components

    assert np.array_equal(labels1, labels2), "Component labels should be cached"
    assert n_comp1 == n_comp2, "n_connected_components should be cached"


def test_precomputed_graph_connectivity():
    """Test connectivity with precomputed distance matrix"""
    from scipy.spatial.distance import pdist, squareform

    # Create small disconnected dataset
    cluster1 = np.array([[0, 0], [0, 1], [1, 0]])
    cluster2 = np.array([[100, 100], [100, 101], [101, 100]])
    disconnected_data = np.vstack([cluster1, cluster2])

    # Compute distance matrix
    dist_matrix = squareform(pdist(disconnected_data))

    # For precomputed graphs, n_pca must be None
    G = build_graph(dist_matrix, n_pca=None, precomputed="distance", decay=10, knn=2)

    # Should be disconnected
    assert not G.is_connected, "Precomputed disconnected graph should be disconnected"
    assert G.n_connected_components == 2, "Should have exactly 2 components"


def test_landmark_graph_connectivity():
    """Test connectivity with landmark graphs"""
    G = build_graph(data, n_pca=20, decay=10, knn=5, n_landmark=100)

    # Landmark graphs should still support connectivity checks
    assert hasattr(G, 'is_connected'), "Landmark graph should have is_connected property"
    assert hasattr(G, 'n_connected_components'), "Landmark graph should have n_connected_components"
    assert hasattr(G, 'component_labels'), "Landmark graph should have component_labels"

    # Check that properties work
    is_conn = G.is_connected
    n_comp = G.n_connected_components
    labels = G.component_labels

    assert isinstance(is_conn, (bool, np.bool_)), "is_connected should return boolean"
    assert isinstance(n_comp, (int, np.integer)), "n_connected_components should return int"
    assert labels.shape[0] == data.shape[0], "component_labels should match data size"


def test_knn_graph_connectivity():
    """Test connectivity with different knn values"""
    # With high knn, should be connected
    G_high_knn = build_graph(data, n_pca=20, knn=10, decay=10)
    assert G_high_knn.is_connected, "Graph with high knn should be connected"

    # Create data that might disconnect with very low knn
    sparse_data = np.random.randn(100, 10) * 2
    G_low_knn = build_graph(sparse_data, n_pca=None, knn=2, decay=10)

    # Just check that the properties work (connectivity depends on data)
    assert isinstance(G_low_knn.is_connected, (bool, np.bool_))
    assert G_low_knn.n_connected_components >= 1


def test_component_labels_range():
    """Test that component labels are in the correct range"""
    cluster1 = np.random.randn(20, 5)
    cluster2 = np.random.randn(20, 5) + 100
    cluster3 = np.random.randn(20, 5) - 100
    disconnected_data = np.vstack([cluster1, cluster2, cluster3])

    G = build_graph(disconnected_data, n_pca=None, decay=10, knn=2)

    labels = G.component_labels
    n_comp = G.n_connected_components

    # Labels should be in range [0, n_components)
    assert labels.min() >= 0, "Minimum label should be >= 0"
    assert labels.max() < n_comp, "Maximum label should be < n_connected_components"
    assert len(np.unique(labels)) == n_comp, "Number of unique labels should equal n_components"


def test_connectivity_caching():
    """Test that connectivity properties are properly cached"""
    G = build_graph(data, n_pca=20, decay=10, knn=5)

    # Access properties to trigger caching
    _ = G.is_connected

    # Check that internal cache exists
    assert hasattr(G, '_n_connected_components'), "Should cache n_connected_components"
    assert hasattr(G, '_component_labels'), "Should cache component_labels"

    # Verify cached values are used
    n_comp_cached = G._n_connected_components
    n_comp_property = G.n_connected_components

    assert n_comp_cached == n_comp_property, "Cached value should match property value"
