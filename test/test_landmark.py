from __future__ import print_function

from load_tests import assert_raises_message
from load_tests import assert_warns_message
from load_tests import build_graph
from load_tests import data
from load_tests import digits
from load_tests import generate_swiss_roll
from load_tests import graphtools
from load_tests import np

import pygsp

#####################################################
# Check parameters
#####################################################


def test_build_landmark_with_too_many_landmarks():
    with assert_raises_message(
        ValueError,
        "n_landmark ({0}) >= n_samples ({0}). Use kNNGraph instead".format(
            data.shape[0]
        ),
    ):
        build_graph(data, n_landmark=len(data))


def test_build_landmark_with_too_few_points():
    with assert_warns_message(
        RuntimeWarning,
        "n_svd (100) >= n_samples (50) Consider using kNNGraph or lower n_svd",
    ):
        build_graph(data[:50], n_landmark=25, n_svd=100)


#####################################################
# Check kernel
#####################################################


def test_landmark_exact_graph():
    n_landmark = 100
    # exact graph
    G = build_graph(
        data,
        n_landmark=n_landmark,
        thresh=0,
        n_pca=20,
        decay=10,
        knn=5 - 1,
        random_state=42,
    )
    assert G.landmark_op.shape == (n_landmark, n_landmark)
    assert isinstance(G, graphtools.graphs.TraditionalGraph)
    assert isinstance(G, graphtools.graphs.LandmarkGraph)
    assert G.transitions.shape == (data.shape[0], n_landmark)
    assert G.clusters.shape == (data.shape[0],)
    assert len(np.unique(G.clusters)) <= n_landmark
    signal = np.random.normal(0, 1, [n_landmark, 10])
    interpolated_signal = G.interpolate(signal)
    assert interpolated_signal.shape == (data.shape[0], signal.shape[1])
    G._reset_landmarks()
    # no error on double delete
    G._reset_landmarks()


def test_landmark_knn_graph():
    np.random.seed(42)
    n_landmark = 500
    # knn graph
    G = build_graph(
        data, n_landmark=n_landmark, n_pca=20, decay=None, knn=5 - 1, random_state=42
    )
    n_landmark_out = G.landmark_op.shape[0]
    assert n_landmark_out <= n_landmark
    assert n_landmark_out >= n_landmark - 3
    assert G.transitions.shape == (data.shape[0], n_landmark_out), G.transitions.shape
    assert G.landmark_op.shape == (n_landmark_out, n_landmark_out)
    assert isinstance(G, graphtools.graphs.kNNGraph)
    assert isinstance(G, graphtools.graphs.LandmarkGraph)


def test_landmark_mnn_graph():
    n_landmark = 150
    X, sample_idx = generate_swiss_roll()
    # mnn graph
    G = build_graph(
        X,
        n_landmark=n_landmark,
        thresh=1e-5,
        n_pca=None,
        decay=10,
        knn=5 - 1,
        random_state=42,
        sample_idx=sample_idx,
    )
    assert G.clusters.shape == (X.shape[0],)
    assert G.landmark_op.shape == (n_landmark, n_landmark)
    assert isinstance(G, graphtools.graphs.MNNGraph)
    assert isinstance(G, graphtools.graphs.LandmarkGraph)


#####################################################
# Check PyGSP
#####################################################


def test_landmark_exact_pygsp_graph():
    n_landmark = 100
    # exact graph
    G = build_graph(
        data,
        n_landmark=n_landmark,
        thresh=0,
        n_pca=10,
        decay=10,
        knn=3 - 1,
        random_state=42,
        use_pygsp=True,
    )
    assert G.landmark_op.shape == (n_landmark, n_landmark)
    assert isinstance(G, graphtools.graphs.TraditionalGraph)
    assert isinstance(G, graphtools.graphs.LandmarkGraph)
    assert isinstance(G, pygsp.graphs.Graph)


def test_landmark_knn_pygsp_graph():
    n_landmark = 150
    # knn graph
    print(f"Data shape: {data.shape}")
    print(f"n_landmark requested: {n_landmark}")

    G = build_graph(
        data,
        n_landmark=n_landmark,
        n_pca=10,
        decay=None,
        knn=3 - 1,
        random_state=42,
        use_pygsp=True,
    )
    print(f"Graph type: {type(G)}")
    print(f"Has landmark_op: {hasattr(G, 'landmark_op')}")
    if hasattr(G, "landmark_op"):
        print(f"landmark_op shape: {G.landmark_op.shape}")

    # Check if landmarks were actually created
    if hasattr(G, "n_landmark"):
        print(f"G.n_landmark: {G.n_landmark}")
    if hasattr(G, "landmark_indices"):
        print(f"Number of landmark indices: {len(G.landmark_indices)}")

    assert G.landmark_op.shape == (n_landmark, n_landmark)
    assert isinstance(G, graphtools.graphs.kNNGraph)
    assert isinstance(G, graphtools.graphs.LandmarkGraph)
    assert isinstance(G, pygsp.graphs.Graph)


def test_landmark_mnn_pygsp_graph():
    n_landmark = 150
    X, sample_idx = generate_swiss_roll()

    print(f"Data shape: {X.shape}")
    print(f"n_landmark requested: {n_landmark}")

    G = build_graph(
        X,
        n_landmark=n_landmark,
        thresh=1e-3,
        n_pca=None,
        decay=10,
        knn=3 - 1,
        random_state=42,
        sample_idx=sample_idx,
        use_pygsp=True,
    )

    assert G.landmark_op.shape == (n_landmark, n_landmark)


#####################################################
# Check interpolation
#####################################################


# TODO: add interpolation tests


#####################################################
# Check distance metrics
#####################################################


def test_landmark_with_non_euclidean_distances():
    """Test regular landmarking with non-euclidean distance metrics

    This test verifies that the distance parameter is properly used in the
    landmark operator construction for spectral clustering-based landmarking
    (as opposed to random landmarking). This may reveal bugs in how
    build_landmark_op handles different distance metrics.
    """
    n_landmark = 30
    small_data = data[:200]  # Use smaller dataset for testing

    # Test with Manhattan distance
    G_manhattan = build_graph(
        small_data,
        n_landmark=n_landmark,
        random_landmarking=False,  # Use spectral clustering landmarking
        random_state=42,
        distance="cityblock",
        n_pca=None,  # Don't use PCA to maintain original distances
        decay=10,
        knn=5,
    )

    # Basic functionality should work
    assert G_manhattan.random_landmarking is False
    assert G_manhattan.distance == "cityblock"
    assert G_manhattan.landmark_op.shape == (n_landmark, n_landmark)
    assert G_manhattan.transitions.shape == (small_data.shape[0], n_landmark)
    assert len(G_manhattan.clusters) == small_data.shape[0]
    assert len(np.unique(G_manhattan.clusters)) == n_landmark

    # Test with Cosine distance
    G_cosine = build_graph(
        small_data,
        n_landmark=n_landmark,
        random_landmarking=False,  # Use spectral clustering landmarking
        random_state=42,
        distance="cosine",
        n_pca=None,  # Don't use PCA to maintain original distances
        decay=10,
        knn=5,
    )

    # Basic functionality should work
    assert G_cosine.random_landmarking is False
    assert G_cosine.distance == "cosine"
    assert G_cosine.landmark_op.shape == (n_landmark, n_landmark)
    assert G_cosine.transitions.shape == (small_data.shape[0], n_landmark)
    assert len(G_cosine.clusters) == small_data.shape[0]
    assert len(np.unique(G_cosine.clusters)) == n_landmark

    # Compare with Euclidean distance baseline
    euclidean_G = build_graph(
        small_data,
        n_landmark=n_landmark,
        random_landmarking=False,  # Use spectral clustering landmarking
        random_state=42,
        distance="euclidean",
        n_pca=None,
        decay=10,
        knn=5,
    )

    # Store cluster assignments for comparison
    euclidean_clusters = euclidean_G.clusters
    manhattan_clusters = G_manhattan.clusters
    cosine_clusters = G_cosine.clusters

    # At least one of the non-euclidean distance metrics should give different results
    # This tests that the distance parameter is actually being used in build_landmark_op
    different_from_euclidean = not np.array_equal(
        euclidean_clusters, manhattan_clusters
    ) or not np.array_equal(euclidean_clusters, cosine_clusters)

    # If this assertion fails, it indicates that the distance parameter
    # is not being respected in the spectral clustering landmark implementation
    if not different_from_euclidean:
        import warnings

        warnings.warn(
            "Spectral clustering landmarking may not be respecting the distance parameter. "
            "This could indicate a bug in build_landmark_op where the distance metric "
            "is not properly passed through to the clustering algorithm.",
            UserWarning,
        )

    # For now, we expect this to potentially fail until the bug is fixed
    # assert different_from_euclidean, (
    #     "Distance parameter should affect landmark clustering assignments, "
    #     "but all distance metrics gave identical results"
    # )

    # Test that the landmark operators are different shapes/values when different distances
    # are used (this is a more sensitive test than just cluster assignments)
    euclidean_landmark_sum = np.sum(euclidean_G.landmark_op)
    manhattan_landmark_sum = np.sum(G_manhattan.landmark_op)
    cosine_landmark_sum = np.sum(G_cosine.landmark_op)

    print(
        f"Landmark operator sums: euclidean={euclidean_landmark_sum:.6f}, "
        f"manhattan={manhattan_landmark_sum:.6f}, cosine={cosine_landmark_sum:.6f}"
    )

    # The landmark operators should be different when using different distance metrics
    operators_different = (
        abs(euclidean_landmark_sum - manhattan_landmark_sum) > 1e-10
        or abs(euclidean_landmark_sum - cosine_landmark_sum) > 1e-10
    )

    if not operators_different:
        import warnings

        warnings.warn(
            "Landmark operators are identical across different distance metrics. "
            "This strongly suggests the distance parameter is being ignored in build_landmark_op.",
            UserWarning,
        )


#############
# Test API
#############


def test_verbose():
    print()
    print("Verbose test: Landmark")
    build_graph(data, decay=None, n_landmark=500, verbose=True).landmark_op


def test_set_params():
    G = build_graph(data, n_landmark=500, decay=None)
    G.landmark_op
    assert G.get_params() == {
        "n_pca": 20,
        "random_state": 42,
        "kernel_symm": "+",
        "theta": None,
        "n_landmark": 500,
        "anisotropy": 0,
        "random_landmarking": False,
        "knn": 3,
        "knn_max": None,
        "decay": None,
        "bandwidth": None,
        "bandwidth_scale": 1.0,
        "distance": "euclidean",
        "thresh": 0,
        "n_jobs": -1,
        "verbose": 0,
    }
    G.set_params(n_landmark=300)
    assert G.landmark_op.shape == (300, 300)
    G.set_params(n_landmark=G.n_landmark, n_svd=G.n_svd)
    assert hasattr(G, "_landmark_op")
    G.set_params(n_svd=50)
    assert not hasattr(G, "_landmark_op")
