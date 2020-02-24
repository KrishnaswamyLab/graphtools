from __future__ import print_function
from load_tests import (
    graphtools,
    np,
    nose2,
    data,
    digits,
    build_graph,
    generate_swiss_roll,
    assert_raises_message,
    assert_warns_message,
)
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
    n_landmark = 500
    # knn graph
    G = build_graph(
        data, n_landmark=n_landmark, n_pca=20, decay=None, knn=5 - 1, random_state=42
    )
    assert G.transitions.shape == (data.shape[0], n_landmark)
    assert G.landmark_op.shape == (n_landmark, n_landmark)
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
    n_landmark = 500
    # knn graph
    G = build_graph(
        data,
        n_landmark=n_landmark,
        n_pca=10,
        decay=None,
        knn=3 - 1,
        random_state=42,
        use_pygsp=True,
    )
    assert G.landmark_op.shape == (n_landmark, n_landmark)
    assert isinstance(G, graphtools.graphs.kNNGraph)
    assert isinstance(G, graphtools.graphs.LandmarkGraph)
    assert isinstance(G, pygsp.graphs.Graph)


def test_landmark_mnn_pygsp_graph():
    n_landmark = 150
    X, sample_idx = generate_swiss_roll()
    # mnn graph
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
    assert isinstance(G, graphtools.graphs.MNNGraph)
    assert isinstance(G, graphtools.graphs.LandmarkGraph)
    assert isinstance(G, pygsp.graphs.Graph)


#####################################################
# Check interpolation
#####################################################


# TODO: add interpolation tests


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
        "knn": 3,
        "knn_max": None,
        "decay": None,
        "bandwidth": None,
        "bandwidth_scale": 1,
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
