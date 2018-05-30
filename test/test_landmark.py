from . import (
    graphtools,
    nose2,
    data,
    digits,
    build_graph,
    raises,
    warns,
)


#####################################################
# Check parameters
#####################################################


@raises(ValueError)
def test_build_landmark_with_too_many_landmarks():
    build_graph(data, n_landmark=len(data))


@warns(RuntimeWarning)
def test_build_landmark_with_too_few_points():
    build_graph(data[:50], n_landmark=25, n_svd=100)


#####################################################
# Check kernel
#####################################################


def test_landmark_exact_graph():
    n_landmark = 100
    # exact graph
    G = build_graph(data, n_landmark=n_landmark,
                    thresh=0, n_pca=20,
                    decay=10, knn=5, random_state=42)
    assert(G.landmark_op.shape == (n_landmark, n_landmark))
    assert(isinstance(G, graphtools.graphs.TraditionalGraph))
    assert(isinstance(G, graphtools.graphs.LandmarkGraph))


def test_landmark_knn_graph():
    n_landmark = 500
    # knn graph
    G = build_graph(data, n_landmark=n_landmark, n_pca=20,
                    decay=None, knn=5, random_state=42)
    assert(G.landmark_op.shape == (n_landmark, n_landmark))
    assert(isinstance(G, graphtools.graphs.kNNGraph))
    assert(isinstance(G, graphtools.graphs.LandmarkGraph))


def test_landmark_mnn_graph():
    n_landmark = 500
    # mnn graph
    G = build_graph(data, n_landmark=n_landmark,
                    thresh=1e-5, n_pca=20,
                    decay=10, knn=5, random_state=42,
                    sample_idx=digits['target'])
    assert(G.landmark_op.shape == (n_landmark, n_landmark))
    assert(isinstance(G, graphtools.graphs.MNNGraph))
    assert(isinstance(G, graphtools.graphs.LandmarkGraph))


#####################################################
# Check interpolation
#####################################################


# TODO: add interpolation tests


def test_verbose():
    build_graph(data, decay=None, n_landmark=500, verbose=True).landmark_op


if __name__ == "__main__":
    exit(nose2.run())
