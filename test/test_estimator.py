import graphtools
import graphtools.estimator
import pygsp
import anndata
import warnings
import numpy as np
from load_tests import data, assert_raises_message
from scipy import sparse
from parameterized import parameterized


class Estimator(graphtools.estimator.GraphEstimator):
    def _reset_graph(self):
        self.reset = True


def test_estimator():
    E = Estimator(verbose=True)
    assert E.verbose == 1
    E = Estimator(verbose=False)
    assert E.verbose == 0
    E.fit(data)
    assert np.all(E.X == data)
    assert isinstance(E.graph, graphtools.graphs.kNNGraph)
    assert not isinstance(E.graph, graphtools.graphs.LandmarkGraph)
    assert not hasattr(E, "reset")
    # convert non landmark to landmark
    E.set_params(n_landmark=data.shape[0] // 2)
    assert E.reset
    assert isinstance(E.graph, graphtools.graphs.LandmarkGraph)
    del E.reset
    # convert landmark to non landmark
    E.set_params(n_landmark=None)
    assert E.reset
    assert not isinstance(E.graph, graphtools.graphs.LandmarkGraph)
    del E.reset
    # change parameters that force reset
    E.set_params(knn=E.knn * 2)
    assert E.reset
    assert E.graph is None


@parameterized(
    [
        ("precomputed", 1 - np.eye(10), "distance"),
        ("precomputed", np.eye(10), "affinity"),
        ("precomputed", sparse.coo_matrix(1 - np.eye(10)), "distance"),
        ("precomputed", sparse.eye(10), "affinity"),
        ("precomputed_affinity", 1 - np.eye(10), "affinity"),
        ("precomputed_distance", np.ones((10, 10)), "distance"),
    ]
)
def test_precomputed(distance, X, precomputed):
    E = Estimator(verbose=False, distance=distance)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="K should have a non-zero diagonal")
        E.fit(X)
    assert isinstance(E.graph, graphtools.graphs.TraditionalGraph)
    assert E.graph.precomputed == precomputed


def test_graph_input():
    X = np.random.normal(0, 1, (10, 2))
    E = Estimator(verbose=0)
    G = graphtools.Graph(X)
    E.fit(G)
    assert E.graph == G
    G = graphtools.Graph(X, knn=2, decay=5, distance="cosine", thresh=0)
    E.fit(G)
    assert E.graph == G
    assert E.knn == G.knn
    assert E.decay == G.decay
    assert E.distance == G.distance
    assert E.thresh == G.thresh
    W = G.K - np.eye(X.shape[0])
    G = pygsp.graphs.Graph(W)
    E.fit(G, use_pygsp=True)
    assert np.all(E.graph.W.toarray() == W)


def test_pca():
    X = np.random.normal(0, 1, (10, 6))
    E = Estimator(verbose=0)
    E.fit(X)
    G = E.graph
    E.set_params(n_pca=100)
    E.fit(X)
    assert E.graph is G
    E.set_params(n_pca=3)
    E.fit(X)
    assert E.graph is not G
    assert E.graph.n_pca == 3


def test_anndata_input():
    X = np.random.normal(0, 1, (10, 2))
    E = Estimator(verbose=0)
    E.fit(X.astype(np.float32))
    E2 = Estimator(verbose=0)
    E2.fit(anndata.AnnData(X))
    np.testing.assert_allclose(
        E.graph.K.toarray(), E2.graph.K.toarray(), rtol=1e-6, atol=2e-7
    )


def test_new_input():
    X = np.random.normal(0, 1, (10, 2))
    X2 = np.random.normal(0, 1, (10, 2))
    E = Estimator(verbose=0)
    E.fit(X)
    G = E.graph
    E.fit(X)
    assert E.graph is G
    E.fit(X.copy())
    assert E.graph is G
    E.n_landmark = 500
    E.fit(X)
    assert E.graph is G
    E.n_landmark = 5
    E.fit(X)
    assert np.all(E.graph.K.toarray() == G.K.toarray())
    G = E.graph
    E.fit(X2)
    assert E.graph is not G
