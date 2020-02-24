import graphtools
import graphtools.estimator
import pygsp
import numpy as np
from load_tests import data
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
        (1 - np.eye(10), "distance"),
        (np.eye(10), "affinity"),
        (sparse.coo_matrix(1 - np.eye(10)), "distance"),
        (sparse.eye(10), "affinity"),
    ]
)
def test_precomputed(X, precomputed):
    E = Estimator(verbose=False, distance="precomputed")
    assert E._detect_precomputed_matrix_type(X) == precomputed
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
