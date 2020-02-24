import graphtools
import graphtools.estimator
import numpy as np
from load_tests import data


class Estimator(graphtools.estimator.GraphEstimator):
    def _reset_graph(self):
        self.reset = True


def test_estimator():
    E = Estimator(verbose=True)
    assert E.verbose == 1
    E = Estimator(verbose=False, n_landmark=None)
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
