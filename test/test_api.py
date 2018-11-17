from load_tests import (
    nose2,
    data,
    build_graph,
    raises,
)
import warnings

import igraph
import numpy as np
import graphtools


def test_from_igraph():
    n = 100
    m = 500
    K = np.zeros((n, n))
    for _ in range(m):
        e = np.random.choice(n, 2, replace=False)
        K[e[0], e[1]] = K[e[1], e[0]] = 1
    g = igraph.Graph.Adjacency(K.tolist())
    G = graphtools.from_igraph(g)
    G2 = graphtools.Graph(K, precomputed='adjacency')
    assert np.all(G.K == G2.K)


def test_to_pygsp():
    G = build_graph(data)
    G2 = G.to_pygsp()
    assert isinstance(G2, graphtools.graphs.PyGSPGraph)
    assert np.all(G2.K == G.K)

#####################################################
# Check parameters
#####################################################


@raises(TypeError)
def test_unknown_parameter():
    build_graph(data, hello='world')


@raises(ValueError)
def test_invalid_graphtype():
    build_graph(data, graphtype='hello world')
