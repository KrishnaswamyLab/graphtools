from __future__ import print_function
from load_tests import data, build_graph, assert_raises_message, assert_warns_message


import igraph
import numpy as np
import graphtools
import tempfile
import os
import pickle


def test_from_igraph():
    n = 100
    m = 500
    K = np.zeros((n, n))
    for _ in range(m):
        e = np.random.choice(n, 2, replace=False)
        K[e[0], e[1]] = K[e[1], e[0]] = 1
    g = igraph.Graph.Adjacency(K.tolist())
    G = graphtools.from_igraph(g, attribute=None)
    G2 = graphtools.Graph(K, precomputed="adjacency")
    assert np.all(G.K == G2.K)


def test_from_igraph_weighted():
    n = 100
    m = 500
    K = np.zeros((n, n))
    for _ in range(m):
        e = np.random.choice(n, 2, replace=False)
        K[e[0], e[1]] = K[e[1], e[0]] = np.random.uniform(0, 1)
    g = igraph.Graph.Weighted_Adjacency(K.tolist())
    G = graphtools.from_igraph(g)
    G2 = graphtools.Graph(K, precomputed="adjacency")
    assert np.all(G.K == G2.K)


def test_from_igraph_invalid_precomputed():
    with assert_warns_message(
        UserWarning,
        "Cannot build graph from igraph with precomputed=affinity. Use 'adjacency' instead.",
    ):
        n = 100
        m = 500
        K = np.zeros((n, n))
        for _ in range(m):
            e = np.random.choice(n, 2, replace=False)
            K[e[0], e[1]] = K[e[1], e[0]] = 1
        g = igraph.Graph.Adjacency(K.tolist())
        G = graphtools.from_igraph(g, attribute=None, precomputed="affinity")


def test_from_igraph_invalid_attribute():
    with assert_warns_message(
        UserWarning, "Edge attribute invalid not found. Returning unweighted graph"
    ):
        n = 100
        m = 500
        K = np.zeros((n, n))
        for _ in range(m):
            e = np.random.choice(n, 2, replace=False)
            K[e[0], e[1]] = K[e[1], e[0]] = 1
        g = igraph.Graph.Adjacency(K.tolist())
        G = graphtools.from_igraph(g, attribute="invalid")


def test_to_pygsp():
    G = build_graph(data)
    G2 = G.to_pygsp()
    assert isinstance(G2, graphtools.graphs.PyGSPGraph)
    assert np.all(G2.K == G.K)


def test_to_igraph():
    G = build_graph(data, use_pygsp=True)
    G2 = G.to_igraph()
    assert isinstance(G2, igraph.Graph)
    assert np.all(np.array(G2.get_adjacency(attribute="weight").data) == G.W)
    G3 = build_graph(data, use_pygsp=False)
    G2 = G3.to_igraph()
    assert isinstance(G2, igraph.Graph)
    assert np.all(np.array(G2.get_adjacency(attribute="weight").data) == G.W)


def test_pickle_io_knngraph():
    G = build_graph(data, knn=5, decay=None)
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "tmp.pkl")
        G.to_pickle(path)
        G_prime = graphtools.read_pickle(path)
    assert isinstance(G_prime, type(G))


def test_pickle_io_traditionalgraph():
    G = build_graph(data, knn=5, decay=10, thresh=0)
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "tmp.pkl")
        G.to_pickle(path)
        G_prime = graphtools.read_pickle(path)
    assert isinstance(G_prime, type(G))


def test_pickle_io_landmarkgraph():
    G = build_graph(data, knn=5, decay=None, n_landmark=data.shape[0] // 2)
    L = G.landmark_op
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "tmp.pkl")
        G.to_pickle(path)
        G_prime = graphtools.read_pickle(path)
    assert isinstance(G_prime, type(G))
    np.testing.assert_array_equal(L, G_prime._landmark_op)


def test_pickle_io_pygspgraph():
    G = build_graph(data, knn=5, decay=None, use_pygsp=True)
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "tmp.pkl")
        G.to_pickle(path)
        G_prime = graphtools.read_pickle(path)
    assert isinstance(G_prime, type(G))
    assert G_prime.logger.name == G.logger.name


def test_pickle_bad_pickle():
    with assert_warns_message(
        UserWarning, "Returning object that is not a graphtools.base.BaseGraph"
    ):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "tmp.pkl")
            with open(path, "wb") as f:
                pickle.dump("hello world", f)
            G = graphtools.read_pickle(path)


def test_to_pygsp_invalid_precomputed():
    with assert_warns_message(
        UserWarning,
        "Cannot build PyGSPGraph with precomputed=adjacency. Using 'affinity' instead.",
    ):
        G = build_graph(data)
        G2 = G.to_pygsp(precomputed="adjacency")


def test_to_pygsp_invalid_use_pygsp():
    with assert_warns_message(
        UserWarning, "Cannot build PyGSPGraph with use_pygsp=False. Use True instead."
    ):
        G = build_graph(data)
        G2 = G.to_pygsp(use_pygsp=False)


#####################################################
# Check parameters
#####################################################


def test_unknown_parameter():
    with assert_raises_message(
        TypeError, "__init__() got an unexpected keyword argument 'hello'"
    ):
        build_graph(data, hello="world")


def test_invalid_graphtype():
    with assert_raises_message(
        ValueError,
        "graphtype 'hello world' not recognized. Choose from ['knn', 'mnn', 'exact', 'auto']",
    ):
        build_graph(data, graphtype="hello world")
