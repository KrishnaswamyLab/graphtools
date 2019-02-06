from __future__ import print_function
from load_tests import (
    data,
    build_graph,
    raises,
    warns,
)

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
    G = graphtools.from_igraph(g, attribute=None)
    G2 = graphtools.Graph(K, precomputed='adjacency')
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
    G2 = graphtools.Graph(K, precomputed='adjacency')
    assert np.all(G.K == G2.K)


@warns(UserWarning)
def test_from_igraph_invalid_precomputed():
    n = 100
    m = 500
    K = np.zeros((n, n))
    for _ in range(m):
        e = np.random.choice(n, 2, replace=False)
        K[e[0], e[1]] = K[e[1], e[0]] = 1
    g = igraph.Graph.Adjacency(K.tolist())
    G = graphtools.from_igraph(g, attribute=None, precomputed='affinity')


@warns(UserWarning)
def test_from_igraph_invalid_attribute():
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
    assert np.all(np.array(G2.get_adjacency(
        attribute="weight").data) == G.W)
    G3 = build_graph(data, use_pygsp=False)
    G2 = G3.to_igraph()
    assert isinstance(G2, igraph.Graph)
    assert np.all(np.array(G2.get_adjacency(
        attribute="weight").data) == G.W)


@warns(UserWarning)
def test_to_pygsp_invalid_precomputed():
    G = build_graph(data)
    G2 = G.to_pygsp(precomputed='adjacency')


@warns(UserWarning)
def test_to_pygsp_invalid_use_pygsp():
    G = build_graph(data)
    G2 = G.to_pygsp(use_pygsp=False)

#####################################################
# Check parameters
#####################################################


@raises(TypeError)
def test_unknown_parameter():
    build_graph(data, hello='world')


@raises(ValueError)
def test_invalid_graphtype():
    build_graph(data, graphtype='hello world')
