from sklearn.decomposition import PCA
from sklearn import datasets
from scipy.spatial.distance import pdist, squareform
from scipy.io import loadmat
import pygsp
import graphtools
import numpy as np
import scipy.sparse as sp


def digits_graph(n_pca=20, thresh=0,
                 decay=10, knn=3,
                 random_state=42,
                 sparse=False):
    digits = datasets.load_digits()
    data = digits['data']
    if sparse:
        data = sp.coo_matrix(data)
    return graphtools.Graph(data, thresh=0, n_pca=n_pca,
                            decay=decay, knn=knn, random_state=42)


def test_digits():
    digits = datasets.load_digits()
    X = digits['data']
    k = 3
    a = 13
    n_pca = 20
    pca = PCA(n_pca, svd_solver='randomized', random_state=42)
    data = pca.fit_transform(X)
    pdx = squareform(pdist(data, metric='euclidean'))
    knn_dist = np.partition(pdx, k, axis=1)[:, :k]
    epsilon = np.max(knn_dist, axis=1)
    pdx = (pdx / epsilon).T
    K = np.exp(-1 * pdx**a)
    K = K + K.T
    W = np.divide(K, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = digits_graph(thresh=0, n_pca=n_pca,
                      decay=a, knn=k, random_state=42)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)


def test_transform():
    G = digits_graph(n_pca=20)
    assert(np.all(G.data_nu == G.transform(G.data)))
    G = digits_graph(n_pca=None)
    assert(np.all(G.data_nu == G.transform(G.data)))
    G = digits_graph(sparse=True, n_pca=20)
    assert(np.all(G.data_nu == G.transform(G.data)))

if __name__ == "__main__":
    test_transform()
    test_digits()
