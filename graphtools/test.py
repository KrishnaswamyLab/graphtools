from sklearn.decomposition import PCA
from sklearn import datasets
from scipy.spatial.distance import pdist, squareform
from scipy.io import loadmat
import pygsp
import graphtools
import numpy as np


def test_digits():
    digits = datasets.load_digits()
    X = digits['data']
    k = 3
    a = 13
    n_pca = 20
    pca = PCA(n_pca, svd_solver='randomized')
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
    G2 = graphtools.Graph(data, thresh=0, n_pca=n_pca, decay=a, knn=k)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
