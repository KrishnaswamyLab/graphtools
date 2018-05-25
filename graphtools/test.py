from sklearn.decomposition import PCA
from sklearn import datasets
from scipy.spatial.distance import pdist, cdist, squareform
import pygsp
import graphtools
import numpy as np
import scipy.sparse as sp
import warnings
import pandas as pd
import sklearn.utils
from scipy import stats
from mpl_toolkits.mplot3d.axes3d import Axes3D

import nose
from nose.tools import raises, assert_raises, make_decorator
warnings.filterwarnings("error")

global digits
global data
digits = datasets.load_digits()
data = digits['data']


def generate_swiss_roll(n_samples=3000, noise=0.5, seed=42):
    generator = np.random.RandomState(seed)
    t = 1.5 * np.pi * (1 + 2 * generator.rand(1, n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    sample_idx = np.random.choice([0, 1], n_samples, replace=True)
    z = sample_idx
    t = np.squeeze(t)
    X = np.concatenate((x, y))
    X += noise * generator.randn(2, n_samples)
    X = X.T[np.argsort(t)]
    X = np.hstack((X, z.reshape(3000, 1)))
    return X, sample_idx


def build_graph(data, n_pca=20, thresh=0,
                decay=10, knn=3,
                random_state=42,
                sparse=False,
                graph_class=graphtools.Graph,
                **kwargs):
    if sparse:
        data = sp.coo_matrix(data)
    return graph_class(data, thresh=thresh, n_pca=n_pca,
                       decay=decay, knn=knn,
                       random_state=42, **kwargs)


def warns(*warns):
    """Test must raise one of expected warnings to pass.
    Example use::
      @warns(RuntimeWarning, UserWarning)
      def test_raises_type_error():
          warnings.warn("This test passes", RuntimeWarning)
      @warns(ImportWarning)
      def test_that_fails_by_passing():
          pass
    """
    valid = ' or '.join([w.__name__ for w in warns])

    def decorate(func):
        name = func.__name__

        def newfunc(*arg, **kw):
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("default")
                func(*arg, **kw)
                warnings.filterwarnings("error")
            try:
                for warn in w:
                    raise warn.category
            except warns:
                pass
            except:
                raise
            else:
                message = "%s() did not raise %s" % (name, valid)
                raise AssertionError(message)
        newfunc = make_decorator(func)(newfunc)
        return newfunc
    return decorate


#####################################################
# Check parameters
#####################################################


@raises(ValueError)
def test_1d_data():
    build_graph(data[:, 0])


@raises(ValueError)
def test_3d_data():
    build_graph(data[:, :, None])


@raises(ValueError)
def test_sample_idx_and_precomputed():
    build_graph(data, n_pca=None, sample_idx=np.arange(10),
                precomputed='distance')


@raises(ValueError)
def test_sample_idx_wrong_length():
    build_graph(data, graphtype='mnn', sample_idx=np.arange(10))


@raises(ValueError)
def test_sample_idx_unique():
    build_graph(data, graph_class=graphtools.MNNGraph,
                sample_idx=np.ones(len(data)))


@raises(ValueError)
def test_sample_idx_none():
    build_graph(data, graphtype='mnn', sample_idx=None)


@raises(ValueError)
def test_invalid_precomputed():
    build_graph(data, n_pca=None, precomputed='hello world')


@raises(ValueError)
def test_precomputed_not_square():
    build_graph(data, n_pca=None, precomputed='distance')


@raises(ValueError)
def test_build_knn_with_exact_alpha():
    build_graph(data, graphtype='knn', decay=10, thresh=0)


@raises(ValueError)
def test_build_knn_with_precomputed():
    build_graph(data, n_pca=None, graphtype='knn', precomputed='distance')


@raises(ValueError)
def test_build_mnn_with_precomputed():
    build_graph(data, n_pca=None, graphtype='mnn', precomputed='distance')


@raises(ValueError)
def test_build_knn_with_sample_idx():
    build_graph(data, graphtype='knn', sample_idx=np.arange(len(data)))


@raises(ValueError)
def test_build_exact_with_sample_idx():
    build_graph(data, graphtype='exact', sample_idx=np.arange(len(data)))


@raises(ValueError)
def test_invalid_graphtype():
    build_graph(data, graphtype='hello world')


@raises(ValueError)
def test_build_landmark_with_too_many_landmarks():
    build_graph(data, n_landmark=len(data))


@warns(RuntimeWarning)
def test_build_landmark_with_too_few_points():
    build_graph(data[:50], n_landmark=25, n_svd=100)


@warns(RuntimeWarning)
def test_too_many_n_pca():
    build_graph(data, n_pca=data.shape[1])


@warns(RuntimeWarning)
def test_precomputed_with_pca():
    build_graph(squareform(pdist(data)),
                precomputed='distance',
                n_pca=20)


#####################################################
# Check data
#####################################################


def test_transform():
    G = build_graph(data, n_pca=20)
    assert(np.all(G.data_nu == G.transform(G.data)))
    assert_raises(ValueError, G.transform, G.data[:, 0])
    assert_raises(ValueError, G.transform, G.data[:, None, :15])
    assert_raises(ValueError, G.transform, G.data[:, :15])
    G = build_graph(data, n_pca=None)
    assert(np.all(G.data_nu == G.transform(G.data)))
    assert_raises(ValueError, G.transform, G.data[:, :15])
    G = build_graph(data, sparse=True, n_pca=20)
    assert(np.all(G.data_nu == G.transform(G.data)))
    assert_raises(ValueError, G.transform, sp.csr_matrix(G.data)[:, :15])
    G = build_graph(data, sparse=True, n_pca=None)
    assert(np.all(G.data_nu == G.transform(G.data)))
    assert_raises(ValueError, G.transform, sp.csr_matrix(G.data)[:, :15])

#####################################################
# Check kernel
#####################################################


def test_exact_graph():
    k = 3
    a = 13
    n_pca = 20
    pca = PCA(n_pca, svd_solver='randomized', random_state=42).fit(data)
    data_nu = pca.transform(data)
    pdx = squareform(pdist(data_nu, metric='euclidean'))
    knn_dist = np.partition(pdx, k, axis=1)[:, :k]
    epsilon = np.max(knn_dist, axis=1)
    weighted_pdx = (pdx.T / epsilon).T
    K = np.exp(-1 * weighted_pdx**a)
    K = K + K.T
    W = np.divide(K, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(data, thresh=0, n_pca=n_pca,
                     decay=a, knn=k, random_state=42)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.TraditionalGraph))
    G2 = build_graph(pdx, n_pca=None, precomputed='distance',
                     decay=a, knn=k, random_state=42)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.TraditionalGraph))
    G2 = build_graph(K / 2, n_pca=None,
                     precomputed='affinity',
                     random_state=42)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.TraditionalGraph))
    G2 = build_graph(W, n_pca=None,
                     precomputed='adjacency',
                     random_state=42)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.TraditionalGraph))


def test_knn_graph():
    k = 3
    n_pca = 20
    pca = PCA(n_pca, svd_solver='randomized', random_state=42).fit(data)
    data_nu = pca.transform(data)
    pdx = squareform(pdist(data_nu, metric='euclidean'))
    knn_dist = np.partition(pdx, k, axis=1)[:, :k]
    epsilon = np.max(knn_dist, axis=1)
    K = np.empty_like(pdx)
    for i in range(len(pdx)):
        K[i, pdx[i, :] <= epsilon[i]] = 1
        K[i, pdx[i, :] > epsilon[i]] = 0

    K = K + K.T
    W = np.divide(K, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(data, n_pca=n_pca,
                     decay=None, knn=k, random_state=42)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.kNNGraph))


def test_sparse_alpha_knn_graph():
    data = datasets.make_swiss_roll(n_samples=5000)[0]
    k = 5
    a = 0.45
    thresh = 0.01
    pdx = squareform(pdist(data, metric='euclidean'))
    knn_dist = np.partition(pdx, k, axis=1)[:, :k]
    epsilon = np.max(knn_dist, axis=1)
    pdx = (pdx.T / epsilon).T
    K = np.exp(-1 * pdx**a)
    K = K + K.T
    W = np.divide(K, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(data, n_pca=None,  # n_pca,
                     decay=a, knn=k, thresh=thresh,
                     random_state=42)
    assert(np.abs(G.W - G2.W).max() < thresh)
    assert(G.N == G2.N)
    assert(isinstance(G2, graphtools.kNNGraph))


def test_mnn_graph_float_gamma():
    X, sample_idx = generate_swiss_roll()
    gamma = 0.9
    k = 10
    a = 20
    metric = 'euclidean'
    beta = 0
    samples = np.unique(sample_idx)

    K = np.zeros((len(X), len(X)))
    K[:] = np.nan
    K = pd.DataFrame(K)

    for si in samples:
        X_i = X[sample_idx == si]            # get observations in sample i
        for sj in samples:
            X_j = X[sample_idx == sj]        # get observation in sample j
            pdx_ij = cdist(X_i, X_j, metric=metric)  # pairwise distances
            kdx_ij = np.sort(pdx_ij, axis=1)  # get kNN
            e_ij = kdx_ij[:, k]             # dist to kNN
            pdxe_ij = pdx_ij / e_ij[:, np.newaxis]  # normalize
            k_ij = np.exp(-1 * (pdxe_ij ** a))  # apply α-decaying kernel
            if si == sj:
                K.iloc[sample_idx == si, sample_idx == sj] = k_ij * \
                    (1 - beta)  # fill out values in K for NN on diagnoal
            else:
                # fill out values in K for NN on diagnoal
                K.iloc[sample_idx == si, sample_idx == sj] = k_ij

    W = np.array((gamma * np.minimum(K, K.T)) +
                 ((1 - gamma) * np.maximum(K, K.T)))
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = graphtools.Graph(X, knn=k + 1, decay=a, beta=1 - beta, gamma=gamma,
                          distance=metric, sample_idx=sample_idx, thresh=0)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.MNNGraph))


def test_mnn_graph_matrix_gamma():
    X, sample_idx = generate_swiss_roll()
    bs = 0.8
    gamma = np.array([[1, bs],  # 0
                      [bs,  1]])  # 3
    k = 10
    a = 20
    metric = 'euclidean'
    beta = 0
    samples = np.unique(sample_idx)

    K = np.zeros((len(X), len(X)))
    K[:] = np.nan
    K = pd.DataFrame(K)

    for si in samples:
        X_i = X[sample_idx == si]            # get observations in sample i
        for sj in samples:
            X_j = X[sample_idx == sj]        # get observation in sample j
            pdx_ij = cdist(X_i, X_j, metric=metric)  # pairwise distances
            kdx_ij = np.sort(pdx_ij, axis=1)  # get kNN
            e_ij = kdx_ij[:, k]             # dist to kNN
            pdxe_ij = pdx_ij / e_ij[:, np.newaxis]  # normalize
            k_ij = np.exp(-1 * (pdxe_ij ** a))  # apply α-decaying kernel
            if si == sj:
                K.iloc[sample_idx == si, sample_idx == sj] = k_ij * \
                    (1 - beta)  # fill out values in K for NN on diagnoal
            else:
                # fill out values in K for NN on diagnoal
                K.iloc[sample_idx == si, sample_idx == sj] = k_ij

    K = np.array(K)

    matrix_gamma = pd.DataFrame(np.zeros((len(sample_idx), len(sample_idx))))
    for ix, si in enumerate(set(sample_idx)):
        for jx, sj in enumerate(set(sample_idx)):
            matrix_gamma.iloc[sample_idx == si,
                              sample_idx == sj] = gamma[ix, jx]

    W = np.array((matrix_gamma * np.minimum(K, K.T)) +
                 ((1 - matrix_gamma) * np.maximum(K, K.T)))
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = graphtools.Graph(X, knn=k + 1, decay=a, beta=1 - beta, gamma=gamma,
                          distance=metric, sample_idx=sample_idx, thresh=0)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.MNNGraph))


def test_mnn_graph_error():
    n_sample = len(np.unique(digits['target']))
    assert_raises(ValueError, build_graph,
                  data, thresh=0, n_pca=20,
                  decay=10, knn=5, random_state=42,
                  sample_idx=digits['target'],
                  gamma=np.tile(np.linspace(0, 1, n_sample - 1),
                                n_sample).reshape(n_sample - 1, n_sample))
    assert_raises(ValueError, build_graph,
                  data, thresh=0, n_pca=20,
                  decay=10, knn=5, random_state=42,
                  sample_idx=digits['target'],
                  gamma=np.linspace(0, 1, n_sample - 1))


def test_landmark_exact_graph():
    n_landmark = 100
    # exact graph
    G = build_graph(data, n_landmark=n_landmark,
                    thresh=0, n_pca=20,
                    decay=10, knn=5, random_state=42)
    assert(G.landmark_op.shape == (n_landmark, n_landmark))
    assert(isinstance(G, graphtools.TraditionalGraph))
    assert(isinstance(G, graphtools.LandmarkGraph))


def test_landmark_knn_graph():
    n_landmark = 500
    # knn graph
    G = build_graph(data, n_landmark=n_landmark, n_pca=20,
                    decay=None, knn=5, random_state=42)
    assert(G.landmark_op.shape == (n_landmark, n_landmark))
    assert(isinstance(G, graphtools.kNNGraph))
    assert(isinstance(G, graphtools.LandmarkGraph))


def test_landmark_mnn_graph():
    n_landmark = 500
    # mnn graph
    G = build_graph(data, n_landmark=n_landmark,
                    thresh=1e-5, n_pca=20,
                    decay=10, knn=5, random_state=42,
                    sample_idx=digits['target'])
    assert(G.landmark_op.shape == (n_landmark, n_landmark))
    assert(isinstance(G, graphtools.MNNGraph))
    assert(isinstance(G, graphtools.LandmarkGraph))


#####################################################
# Check interpolation
#####################################################


def _test_build_kernel_to_data(**kwargs):
    G = build_graph(data, **kwargs)
    K = G.build_kernel_to_data(G.data)
    assert(np.sum(G.kernel != (K + K.T) / 2) == 0)
    K = G.build_kernel_to_data(G.data_nu)
    assert(np.sum(G.kernel != (K + K.T) / 2) == 0)


def test_build_exact_kernel_to_data(**kwargs):
    _test_build_kernel_to_data(decay=10)
    _test_build_kernel_to_data(decay=10, sparse=True)


def test_build_knn_kernel_to_data():
    _test_build_kernel_to_data(decay=None)
    _test_build_kernel_to_data(decay=None, sparse=True)


def test_interpolate():
    G = build_graph(data)
    assert_raises(ValueError, G.interpolate, data)
    pca_data = PCA(2).fit_transform(data)
    transitions = G.extend_to_data(data)
    assert(np.all(G.interpolate(pca_data, Y=data) ==
                  G.interpolate(pca_data, transitions=transitions)))


@raises(ValueError)
def test_precomputed_interpolate():
    G = build_graph(squareform(pdist(data)), n_pca=None,
                    precomputed='distance')
    G.build_kernel_to_data(data)


if __name__ == "__main__":
    exit(nose.run())
