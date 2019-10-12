from __future__ import print_function
from sklearn.utils.graph import graph_shortest_path
from load_tests import (
    graphtools,
    np,
    sp,
    pygsp,
    nose2,
    data,
    build_graph,
    assert_raises,
    raises,
    warns,
    squareform,
    pdist,
    PCA,
    TruncatedSVD
)

#####################################################
# Check parameters
#####################################################


@raises(ValueError)
def test_sample_idx_and_precomputed():
    build_graph(squareform(pdist(data)), n_pca=None,
                sample_idx=np.arange(10),
                precomputed='distance',
                decay=10)


@raises(ValueError)
def test_invalid_precomputed():
    build_graph(squareform(pdist(data)), n_pca=None,
                precomputed='hello world',
                decay=10)


@raises(ValueError)
def test_precomputed_not_square():
    build_graph(data, n_pca=None, precomputed='distance',
                decay=10)


@raises(ValueError)
def test_build_exact_with_sample_idx():
    build_graph(data, graphtype='exact', sample_idx=np.arange(len(data)),
                decay=10)


@warns(RuntimeWarning)
def test_precomputed_with_pca():
    build_graph(squareform(pdist(data)),
                precomputed='distance',
                n_pca=20,
                decay=10)


@raises(ValueError)
def test_exact_no_decay():
    build_graph(data, graphtype='exact',
                decay=None)


@raises(ValueError)
def test_exact_no_knn_no_bandwidth():
    build_graph(data, graphtype='exact',
                knn=None, bandwidth=None)


@raises(ValueError)
def test_precomputed_negative():
    build_graph(np.random.normal(0, 1, [200, 200]),
                precomputed='distance',
                n_pca=None)


@raises(ValueError)
def test_precomputed_invalid():
    build_graph(np.random.uniform(0, 1, [200, 200]),
                precomputed='invalid',
                n_pca=None)


@warns(RuntimeWarning)
def test_duplicate_data():
    build_graph(np.vstack([data, data[:10]]),
                n_pca=20,
                decay=10,
                thresh=0)


@warns(RuntimeWarning)
def test_many_duplicate_data():
    build_graph(np.vstack([data, data]),
                n_pca=20,
                decay=10,
                thresh=0)


@warns(UserWarning)
def test_k_too_large():
    build_graph(data,
                n_pca=20,
                decay=10,
                knn=len(data) - 1,
                thresh=0)


#####################################################
# Check kernel
#####################################################


def test_exact_graph():
    k = 3
    a = 13
    n_pca = 20
    bandwidth_scale = 1.3
    data_small = data[np.random.choice(
        len(data), len(data) // 2, replace=False)]
    pca = PCA(n_pca, svd_solver='randomized', random_state=42).fit(data_small)
    data_small_nu = pca.transform(data_small)
    pdx = squareform(pdist(data_small_nu, metric='euclidean'))
    knn_dist = np.partition(pdx, k, axis=1)[:, :k]
    epsilon = np.max(knn_dist, axis=1) * bandwidth_scale
    weighted_pdx = (pdx.T / epsilon).T
    K = np.exp(-1 * weighted_pdx**a)
    W = K + K.T
    W = np.divide(W, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(data_small, thresh=0, n_pca=n_pca,
                     decay=a, knn=k - 1, random_state=42,
                     bandwidth_scale=bandwidth_scale,
                     use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(pdx, n_pca=None, precomputed='distance',
                     bandwidth_scale=bandwidth_scale,
                     decay=a, knn=k - 1, random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(sp.coo_matrix(K), n_pca=None,
                     precomputed='affinity',
                     random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(K, n_pca=None,
                     precomputed='affinity',
                     random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(W, n_pca=None,
                     precomputed='adjacency',
                     random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))


def test_truncated_exact_graph():
    k = 3
    a = 13
    n_pca = 20
    thresh = 1e-4
    data_small = data[np.random.choice(
        len(data), len(data) // 2, replace=False)]
    pca = PCA(n_pca, svd_solver='randomized', random_state=42).fit(data_small)
    data_small_nu = pca.transform(data_small)
    pdx = squareform(pdist(data_small_nu, metric='euclidean'))
    knn_dist = np.partition(pdx, k, axis=1)[:, :k]
    epsilon = np.max(knn_dist, axis=1)
    weighted_pdx = (pdx.T / epsilon).T
    K = np.exp(-1 * weighted_pdx**a)
    K[K < thresh] = 0
    W = K + K.T
    W = np.divide(W, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(data_small, thresh=thresh,
                     graphtype='exact',
                     n_pca=n_pca,
                     decay=a, knn=k - 1, random_state=42,
                     use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(pdx, n_pca=None, precomputed='distance',
                     thresh=thresh,
                     decay=a, knn=k - 1, random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(K, n_pca=None,
                     precomputed='affinity',
                     thresh=thresh,
                     random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(W, n_pca=None,
                     precomputed='adjacency',
                     random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))


def test_truncated_exact_graph_sparse():
    k = 3
    a = 13
    n_pca = 20
    thresh = 1e-4
    data_small = data[np.random.choice(
        len(data), len(data) // 2, replace=False)]
    pca = TruncatedSVD(n_pca,
                       random_state=42).fit(data_small)
    data_small_nu = pca.transform(data_small)
    pdx = squareform(pdist(data_small_nu, metric='euclidean'))
    knn_dist = np.partition(pdx, k, axis=1)[:, :k]
    epsilon = np.max(knn_dist, axis=1)
    weighted_pdx = (pdx.T / epsilon).T
    K = np.exp(-1 * weighted_pdx**a)
    K[K < thresh] = 0
    W = K + K.T
    W = np.divide(W, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(sp.coo_matrix(data_small), thresh=thresh,
                     graphtype='exact',
                     n_pca=n_pca,
                     decay=a, knn=k - 1, random_state=42,
                     use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_allclose(G2.W.toarray(), G.W.toarray())
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(sp.bsr_matrix(pdx), n_pca=None, precomputed='distance',
                     thresh=thresh,
                     decay=a, knn=k - 1, random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(sp.lil_matrix(K), n_pca=None,
                     precomputed='affinity',
                     thresh=thresh,
                     random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(sp.dok_matrix(W), n_pca=None,
                     precomputed='adjacency',
                     random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))


def test_truncated_exact_graph_no_pca():
    k = 3
    a = 13
    n_pca = None
    thresh = 1e-4
    data_small = data[np.random.choice(
        len(data), len(data) // 10, replace=False)]
    pdx = squareform(pdist(data_small, metric='euclidean'))
    knn_dist = np.partition(pdx, k, axis=1)[:, :k]
    epsilon = np.max(knn_dist, axis=1)
    weighted_pdx = (pdx.T / epsilon).T
    K = np.exp(-1 * weighted_pdx**a)
    K[K < thresh] = 0
    W = K + K.T
    W = np.divide(W, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(data_small, thresh=thresh,
                     graphtype='exact',
                     n_pca=n_pca,
                     decay=a, knn=k - 1, random_state=42,
                     use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(sp.csr_matrix(data_small), thresh=thresh,
                     graphtype='exact',
                     n_pca=n_pca,
                     decay=a, knn=k - 1, random_state=42,
                     use_pygsp=True)
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))


def test_exact_graph_fixed_bandwidth():
    decay = 2
    knn = None
    bandwidth = 2
    n_pca = 20
    pca = PCA(n_pca, svd_solver='randomized', random_state=42).fit(data)
    data_nu = pca.transform(data)
    pdx = squareform(pdist(data_nu, metric='euclidean'))
    K = np.exp(-1 * (pdx / bandwidth)**decay)
    K = K + K.T
    W = np.divide(K, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(data, n_pca=n_pca,
                     graphtype='exact', knn=knn,
                     decay=decay, bandwidth=bandwidth,
                     random_state=42,
                     thresh=0,
                     use_pygsp=True)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    assert(G.N == G2.N)
    np.testing.assert_allclose(G.dw, G2.dw)
    np.testing.assert_allclose((G2.W - G.W).data, 0, atol=1e-14)
    bandwidth = np.random.gamma(5, 0.5, len(data))
    K = np.exp(-1 * (pdx.T / bandwidth).T**decay)
    K = K + K.T
    W = np.divide(K, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(data, n_pca=n_pca,
                     graphtype='exact', knn=knn,
                     decay=decay, bandwidth=bandwidth,
                     random_state=42,
                     thresh=0,
                     use_pygsp=True)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    assert(G.N == G2.N)
    np.testing.assert_allclose(G.dw, G2.dw)
    np.testing.assert_allclose((G2.W - G.W).data, 0, atol=1e-14)


def test_exact_graph_callable_bandwidth():
    decay = 2
    knn = 5
    bandwidth = lambda x: 2
    n_pca = 20
    thresh = 1e-4
    pca = PCA(n_pca, svd_solver='randomized', random_state=42).fit(data)
    data_nu = pca.transform(data)
    pdx = squareform(pdist(data_nu, metric='euclidean'))
    K = np.exp(-1 * (pdx / bandwidth(pdx))**decay)
    K[K < thresh] = 0
    K = K + K.T
    W = np.divide(K, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(data, n_pca=n_pca, knn=knn - 1,
                     decay=decay, bandwidth=bandwidth,
                     random_state=42,
                     thresh=thresh,
                     use_pygsp=True)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G2.W != G.W).sum() == 0)
    assert((G.W != G2.W).nnz == 0)
    bandwidth = lambda x: np.percentile(x, 10, axis=1)
    K = np.exp(-1 * (pdx / bandwidth(pdx))**decay)
    K[K < thresh] = 0
    K = K + K.T
    W = np.divide(K, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(data, n_pca=n_pca, knn=knn - 1,
                     decay=decay, bandwidth=bandwidth,
                     random_state=42,
                     thresh=thresh,
                     use_pygsp=True)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    assert(G.N == G2.N)
    np.testing.assert_allclose(G.dw, G2.dw)
    np.testing.assert_allclose((G2.W - G.W).data, 0, atol=1e-14)


#####################################################
# Check anisotropy
#####################################################

def test_exact_graph_anisotropy():
    k = 3
    a = 13
    n_pca = 20
    anisotropy = 0.9
    data_small = data[np.random.choice(
        len(data), len(data) // 2, replace=False)]
    pca = PCA(n_pca, svd_solver='randomized', random_state=42).fit(data_small)
    data_small_nu = pca.transform(data_small)
    pdx = squareform(pdist(data_small_nu, metric='euclidean'))
    knn_dist = np.partition(pdx, k, axis=1)[:, :k]
    epsilon = np.max(knn_dist, axis=1)
    weighted_pdx = (pdx.T / epsilon).T
    K = np.exp(-1 * weighted_pdx**a)
    K = K + K.T
    K = np.divide(K, 2)
    d = K.sum(1)
    W = K / (np.outer(d, d) ** anisotropy)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(data_small, thresh=0, n_pca=n_pca,
                     decay=a, knn=k - 1, random_state=42,
                     use_pygsp=True, anisotropy=anisotropy)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    assert(G.N == G2.N)
    np.testing.assert_equal(G.dw, G2.dw)
    assert((G2.W != G.W).sum() == 0)
    assert((G.W != G2.W).nnz == 0)
    assert_raises(ValueError, build_graph,
                  data_small, thresh=0, n_pca=n_pca,
                  decay=a, knn=k - 1, random_state=42,
                  use_pygsp=True, anisotropy=-1)
    assert_raises(ValueError, build_graph,
                  data_small, thresh=0, n_pca=n_pca,
                  decay=a, knn=k - 1, random_state=42,
                  use_pygsp=True, anisotropy=2)
    assert_raises(ValueError, build_graph,
                  data_small, thresh=0, n_pca=n_pca,
                  decay=a, knn=k - 1, random_state=42,
                  use_pygsp=True, anisotropy='invalid')

#####################################################
# Check extra functionality
#####################################################


def test_shortest_path_affinity():
    data_small = data[np.random.choice(
        len(data), len(data) // 4, replace=False)]
    G = build_graph(data_small, knn=5, decay=15)
    D = -1 * np.where(G.K != 0, np.log(np.where(G.K != 0, G.K, np.nan)), 0)
    P = graph_shortest_path(D)
    # sklearn returns 0 if no path exists
    P[np.where(P == 0)] = np.inf
    # diagonal should actually be zero
    np.fill_diagonal(P, 0)
    np.testing.assert_allclose(P, G.shortest_path(distance='affinity'))
    np.testing.assert_allclose(P, G.shortest_path())


def test_shortest_path_affinity_precomputed():
    data_small = data[np.random.choice(
        len(data), len(data) // 4, replace=False)]
    G = build_graph(data_small, knn=5, decay=15)
    G = graphtools.Graph(G.K, precomputed='affinity')
    D = -1 * np.where(G.K != 0, np.log(np.where(G.K != 0, G.K, np.nan)), 0)
    P = graph_shortest_path(D)
    # sklearn returns 0 if no path exists
    P[np.where(P == 0)] = np.inf
    # diagonal should actually be zero
    np.fill_diagonal(P, 0)
    np.testing.assert_allclose(P, G.shortest_path(distance='affinity'))
    np.testing.assert_allclose(P, G.shortest_path())


@raises(NotImplementedError)
def test_shortest_path_decay_constant():
    data_small = data[np.random.choice(
        len(data), len(data) // 4, replace=False)]
    G = build_graph(data_small, knn=5, decay=15)
    G.shortest_path(distance='constant')


@raises(NotImplementedError)
def test_shortest_path_precomputed_decay_constant():
    data_small = data[np.random.choice(
        len(data), len(data) // 4, replace=False)]
    G = build_graph(data_small, knn=5, decay=15)
    G = graphtools.Graph(G.K, precomputed='affinity')
    G.shortest_path(distance='constant')


@raises(NotImplementedError)
def test_shortest_path_decay_data():
    data_small = data[np.random.choice(
        len(data), len(data) // 4, replace=False)]
    G = build_graph(data_small, knn=5, decay=15)
    G.shortest_path(distance='data')


@raises(ValueError)
def test_shortest_path_precomputed_data():
    data_small = data[np.random.choice(
        len(data), len(data) // 4, replace=False)]
    G = build_graph(data_small, knn=5, decay=15)
    G = graphtools.Graph(G.K, precomputed='affinity')
    G.shortest_path(distance='data')


#####################################################
# Check interpolation
#####################################################


def test_build_dense_exact_kernel_to_data(**kwargs):
    G = build_graph(data, decay=10, thresh=0)
    n = G.data.shape[0]
    K = G.build_kernel_to_data(data[:n // 2, :])
    assert(K.shape == (n // 2, n))
    K = G.build_kernel_to_data(G.data, knn=G.knn + 1)
    np.testing.assert_equal(G.kernel - (K + K.T) / 2, 0)
    K = G.build_kernel_to_data(G.data_nu, knn=G.knn + 1)
    np.testing.assert_equal(G.kernel - (K + K.T) / 2, 0)


def test_build_dense_exact_callable_bw_kernel_to_data(**kwargs):
    G = build_graph(data, decay=10, thresh=0, bandwidth=lambda x: x.mean(1))
    n = G.data.shape[0]
    K = G.build_kernel_to_data(data[:n // 2, :])
    assert(K.shape == (n // 2, n))
    K = G.build_kernel_to_data(G.data, knn=G.knn + 1)
    np.testing.assert_equal(G.kernel - (K + K.T) / 2, 0)
    K = G.build_kernel_to_data(G.data_nu, knn=G.knn + 1)
    np.testing.assert_equal(G.kernel - (K + K.T) / 2, 0)


def test_build_sparse_exact_kernel_to_data(**kwargs):
    G = build_graph(data, decay=10, thresh=0, sparse=True)
    n = G.data.shape[0]
    K = G.build_kernel_to_data(data[:n // 2, :])
    assert(K.shape == (n // 2, n))
    K = G.build_kernel_to_data(G.data, knn=G.knn + 1)
    np.testing.assert_equal(G.kernel - (K + K.T) / 2, 0)
    K = G.build_kernel_to_data(G.data_nu, knn=G.knn + 1)
    np.testing.assert_equal(G.kernel - (K + K.T) / 2, 0)


def test_exact_interpolate():
    G = build_graph(data, decay=10, thresh=0)
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


####################
# Test API
####################


def test_verbose():
    print()
    print("Verbose test: Exact")
    build_graph(data, decay=10, thresh=0, verbose=True)


def test_set_params():
    G = build_graph(data, decay=10, thresh=0)
    assert G.get_params() == {'n_pca': 20,
                              'random_state': 42,
                              'kernel_symm': '+',
                              'theta': None,
                              'knn': 3,
                              'anisotropy': 0,
                              'decay': 10,
                              'bandwidth': None,
                              'bandwidth_scale': 1,
                              'distance': 'euclidean',
                              'precomputed': None}
    assert_raises(ValueError, G.set_params, knn=15)
    assert_raises(ValueError, G.set_params, decay=15)
    assert_raises(ValueError, G.set_params, distance='manhattan')
    assert_raises(ValueError, G.set_params, precomputed='distance')
    assert_raises(ValueError, G.set_params, bandwidth=5)
    assert_raises(ValueError, G.set_params, bandwidth_scale=5)
    G.set_params(knn=G.knn,
                 decay=G.decay,
                 distance=G.distance,
                 precomputed=G.precomputed)
