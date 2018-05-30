from . import (
    graphtools,
    np,
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
)

#####################################################
# Check parameters
#####################################################


@raises(ValueError)
def test_sample_idx_and_precomputed():
    build_graph(squareform(pdist(data)), n_pca=None,
                sample_idx=np.arange(10),
                precomputed='distance')


@raises(ValueError)
def test_invalid_precomputed():
    build_graph(squareform(pdist(data)), n_pca=None,
                precomputed='hello world')


@raises(ValueError)
def test_precomputed_not_square():
    build_graph(data, n_pca=None, precomputed='distance')


@raises(ValueError)
def test_build_exact_with_sample_idx():
    build_graph(data, graphtype='exact', sample_idx=np.arange(len(data)))


@warns(RuntimeWarning)
def test_precomputed_with_pca():
    build_graph(squareform(pdist(data)),
                precomputed='distance',
                n_pca=20)


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
    W = K + K.T
    W = np.divide(W, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(data, thresh=0, n_pca=n_pca,
                     decay=a, knn=k, random_state=42,
                     use_pygsp=True)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(pdx, n_pca=None, precomputed='distance',
                     decay=a, knn=k, random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(K, n_pca=None,
                     precomputed='affinity',
                     random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(W, n_pca=None,
                     precomputed='adjacency',
                     random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
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
    W = K + K.T
    W = np.divide(W, 2)
    W[W < thresh] = 0
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(data_small, thresh=thresh,
                     graphtype='exact',
                     n_pca=n_pca,
                     decay=a, knn=k, random_state=42,
                     use_pygsp=True)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(pdx, n_pca=None, precomputed='distance',
                     thresh=thresh,
                     decay=a, knn=k, random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(K, n_pca=None,
                     precomputed='affinity',
                     thresh=thresh,
                     random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))
    G2 = build_graph(W, n_pca=None,
                     precomputed='adjacency',
                     random_state=42, use_pygsp=True)
    assert(G.N == G2.N)
    assert(np.all(G.d == G2.d))
    assert((G.W != G2.W).nnz == 0)
    assert((G2.W != G.W).sum() == 0)
    assert(isinstance(G2, graphtools.graphs.TraditionalGraph))


#####################################################
# Check interpolation
#####################################################


def test_build_dense_exact_kernel_to_data(**kwargs):
    G = build_graph(data, decay=10, thresh=0)
    n = G.data.shape[0]
    K = G.build_kernel_to_data(data[:n // 2, :])
    assert(K.shape == (n // 2, n))
    K = G.build_kernel_to_data(G.data)
    assert(np.sum(G.kernel != (K + K.T) / 2) == 0)
    K = G.build_kernel_to_data(G.data_nu)
    assert(np.sum(G.kernel != (K + K.T) / 2) == 0)


def test_build_sparse_exact_kernel_to_data(**kwargs):
    G = build_graph(data, decay=10, thresh=0, sparse=True)
    n = G.data.shape[0]
    K = G.build_kernel_to_data(data[:n // 2, :])
    assert(K.shape == (n // 2, n))
    K = G.build_kernel_to_data(G.data)
    assert(np.sum(G.kernel != (K + K.T) / 2) == 0)
    K = G.build_kernel_to_data(G.data_nu)
    assert(np.sum(G.kernel != (K + K.T) / 2) == 0)


def test_exact_interpolate():
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


def test_verbose():
    build_graph(data, verbose=True)


if __name__ == "__main__":
    exit(nose2.run())
