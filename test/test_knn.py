from __future__ import print_function, division
from sklearn.utils.graph import graph_shortest_path
from scipy.spatial.distance import pdist, squareform
from load_tests import assert_raises_message, assert_warns_message
from nose.tools import assert_raises_regex, assert_warns_regex
import warnings
from load_tests import (
    graphtools,
    np,
    sp,
    pygsp,
    data,
    datasets,
    build_graph,
    PCA,
    TruncatedSVD,
)


#####################################################
# Check parameters
#####################################################


def test_build_knn_with_exact_alpha():
    with assert_raises_message(
        ValueError,
        "Cannot instantiate a kNNGraph with `decay=None`, `thresh=0` and `knn_max=None`. Use a TraditionalGraph instead.",
    ):
        build_graph(data, graphtype="knn", decay=10, thresh=0)


def test_build_knn_with_precomputed():
    with assert_raises_message(
        ValueError,
        "kNNGraph does not support precomputed values. Use `graphtype='exact'` or `precomputed=None`",
    ):
        build_graph(data, n_pca=None, graphtype="knn", precomputed="distance")


def test_build_knn_with_sample_idx():
    with assert_raises_message(
        ValueError,
        "kNNGraph does not support batch correction. Use `graphtype='mnn'` or `sample_idx=None`",
    ):
        build_graph(data, graphtype="knn", sample_idx=np.arange(len(data)))


def test_duplicate_data():
    with assert_warns_regex(
        RuntimeWarning,
        r"Detected zero distance between samples ([0-9and,\s]*). Consider removing duplicates to avoid errors in downstream processing.",
    ):
        build_graph(np.vstack([data, data[:9]]), n_pca=20, decay=10, thresh=1e-4)


def test_duplicate_data_many():
    with assert_warns_regex(
        RuntimeWarning,
        "Detected zero distance between ([0-9]*) pairs of samples. Consider removing duplicates to avoid errors in downstream processing.",
    ):
        build_graph(np.vstack([data, data[:21]]), n_pca=20, decay=10, thresh=1e-4)


def test_balltree_cosine():
    with assert_warns_message(
        UserWarning,
        "Metric cosine not valid for `sklearn.neighbors.BallTree`. Graph instantiation may be slower than normal.",
    ):
        build_graph(data, n_pca=20, decay=10, distance="cosine", thresh=1e-4)


def test_k_too_large():
    with assert_warns_message(
        UserWarning,
        "Cannot set knn ({1}) to be greater than n_samples - 2 ({0}). Setting knn={0}".format(
            data.shape[0] - 2, data.shape[0] - 1
        ),
    ):
        build_graph(data, n_pca=20, decay=10, knn=len(data) - 1, thresh=1e-4)


def test_knnmax_too_large():
    with assert_warns_message(
        UserWarning,
        "Cannot set knn_max (9) to be less than knn (10). Setting knn_max=10",
    ):
        build_graph(data, n_pca=20, decay=10, knn=10, knn_max=9, thresh=1e-4)


def test_bandwidth_no_decay():
    with assert_warns_message(
        UserWarning, "`bandwidth` is not used when `decay=None`."
    ):
        build_graph(data, n_pca=20, decay=None, bandwidth=3, thresh=1e-4)


def test_knn_no_knn_no_bandwidth():
    with assert_raises_message(
        ValueError, "Either `knn` or `bandwidth` must be provided."
    ):
        build_graph(data, graphtype="knn", knn=None, bandwidth=None, thresh=1e-4)


def test_knn_graph_invalid_symm():
    with assert_raises_message(
        ValueError,
        "kernel_symm 'invalid' not recognized. Choose from '+', '*', 'mnn', or 'none'.",
    ):
        build_graph(data, graphtype="knn", knn=5, thresh=1e-4, kernel_symm="invalid")


#####################################################
# Check kernel
#####################################################


def test_knn_graph():
    k = 3
    n_pca = 20
    pca = PCA(n_pca, svd_solver="randomized", random_state=42).fit(data)
    data_nu = pca.transform(data)
    pdx = squareform(pdist(data_nu, metric="euclidean"))
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
    G2 = build_graph(
        data, n_pca=n_pca, decay=None, knn=k - 1, random_state=42, use_pygsp=True
    )
    assert G.N == G2.N
    np.testing.assert_equal(G.dw, G2.dw)
    assert (G.W - G2.W).nnz == 0
    assert (G2.W - G.W).sum() == 0
    assert isinstance(G2, graphtools.graphs.kNNGraph)

    K2 = G2.build_kernel_to_data(G2.data_nu, knn=k)
    K2 = (K2 + K2.T) / 2
    assert (G2.K - K2).nnz == 0
    assert (
        G2.build_kernel_to_data(G2.data_nu, knn=data.shape[0]).nnz
        == data.shape[0] * data.shape[0]
    )
    with assert_warns_message(
        UserWarning,
        "Cannot set knn ({}) to be greater than "
        "n_samples ({}). Setting knn={}".format(
            data.shape[0] + 1, data.shape[0], data.shape[0]
        ),
    ):
        G2.build_kernel_to_data(
            Y=G2.data_nu, knn=data.shape[0] + 1,
        )


def test_knn_graph_multiplication_symm():
    k = 3
    n_pca = 20
    pca = PCA(n_pca, svd_solver="randomized", random_state=42).fit(data)
    data_nu = pca.transform(data)
    pdx = squareform(pdist(data_nu, metric="euclidean"))
    knn_dist = np.partition(pdx, k, axis=1)[:, :k]
    epsilon = np.max(knn_dist, axis=1)
    K = np.empty_like(pdx)
    for i in range(len(pdx)):
        K[i, pdx[i, :] <= epsilon[i]] = 1
        K[i, pdx[i, :] > epsilon[i]] = 0

    W = K * K.T
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(
        data,
        n_pca=n_pca,
        decay=None,
        knn=k - 1,
        random_state=42,
        use_pygsp=True,
        kernel_symm="*",
    )
    assert G.N == G2.N
    np.testing.assert_equal(G.dw, G2.dw)
    assert (G.W - G2.W).nnz == 0
    assert (G2.W - G.W).sum() == 0
    assert isinstance(G2, graphtools.graphs.kNNGraph)


def test_knn_graph_sparse():
    k = 3
    n_pca = 20
    pca = TruncatedSVD(n_pca, random_state=42).fit(data)
    data_nu = pca.transform(data)
    pdx = squareform(pdist(data_nu, metric="euclidean"))
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
    G2 = build_graph(
        sp.coo_matrix(data),
        n_pca=n_pca,
        decay=None,
        knn=k - 1,
        random_state=42,
        use_pygsp=True,
    )
    assert G.N == G2.N
    np.testing.assert_allclose(G2.W.toarray(), G.W.toarray())
    assert isinstance(G2, graphtools.graphs.kNNGraph)


def test_sparse_alpha_knn_graph():
    data = datasets.make_swiss_roll()[0]
    k = 5
    a = 0.45
    thresh = 0.01
    bandwidth_scale = 1.3
    pdx = squareform(pdist(data, metric="euclidean"))
    knn_dist = np.partition(pdx, k, axis=1)[:, :k]
    epsilon = np.max(knn_dist, axis=1) * bandwidth_scale
    pdx = (pdx.T / epsilon).T
    K = np.exp(-1 * pdx ** a)
    K = K + K.T
    W = np.divide(K, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(
        data,
        n_pca=None,  # n_pca,
        decay=a,
        knn=k - 1,
        thresh=thresh,
        bandwidth_scale=bandwidth_scale,
        random_state=42,
        use_pygsp=True,
    )
    assert np.abs(G.W - G2.W).max() < thresh
    assert G.N == G2.N
    assert isinstance(G2, graphtools.graphs.kNNGraph)


def test_knnmax():
    data = datasets.make_swiss_roll()[0]
    k = 5
    k_max = 10
    a = 0.45
    thresh = 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "K should be symmetric", RuntimeWarning)
        G = build_graph(
            data,
            n_pca=None,  # n_pca,
            decay=a,
            knn=k - 1,
            knn_max=k_max - 1,
            thresh=0,
            random_state=42,
            kernel_symm=None,
        )
        assert np.all((G.K > 0).sum(axis=1) == k_max)

    pdx = squareform(pdist(data, metric="euclidean"))
    knn_dist = np.partition(pdx, k, axis=1)[:, :k]
    knn_max_dist = np.max(np.partition(pdx, k_max, axis=1)[:, :k_max], axis=1)
    epsilon = np.max(knn_dist, axis=1)
    pdx_scale = (pdx.T / epsilon).T
    K = np.where(pdx <= knn_max_dist[:, None], np.exp(-1 * pdx_scale ** a), 0)
    K = K + K.T
    W = np.divide(K, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(
        data,
        n_pca=None,  # n_pca,
        decay=a,
        knn=k - 1,
        knn_max=k_max - 1,
        thresh=0,
        random_state=42,
        use_pygsp=True,
    )
    assert isinstance(G2, graphtools.graphs.kNNGraph)
    assert G.N == G2.N
    assert np.all(G.dw == G2.dw)
    assert (G.W - G2.W).nnz == 0


def test_thresh_small():
    data = datasets.make_swiss_roll()[0]
    G = graphtools.Graph(data, thresh=1e-30)
    assert G.thresh == np.finfo("float").eps


def test_no_initialize():
    G = graphtools.Graph(data, thresh=1e-4, initialize=False)
    assert not hasattr(G, "_kernel")
    G.K
    assert hasattr(G, "_kernel")


def test_knn_graph_fixed_bandwidth():
    k = None
    decay = 5
    bandwidth = 10
    bandwidth_scale = 1.3
    n_pca = 20
    thresh = 1e-4
    pca = PCA(n_pca, svd_solver="randomized", random_state=42).fit(data)
    data_nu = pca.transform(data)
    pdx = squareform(pdist(data_nu, metric="euclidean"))
    K = np.exp(-1 * np.power(pdx / (bandwidth * bandwidth_scale), decay))
    K[K < thresh] = 0
    K = K + K.T
    W = np.divide(K, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(
        data,
        n_pca=n_pca,
        decay=decay,
        bandwidth=bandwidth,
        bandwidth_scale=bandwidth_scale,
        knn=k,
        random_state=42,
        thresh=thresh,
        search_multiplier=2,
        use_pygsp=True,
    )
    assert isinstance(G2, graphtools.graphs.kNNGraph)
    np.testing.assert_array_equal(G.N, G2.N)
    np.testing.assert_array_equal(G.d, G2.d)
    np.testing.assert_allclose(
        (G.W - G2.W).data, np.zeros_like((G.W - G2.W).data), atol=1e-14
    )
    bandwidth = np.random.gamma(20, 0.5, len(data))
    K = np.exp(-1 * (pdx.T / (bandwidth * bandwidth_scale)).T ** decay)
    K[K < thresh] = 0
    K = K + K.T
    W = np.divide(K, 2)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(
        data,
        n_pca=n_pca,
        decay=decay,
        bandwidth=bandwidth,
        bandwidth_scale=bandwidth_scale,
        knn=k,
        random_state=42,
        thresh=thresh,
        use_pygsp=True,
    )
    assert isinstance(G2, graphtools.graphs.kNNGraph)
    np.testing.assert_array_equal(G.N, G2.N)
    np.testing.assert_allclose(G.dw, G2.dw, atol=1e-14)
    np.testing.assert_allclose(
        (G.W - G2.W).data, np.zeros_like((G.W - G2.W).data), atol=1e-14
    )


def test_knn_graph_callable_bandwidth():
    with assert_raises_message(
        NotImplementedError,
        "Callable bandwidth is only supported by graphtools.graphs.TraditionalGraph.",
    ):
        k = 3
        decay = 5

        def bandwidth(x):
            return 2

        n_pca = 20
        thresh = 1e-4
        build_graph(
            data,
            n_pca=n_pca,
            knn=k - 1,
            decay=decay,
            bandwidth=bandwidth,
            random_state=42,
            thresh=thresh,
            graphtype="knn",
        )


def test_knn_graph_sparse_no_pca():
    with assert_warns_message(
        UserWarning, "cannot use tree with sparse input: using brute force"
    ):
        build_graph(
            sp.coo_matrix(data),
            n_pca=None,  # n_pca,
            decay=10,
            knn=3,
            thresh=1e-4,
            random_state=42,
            use_pygsp=True,
        )


#####################################################
# Check anisotropy
#####################################################


def test_knn_graph_anisotropy():
    k = 3
    a = 13
    n_pca = 20
    anisotropy = 0.9
    thresh = 1e-4
    data_small = data[np.random.choice(len(data), len(data) // 2, replace=False)]
    pca = PCA(n_pca, svd_solver="randomized", random_state=42).fit(data_small)
    data_small_nu = pca.transform(data_small)
    pdx = squareform(pdist(data_small_nu, metric="euclidean"))
    knn_dist = np.partition(pdx, k, axis=1)[:, :k]
    epsilon = np.max(knn_dist, axis=1)
    weighted_pdx = (pdx.T / epsilon).T
    K = np.exp(-1 * weighted_pdx ** a)
    K[K < thresh] = 0
    K = K + K.T
    K = np.divide(K, 2)
    d = K.sum(1)
    W = K / (np.outer(d, d) ** anisotropy)
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = build_graph(
        data_small,
        n_pca=n_pca,
        thresh=thresh,
        decay=a,
        knn=k - 1,
        random_state=42,
        use_pygsp=True,
        anisotropy=anisotropy,
    )
    assert isinstance(G2, graphtools.graphs.kNNGraph)
    assert G.N == G2.N
    np.testing.assert_allclose(G.dw, G2.dw, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose((G2.W - G.W).data, 0, atol=1e-14, rtol=1e-14)


#####################################################
# Check interpolation
#####################################################


def test_build_dense_knn_kernel_to_data():
    G = build_graph(data, decay=None)
    n = G.data.shape[0]
    K = G.build_kernel_to_data(data[: n // 2, :], knn=G.knn + 1)
    assert K.shape == (n // 2, n)
    K = G.build_kernel_to_data(G.data, knn=G.knn + 1)
    assert (G.kernel - (K + K.T) / 2).nnz == 0
    K = G.build_kernel_to_data(G.data_nu, knn=G.knn + 1)
    assert (G.kernel - (K + K.T) / 2).nnz == 0


def test_build_sparse_knn_kernel_to_data():
    G = build_graph(data, decay=None, sparse=True)
    n = G.data.shape[0]
    K = G.build_kernel_to_data(data[: n // 2, :], knn=G.knn + 1)
    assert K.shape == (n // 2, n)
    K = G.build_kernel_to_data(G.data, knn=G.knn + 1)
    assert (G.kernel - (K + K.T) / 2).nnz == 0
    K = G.build_kernel_to_data(G.data_nu, knn=G.knn + 1)
    assert (G.kernel - (K + K.T) / 2).nnz == 0


def test_knn_interpolate():
    G = build_graph(data, decay=None)
    with assert_raises_message(
        ValueError, "Either `transitions` or `Y` must be provided."
    ):
        G.interpolate(data)
    pca_data = PCA(2).fit_transform(data)
    transitions = G.extend_to_data(data)
    np.testing.assert_equal(
        G.interpolate(pca_data, Y=data),
        G.interpolate(pca_data, transitions=transitions),
    )


def test_knn_interpolate_wrong_shape():
    G = build_graph(data, n_pca=10, decay=None)
    with assert_raises_message(
        ValueError, "Expected a 2D matrix. Y has shape ({},)".format(data.shape[0])
    ):
        G.extend_to_data(data[:, 0])
    with assert_raises_message(
        ValueError,
        "Expected a 2D matrix. Y has shape ({}, {}, 1)".format(
            data.shape[0], data.shape[1]
        ),
    ):
        G.extend_to_data(data[:, :, None])
    with assert_raises_message(
        ValueError, "Y must be of shape either (n, 64) or (n, 10)"
    ):
        G.extend_to_data(data[:, : data.shape[1] // 2])
    G = build_graph(data, n_pca=None, decay=None)
    with assert_raises_message(ValueError, "Y must be of shape (n, 64)"):
        G.extend_to_data(data[:, : data.shape[1] // 2])


#################################################
# Check extra functionality
#################################################


def test_shortest_path_constant():
    data_small = data[np.random.choice(len(data), len(data) // 4, replace=False)]
    G = build_graph(data_small, knn=5, decay=None)
    P = graph_shortest_path(G.K)
    # sklearn returns 0 if no path exists
    P[np.where(P == 0)] = np.inf
    # diagonal should actually be zero
    np.fill_diagonal(P, 0)
    np.testing.assert_equal(P, G.shortest_path(distance="constant"))


def test_shortest_path_precomputed_constant():
    data_small = data[np.random.choice(len(data), len(data) // 4, replace=False)]
    G = build_graph(data_small, knn=5, decay=None)
    G = graphtools.Graph(G.K, precomputed="affinity")
    P = graph_shortest_path(G.K)
    # sklearn returns 0 if no path exists
    P[np.where(P == 0)] = np.inf
    # diagonal should actually be zero
    np.fill_diagonal(P, 0)
    np.testing.assert_equal(P, G.shortest_path(distance="constant"))
    np.testing.assert_equal(P, G.shortest_path())


def test_shortest_path_data():
    data_small = data[np.random.choice(len(data), len(data) // 4, replace=False)]
    G = build_graph(data_small, knn=5, decay=None)
    D = squareform(pdist(G.data_nu)) * np.where(G.K.toarray() > 0, 1, 0)
    P = graph_shortest_path(D)
    # sklearn returns 0 if no path exists
    P[np.where(P == 0)] = np.inf
    # diagonal should actually be zero
    np.fill_diagonal(P, 0)
    np.testing.assert_allclose(P, G.shortest_path(distance="data"))
    np.testing.assert_allclose(P, G.shortest_path())


def test_shortest_path_no_decay_affinity():
    with assert_raises_message(
        ValueError,
        "Graph shortest path with affinity distance only valid for weighted graphs. For unweighted graphs, use `distance='constant'` or `distance='data'`.",
    ):
        data_small = data[np.random.choice(len(data), len(data) // 4, replace=False)]
        G = build_graph(data_small, knn=5, decay=None)
        G.shortest_path(distance="affinity")


def test_shortest_path_precomputed_no_decay_affinity():
    with assert_raises_message(
        ValueError,
        "Graph shortest path with affinity distance only valid for weighted graphs. For unweighted graphs, use `distance='constant'` or `distance='data'`.",
    ):
        data_small = data[np.random.choice(len(data), len(data) // 4, replace=False)]
        G = build_graph(data_small, knn=5, decay=None)
        G = graphtools.Graph(G.K, precomputed="affinity")
        G.shortest_path(distance="affinity")


def test_shortest_path_precomputed_no_decay_data():
    with assert_raises_message(
        ValueError,
        "Graph shortest path with data distance not valid for precomputed graphs. For precomputed graphs, use `distance='constant'` for unweighted graphs and `distance='affinity'` for weighted graphs.",
    ):
        data_small = data[np.random.choice(len(data), len(data) // 4, replace=False)]
        G = build_graph(data_small, knn=5, decay=None)
        G = graphtools.Graph(G.K, precomputed="affinity")
        G.shortest_path(distance="data")


def test_shortest_path_invalid():
    with assert_raises_message(
        ValueError,
        "Expected `distance` in ['constant', 'data', 'affinity']. Got invalid",
    ):
        data_small = data[np.random.choice(len(data), len(data) // 4, replace=False)]
        G = build_graph(data_small, knn=5, decay=None)
        G.shortest_path(distance="invalid")


####################
# Test API
####################


def test_verbose():
    print()
    print("Verbose test: kNN")
    build_graph(data, decay=None, verbose=True)


def test_set_params():
    G = build_graph(data, decay=None)
    assert G.get_params() == {
        "n_pca": 20,
        "random_state": 42,
        "kernel_symm": "+",
        "theta": None,
        "anisotropy": 0,
        "knn": 3,
        "knn_max": None,
        "decay": None,
        "bandwidth": None,
        "bandwidth_scale": 1,
        "distance": "euclidean",
        "thresh": 0,
        "n_jobs": -1,
        "verbose": 0,
    }, G.get_params()
    G.set_params(n_jobs=4)
    assert G.n_jobs == 4
    assert G.knn_tree.n_jobs == 4
    G.set_params(random_state=13)
    assert G.random_state == 13
    G.set_params(verbose=2)
    assert G.verbose == 2
    G.set_params(verbose=0)
    with assert_raises_message(
        ValueError, "Cannot update knn. Please create a new graph"
    ):
        G.set_params(knn=15)
    with assert_raises_message(
        ValueError, "Cannot update knn_max. Please create a new graph"
    ):
        G.set_params(knn_max=15)
    with assert_raises_message(
        ValueError, "Cannot update decay. Please create a new graph"
    ):
        G.set_params(decay=10)
    with assert_raises_message(
        ValueError, "Cannot update distance. Please create a new graph"
    ):
        G.set_params(distance="manhattan")
    with assert_raises_message(
        ValueError, "Cannot update thresh. Please create a new graph"
    ):
        G.set_params(thresh=1e-3)
    with assert_raises_message(
        ValueError, "Cannot update theta. Please create a new graph"
    ):
        G.set_params(theta=0.99)
    with assert_raises_message(
        ValueError, "Cannot update kernel_symm. Please create a new graph"
    ):
        G.set_params(kernel_symm="*")
    with assert_raises_message(
        ValueError, "Cannot update anisotropy. Please create a new graph"
    ):
        G.set_params(anisotropy=0.7)
    with assert_raises_message(
        ValueError, "Cannot update bandwidth. Please create a new graph"
    ):
        G.set_params(bandwidth=5)
    with assert_raises_message(
        ValueError, "Cannot update bandwidth_scale. Please create a new graph"
    ):
        G.set_params(bandwidth_scale=5)
    G.set_params(
        knn=G.knn,
        decay=G.decay,
        thresh=G.thresh,
        distance=G.distance,
        theta=G.theta,
        anisotropy=G.anisotropy,
        kernel_symm=G.kernel_symm,
    )
