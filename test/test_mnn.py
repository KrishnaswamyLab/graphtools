from . import (
    graphtools,
    np,
    pd,
    pygsp,
    nose2,
    data,
    digits,
    build_graph,
    generate_swiss_roll,
    assert_raises,
    raises,
    cdist,
)


#####################################################
# Check parameters
#####################################################


@raises(ValueError)
def test_sample_idx_and_precomputed():
    build_graph(data, n_pca=None, sample_idx=np.arange(10),
                precomputed='distance')


@raises(ValueError)
def test_sample_idx_wrong_length():
    build_graph(data, graphtype='mnn', sample_idx=np.arange(10))


@raises(ValueError)
def test_sample_idx_unique():
    build_graph(data, graph_class=graphtools.graphs.MNNGraph,
                sample_idx=np.ones(len(data)))


@raises(ValueError)
def test_sample_idx_none():
    build_graph(data, graphtype='mnn', sample_idx=None)


@raises(ValueError)
def test_build_mnn_with_precomputed():
    build_graph(data, n_pca=None, graphtype='mnn', precomputed='distance')


#####################################################
# Check kernel
#####################################################


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
            k_ij = np.exp(-1 * (pdxe_ij ** a))  # apply alpha-decaying kernel
            if si == sj:
                K.iloc[sample_idx == si, sample_idx == sj] = k_ij * \
                    (1 - beta)  # fill out values in K for NN on diagonal
            else:
                # fill out values in K for NN on diagonal
                K.iloc[sample_idx == si, sample_idx == sj] = k_ij

    W = np.array((gamma * np.minimum(K, K.T)) +
                 ((1 - gamma) * np.maximum(K, K.T)))
    np.fill_diagonal(W, 0)
    G = pygsp.graphs.Graph(W)
    G2 = graphtools.Graph(X, knn=k + 1, decay=a, beta=1 - beta, gamma=gamma,
                          distance=metric, sample_idx=sample_idx, thresh=0,
                          use_pygsp=True)
    assert G.N == G2.N
    assert np.all(G.d == G2.d), "{} ({}, {})".format(
        np.where(G.d != G2.d),
        G.d[np.argwhere(G.d != G2.d).reshape(-1)],
        G.d2[np.argwhere(G.d != G2.d).reshape(-1)])
    assert (G.W != G2.W).nnz == 0
    assert (G2.W != G.W).sum() == 0
    assert isinstance(G2, graphtools.graphs.MNNGraph)


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
            k_ij = np.exp(-1 * (pdxe_ij ** a))  # apply alpha-decaying kernel
            if si == sj:
                K.iloc[sample_idx == si, sample_idx == sj] = k_ij * \
                    (1 - beta)  # fill out values in K for NN on diagonal
            else:
                # fill out values in K for NN on diagonal
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
                          distance=metric, sample_idx=sample_idx, thresh=0,
                          use_pygsp=True)
    assert G.N == G2.N
    assert np.all(G.d == G2.d), "{} ({}, {})".format(
        np.where(G.d != G2.d),
        G.d[np.argwhere(G.d != G2.d).reshape(-1)],
        G.d2[np.argwhere(G.d != G2.d).reshape(-1)])
    assert (G.W != G2.W).nnz == 0
    assert (G2.W != G.W).sum() == 0
    assert isinstance(G2, graphtools.graphs.MNNGraph)


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


#####################################################
# Check interpolation
#####################################################


# TODO: add interpolation tests

def test_verbose():
    X, sample_idx = generate_swiss_roll()
    build_graph(X, sample_idx=sample_idx, n_pca=None, verbose=True)


if __name__ == "__main__":
    exit(nose2.run())
