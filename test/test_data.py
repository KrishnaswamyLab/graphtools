from __future__ import print_function
from load_tests import (
    np,
    sp,
    pd,
    graphtools,
    nose2,
    data,
    build_graph,
    squareform,
    pdist,
)
from load_tests import assert_raises_message, assert_warns_message
from nose.tools import assert_raises_regex

import numbers
import warnings

try:
    import anndata
except (ImportError, SyntaxError):
    # python2 support is missing
    with warnings.catch_warnings():
        warnings.filterwarnings("always")
        warnings.warn("Warning: failed to import anndata", ImportWarning)
    pass

#####################################################
# Check parameters
#####################################################


def test_1d_data():
    with assert_raises_message(
        ValueError,
        "Expected 2D array, got 1D array instead (shape: ({},).)".format(data.shape[0]),
    ):
        build_graph(data[:, 0])
    with assert_raises_message(
        ValueError,
        "Reshape your data either using array.reshape(-1, 1) "
        "if your data has a single feature or array.reshape(1, -1) if "
        "it contains a single sample.".format(data.shape[0]),
    ):
        build_graph(data[:, 0])


def test_3d_data():
    with assert_raises_message(
        ValueError,
        "Expected 2D array, got 3D array instead (shape: ({0}, 64, 1).)".format(
            data.shape[0]
        ),
    ):
        build_graph(data[:, :, None])


def test_0_n_pca():
    assert build_graph(data, n_pca=0).n_pca is None
    assert build_graph(data, n_pca=False).n_pca is None


def test_badstring_n_pca():
    with assert_raises_message(
        ValueError,
        "n_pca must be an integer 0 <= n_pca < min(n_samples,n_features), or in [None, False, True, 'auto'].",
    ):
        build_graph(data, n_pca="foobar")


def test_uncastable_n_pca():
    with assert_raises_message(
        ValueError,
        "n_pca was not an instance of numbers.Number, could not be cast to False, and not None. Please supply an integer 0 <= n_pca < min(n_samples,n_features) or None",
    ):
        build_graph(data, n_pca=[])


def test_negative_n_pca():
    with assert_raises_message(
        ValueError,
        "n_pca cannot be negative. Please supply an integer 0 <= n_pca < min(n_samples,n_features) or None",
    ):
        build_graph(data, n_pca=-1)


def test_badstring_rank_threshold():
    with assert_raises_message(
        ValueError, "rank_threshold must be positive float or 'auto'."
    ):
        build_graph(data, n_pca=True, rank_threshold="foobar")


def test_negative_rank_threshold():
    with assert_raises_message(
        ValueError, "rank_threshold must be positive float or 'auto'."
    ):
        build_graph(data, n_pca=True, rank_threshold=-1)


def test_True_n_pca_large_threshold():
    with assert_raises_regex(
        ValueError,
        r"Supplied threshold ([0-9\.]*) was greater than maximum singular value ([0-9\.]*) for the data matrix",
    ):
        build_graph(data, n_pca=True, rank_threshold=np.linalg.norm(data) ** 2)


def test_threshold_ignored():
    with assert_warns_message(
        RuntimeWarning,
        "n_pca = 10, therefore rank_threshold of -1 will not be used. To use rank thresholding, set n_pca = True",
    ):
        assert build_graph(data, n_pca=10, rank_threshold=-1).n_pca == 10


def test_invalid_threshold_negative():
    with assert_raises_message(
        ValueError, "rank_threshold must be positive float or 'auto'."
    ):
        build_graph(data, n_pca=True, rank_threshold=-1)


def test_invalid_threshold_list():
    with assert_raises_message(
        ValueError, "rank_threshold must be positive float or 'auto'."
    ):
        build_graph(data, n_pca=True, rank_threshold=[])


def test_True_n_pca():
    assert isinstance(build_graph(data, n_pca=True).n_pca, numbers.Number)


def test_True_n_pca_manual_rank_threshold():
    g = build_graph(data, n_pca=True, rank_threshold=0.1)
    assert isinstance(g.n_pca, numbers.Number)
    assert isinstance(g.rank_threshold, numbers.Number)


def test_True_n_pca_auto_rank_threshold():
    g = build_graph(data, n_pca=True, rank_threshold="auto")
    assert isinstance(g.n_pca, numbers.Number)
    assert isinstance(g.rank_threshold, numbers.Number)
    next_threshold = np.sort(g.data_pca.singular_values_)[2]
    g2 = build_graph(data, n_pca=True, rank_threshold=next_threshold)
    assert g.n_pca > g2.n_pca


def test_goodstring_rank_threshold():
    build_graph(data, n_pca=True, rank_threshold="auto")
    build_graph(data, n_pca=True, rank_threshold="AUTO")


def test_string_n_pca():
    build_graph(data, n_pca="auto")
    build_graph(data, n_pca="AUTO")


def test_fractional_n_pca():
    with assert_warns_message(
        RuntimeWarning, "Cannot perform PCA to fractional 1.5 dimensions. Rounding to 2"
    ):
        build_graph(data, n_pca=1.5)


def test_too_many_n_pca():
    with assert_warns_message(
        RuntimeWarning,
        "Cannot perform PCA to {0} dimensions on data with min(n_samples, n_features) = {0}".format(
            data.shape[1]
        ),
    ):
        build_graph(data, n_pca=data.shape[1])


def test_too_many_n_pca2():
    with assert_warns_message(
        RuntimeWarning,
        "Cannot perform PCA to {0} dimensions on data with min(n_samples, n_features) = {0}".format(
            data.shape[1] - 1
        ),
    ):
        build_graph(data[: data.shape[1] - 1], n_pca=data.shape[1] - 1)


def test_precomputed_with_pca():
    with assert_warns_message(
        RuntimeWarning,
        "n_pca cannot be given on a precomputed graph. Setting n_pca=None",
    ):
        build_graph(squareform(pdist(data)), precomputed="distance", n_pca=20)


#####################################################
# Check data types
#####################################################


def test_pandas_dataframe():
    G = build_graph(pd.DataFrame(data))
    assert isinstance(G, graphtools.base.BaseGraph)
    assert isinstance(G.data, np.ndarray)


def test_pandas_sparse_dataframe():
    try:
        X = pd.DataFrame(data).astype(pd.SparseDtype(float, fill_value=0))
    except AttributeError:
        X = pd.SparseDataFrame(data, default_fill_value=0)
    G = build_graph(X)
    assert isinstance(G, graphtools.base.BaseGraph)
    assert isinstance(G.data, sp.csr_matrix)


def test_anndata():
    try:
        anndata
    except NameError:
        # not installed
        return
    G = build_graph(anndata.AnnData(data))
    assert isinstance(G, graphtools.base.BaseGraph)
    assert isinstance(G.data, np.ndarray)


def test_anndata_sparse():
    try:
        anndata
    except NameError:
        # not installed
        return
    G = build_graph(anndata.AnnData(sp.csr_matrix(data)))
    assert isinstance(G, graphtools.base.BaseGraph)
    assert isinstance(G.data, sp.csr_matrix)


#####################################################
# Check transform
#####################################################


def test_transform_dense_pca():
    G = build_graph(data, n_pca=20)
    assert np.all(G.data_nu == G.transform(G.data))
    with assert_raises_message(
        ValueError,
        "data of shape ({0},) cannot be transformed to graph built on data of shape ({0}, {1}). Expected shape ({0}, {1})".format(
            G.data.shape[0], G.data.shape[1]
        ),
    ):
        G.transform(G.data[:, 0])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 1, 15) cannot be transformed to graph built on data of shape ({0}, {1}). Expected shape ({0}, {1})".format(
            G.data.shape[0], G.data.shape[1]
        ),
    ):
        G.transform(G.data[:, None, :15])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 15) cannot be transformed to graph built on data of shape ({0}, {1}). Expected shape ({0}, {1})".format(
            G.data.shape[0], G.data.shape[1]
        ),
    ):
        G.transform(G.data[:, :15])


def test_transform_dense_no_pca():
    G = build_graph(data, n_pca=None)
    assert np.all(G.data_nu == G.transform(G.data))
    with assert_raises_message(
        ValueError,
        "data of shape ({0},) cannot be transformed to graph built on data of shape ({0}, {1})".format(
            data.shape[0], data.shape[1]
        ),
    ):
        G.transform(G.data[:, 0])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 1, 15) cannot be transformed to graph built on data of shape ({0}, {1})".format(
            data.shape[0], data.shape[1]
        ),
    ):
        G.transform(G.data[:, None, :15])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 15) cannot be transformed to graph built on data of shape ({0}, {1})".format(
            data.shape[0], data.shape[1]
        ),
    ):
        G.transform(G.data[:, :15])


def test_transform_sparse_pca():
    G = build_graph(data, sparse=True, n_pca=20)
    assert np.all(G.data_nu == G.transform(G.data))
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 1) cannot be transformed to graph built on data of shape ({0}, {1}). Expected shape ({0}, {1})".format(
            G.data.shape[0], G.data.shape[1]
        ),
    ):
        G.transform(sp.csr_matrix(G.data)[:, 0])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 15) cannot be transformed to graph built on data of shape ({0}, {1}). Expected shape ({0}, {1})".format(
            G.data.shape[0], G.data.shape[1]
        ),
    ):
        G.transform(sp.csr_matrix(G.data)[:, :15])


def test_transform_sparse_no_pca():
    G = build_graph(data, sparse=True, n_pca=None)
    assert np.sum(G.data_nu != G.transform(G.data)) == 0
    with assert_raises_message(
        ValueError,
        "data of shape {} cannot be transformed to graph built on data of shape {}".format(
            G.data.tocsr()[:, 0].shape, G.data.shape
        ),
    ):
        G.transform(sp.csr_matrix(G.data)[:, 0])
    with assert_raises_message(
        ValueError,
        "data of shape {} cannot be transformed to graph built on data of shape {}".format(
            G.data.tocsr()[:, :15].shape, G.data.shape
        ),
    ):
        G.transform(sp.csr_matrix(G.data)[:, :15])


#####################################################
# Check inverse transform
#####################################################


def test_inverse_transform_dense_pca():
    G = build_graph(data, n_pca=data.shape[1] - 1)
    np.testing.assert_allclose(G.data, G.inverse_transform(G.data_nu), atol=1e-12)
    np.testing.assert_allclose(
        G.data[:, -1, None], G.inverse_transform(G.data_nu, columns=-1), atol=1e-12
    )
    np.testing.assert_allclose(
        G.data[:, 5:7], G.inverse_transform(G.data_nu, columns=[5, 6]), atol=1e-12
    )
    with assert_raises_message(
        IndexError,
        "index {0} is out of bounds for axis 1 with size {0}".format(G.data.shape[1]),
    ):
        G.inverse_transform(G.data_nu, columns=data.shape[1])
    with assert_raises_message(
        ValueError,
        "data of shape ({0},) cannot be inverse transformed from graph built on reduced data of shape ({0}, {1})".format(
            G.data.shape[0], G.n_pca
        ),
    ):
        G.inverse_transform(G.data[:, 0])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 1, 15) cannot be inverse transformed from graph built on reduced data of shape ({0}, {1})".format(
            G.data.shape[0], G.n_pca
        ),
    ):
        G.inverse_transform(G.data[:, None, :15])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 15) cannot be inverse transformed from graph built on reduced data of shape ({0}, {1})".format(
            G.data.shape[0], G.n_pca
        ),
    ):
        G.inverse_transform(G.data[:, :15])


def test_inverse_transform_sparse_svd():
    G = build_graph(data, sparse=True, n_pca=data.shape[1] - 1)
    np.testing.assert_allclose(data, G.inverse_transform(G.data_nu), atol=1e-12)
    np.testing.assert_allclose(
        data[:, -1, None], G.inverse_transform(G.data_nu, columns=-1), atol=1e-12
    )
    np.testing.assert_allclose(
        data[:, 5:7], G.inverse_transform(G.data_nu, columns=[5, 6]), atol=1e-12
    )
    with assert_raises_message(
        IndexError, "index 64 is out of bounds for axis 1 with size 64"
    ):
        G.inverse_transform(G.data_nu, columns=data.shape[1])
    with assert_raises_message(
        TypeError,
        "A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.",
    ):
        G.inverse_transform(sp.csr_matrix(G.data)[:, 0])
    with assert_raises_message(
        TypeError,
        "A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.",
    ):
        G.inverse_transform(sp.csr_matrix(G.data)[:, :15])
    with assert_raises_message(
        ValueError,
        "data of shape ({0},) cannot be inverse transformed from graph built on reduced data of shape ({0}, {1}). Expected shape ({0}, {1})".format(
            data.shape[0], G.n_pca
        ),
    ):
        G.inverse_transform(data[:, 0])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 15) cannot be inverse transformed from graph built on reduced data of shape ({0}, {1}). Expected shape ({0}, {1})".format(
            data.shape[0], G.n_pca
        ),
    ):
        G.inverse_transform(data[:, :15])


def test_inverse_transform_dense_no_pca():
    G = build_graph(data, n_pca=None)
    np.testing.assert_allclose(
        data[:, 5:7], G.inverse_transform(G.data_nu, columns=[5, 6]), atol=1e-12
    )
    assert np.all(G.data == G.inverse_transform(G.data_nu))
    with assert_raises_message(
        ValueError,
        "data of shape ({0},) cannot be inverse transformed from graph built on reduced data of shape ({0}, {1})".format(
            data.shape[0], G.data.shape[1]
        ),
    ):
        G.inverse_transform(G.data[:, 0])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 1, 15) cannot be inverse transformed from graph built on reduced data of shape ({0}, {1})".format(
            data.shape[0], data.shape[1]
        ),
    ):
        G.inverse_transform(G.data[:, None, :15])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 15) cannot be inverse transformed from graph built on reduced data of shape ({0}, {1})".format(
            data.shape[0], data.shape[1]
        ),
    ):
        G.inverse_transform(G.data[:, :15])


def test_inverse_transform_sparse_no_pca():
    G = build_graph(data, sparse=True, n_pca=None)
    assert np.sum(G.data != G.inverse_transform(G.data_nu)) == 0
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 1) cannot be inverse transformed from graph built on reduced data of shape ({0}, {1})".format(
            G.data.shape[0], G.data.shape[1]
        ),
    ):
        G.inverse_transform(sp.csr_matrix(G.data)[:, 0])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 15) cannot be inverse transformed from graph built on reduced data of shape ({0}, {1})".format(
            G.data.shape[0], G.data.shape[1]
        ),
    ):
        G.inverse_transform(sp.csr_matrix(G.data)[:, :15])


#####################################################
# Check adaptive PCA with rank thresholding
#####################################################


def test_transform_adaptive_pca():
    G = build_graph(data, n_pca=True, random_state=42)
    assert np.all(G.data_nu == G.transform(G.data))
    with assert_raises_message(
        ValueError,
        "data of shape ({0},) cannot be transformed to graph built on data of shape ({0}, {1}). Expected shape ({0}, {1})".format(
            G.data.shape[0], G.data.shape[1]
        ),
    ):
        G.transform(G.data[:, 0])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 1, 15) cannot be transformed to graph built on data of shape ({0}, {1}). Expected shape ({0}, {1})".format(
            G.data.shape[0], G.data.shape[1]
        ),
    ):
        G.transform(G.data[:, None, :15])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 15) cannot be transformed to graph built on data of shape ({0}, {1}). Expected shape ({0}, {1})".format(
            G.data.shape[0], G.data.shape[1]
        ),
    ):
        G.transform(G.data[:, :15])

    G2 = build_graph(data, n_pca=True, rank_threshold=G.rank_threshold, random_state=42)
    assert np.allclose(G2.data_nu, G2.transform(G2.data))
    assert np.allclose(G2.data_nu, G.transform(G.data))

    G3 = build_graph(data, n_pca=G2.n_pca, random_state=42)

    assert np.allclose(G3.data_nu, G3.transform(G3.data))
    assert np.allclose(G3.data_nu, G2.transform(G2.data))


def test_transform_sparse_adaptive_pca():
    G = build_graph(data, sparse=True, n_pca=True, random_state=42)
    assert np.all(G.data_nu == G.transform(G.data))
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 1) cannot be transformed to graph built on data of shape ({0}, {1}). Expected shape ({0}, {1})".format(
            G.data.shape[0], G.data.shape[1]
        ),
    ):
        G.transform(sp.csr_matrix(G.data)[:, 0])
    with assert_raises_message(
        ValueError,
        "data of shape ({0}, 15) cannot be transformed to graph built on data of shape ({0}, {1}). Expected shape ({0}, {1})".format(
            G.data.shape[0], G.data.shape[1]
        ),
    ):
        G.transform(sp.csr_matrix(G.data)[:, :15])

    G2 = build_graph(
        data, sparse=True, n_pca=True, rank_threshold=G.rank_threshold, random_state=42
    )
    assert np.allclose(G2.data_nu, G2.transform(G2.data))
    assert np.allclose(G2.data_nu, G.transform(G.data))

    G3 = build_graph(data, sparse=True, n_pca=G2.n_pca, random_state=42)
    assert np.allclose(G3.data_nu, G3.transform(G3.data))
    assert np.allclose(G3.data_nu, G2.transform(G2.data))


#############
# Test API
#############


def test_set_params():
    G = graphtools.base.Data(data, n_pca=20)
    assert G.get_params() == {"n_pca": 20, "random_state": None}
    G.set_params(random_state=13)
    assert G.random_state == 13
    with assert_raises_message(
        ValueError, "Cannot update n_pca. Please create a new graph"
    ):
        G.set_params(n_pca=10)
    G.set_params(n_pca=G.n_pca)
