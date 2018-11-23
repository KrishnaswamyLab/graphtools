from __future__ import print_function
from load_tests import (
    np,
    sp,
    pd,
    graphtools,
    nose2,
    data,
    build_graph,
    assert_raises,
    raises,
    warns,
    squareform,
    pdist,
)
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


@raises(ValueError)
def test_1d_data():
    build_graph(data[:, 0])


@raises(ValueError)
def test_3d_data():
    build_graph(data[:, :, None])


@warns(RuntimeWarning)
def test_too_many_n_pca():
    build_graph(data, n_pca=data.shape[1])


@warns(RuntimeWarning)
def test_precomputed_with_pca():
    build_graph(squareform(pdist(data)),
                precomputed='distance',
                n_pca=20)


#####################################################
# Check data types
#####################################################


def test_pandas_dataframe():
    G = build_graph(pd.DataFrame(data))
    assert isinstance(G, graphtools.base.BaseGraph)
    assert isinstance(G.data, np.ndarray)


def test_pandas_sparse_dataframe():
    G = build_graph(pd.SparseDataFrame(data))
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
    assert(np.all(G.data_nu == G.transform(G.data)))
    assert_raises(ValueError, G.transform, G.data[:, 0])
    assert_raises(ValueError, G.transform, G.data[:, None, :15])
    assert_raises(ValueError, G.transform, G.data[:, :15])


def test_transform_dense_no_pca():
    G = build_graph(data, n_pca=None)
    assert(np.all(G.data_nu == G.transform(G.data)))
    assert_raises(ValueError, G.transform, G.data[:, 0])
    assert_raises(ValueError, G.transform, G.data[:, None, :15])
    assert_raises(ValueError, G.transform, G.data[:, :15])


def test_transform_sparse_pca():
    G = build_graph(data, sparse=True, n_pca=20)
    assert(np.all(G.data_nu == G.transform(G.data)))
    assert_raises(ValueError, G.transform, sp.csr_matrix(G.data)[:, 0])
    assert_raises(ValueError, G.transform, sp.csr_matrix(G.data)[:, :15])


def test_transform_sparse_no_pca():
    G = build_graph(data, sparse=True, n_pca=None)
    assert(np.sum(G.data_nu != G.transform(G.data)) == 0)
    assert_raises(ValueError, G.transform, sp.csr_matrix(G.data)[:, 0])
    assert_raises(ValueError, G.transform, sp.csr_matrix(G.data)[:, :15])


#####################################################
# Check inverse transform
#####################################################


def test_inverse_transform_dense_pca():
    G = build_graph(data, n_pca=data.shape[1] - 1)
    np.testing.assert_allclose(
        G.data, G.inverse_transform(G.data_nu), atol=1e-12)
    np.testing.assert_allclose(G.data[:, -1, None],
                               G.inverse_transform(G.data_nu, columns=-1),
                               atol=1e-12)
    np.testing.assert_allclose(G.data[:, 5:7],
                               G.inverse_transform(G.data_nu, columns=[5, 6]),
                               atol=1e-12)
    assert_raises(IndexError, G.inverse_transform,
                  G.data_nu, columns=data.shape[1])
    assert_raises(ValueError, G.inverse_transform, G.data[:, 0])
    assert_raises(ValueError, G.inverse_transform, G.data[:, None, :15])
    assert_raises(ValueError, G.inverse_transform, G.data[:, :15])


def test_inverse_transform_sparse_svd():
    G = build_graph(data, sparse=True, n_pca=data.shape[1] - 1)
    np.testing.assert_allclose(
        data, G.inverse_transform(G.data_nu), atol=1e-12)
    np.testing.assert_allclose(data[:, -1, None],
                               G.inverse_transform(G.data_nu, columns=-1),
                               atol=1e-12)
    np.testing.assert_allclose(data[:, 5:7],
                               G.inverse_transform(G.data_nu, columns=[5, 6]),
                               atol=1e-12)
    assert_raises(IndexError, G.inverse_transform,
                  G.data_nu, columns=data.shape[1])
    assert_raises(TypeError, G.inverse_transform, sp.csr_matrix(G.data)[:, 0])
    assert_raises(TypeError, G.inverse_transform,
                  sp.csr_matrix(G.data)[:, :15])
    assert_raises(ValueError, G.inverse_transform, data[:, 0])
    assert_raises(ValueError, G.inverse_transform,
                  data[:, :15])


def test_inverse_transform_dense_no_pca():
    G = build_graph(data, n_pca=None)
    np.testing.assert_allclose(data[:, 5:7],
                               G.inverse_transform(G.data_nu, columns=[5, 6]),
                               atol=1e-12)
    assert(np.all(G.data == G.inverse_transform(G.data_nu)))
    assert_raises(ValueError, G.inverse_transform, G.data[:, 0])
    assert_raises(ValueError, G.inverse_transform, G.data[:, None, :15])
    assert_raises(ValueError, G.inverse_transform, G.data[:, :15])


def test_inverse_transform_sparse_no_pca():
    G = build_graph(data, sparse=True, n_pca=None)
    assert(np.sum(G.data != G.inverse_transform(G.data_nu)) == 0)
    assert_raises(ValueError, G.inverse_transform, sp.csr_matrix(G.data)[:, 0])
    assert_raises(ValueError, G.inverse_transform,
                  sp.csr_matrix(G.data)[:, :15])


#############
# Test API
#############


def test_set_params():
    G = graphtools.base.Data(data, n_pca=20)
    assert G.get_params() == {'n_pca': 20, 'random_state': None}
    G.set_params(random_state=13)
    assert G.random_state == 13
    assert_raises(ValueError, G.set_params, n_pca=10)
    G.set_params(n_pca=G.n_pca)
