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
import numbers
import warnings
from load_tests import optimal_singular_value_thresholding_constants as osvth 
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


def test_0_n_pca():
    assert build_graph(data, n_pca=0).n_pca is None
    assert build_graph(data, n_pca=False).n_pca is None


@raises(ValueError)
def test_badstring_n_pca():
    build_graph(data, n_pca="foobar")


@raises(ValueError)
def test_uncastable_n_pca():
    build_graph(data, n_pca=[])


@raises(ValueError)
def test_negative_n_pca():
    build_graph(data, n_pca=-1)


@raises(ValueError)
def test_badstring_rank_threshold():
    build_graph(data, n_pca=True, rank_threshold="foobar")


@raises(ValueError)
def test_negative_rank_threshold():
    build_graph(data, n_pca=True, rank_threshold=-1)


@raises(ValueError)
@warns(RuntimeWarning)
def test_True_n_pca_large_threshold():
    build_graph(data, n_pca=True, rank_threshold=np.linalg.norm(data) ** 2)


@warns(RuntimeWarning)
def test_invalid_threshold1():
    assert build_graph(data, n_pca=10, rank_threshold=-1).n_pca == 10


@raises(ValueError)
def test_invalid_threshold2():
    build_graph(data, n_pca=True, rank_threshold=-1)


@raises(ValueError)
def test_invalid_threshold2():
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


@warns(RuntimeWarning)
def test_fractional_n_pca():
    build_graph(data, n_pca=1.5)


@warns(RuntimeWarning)
def test_too_many_n_pca():
    build_graph(data, n_pca=data.shape[1])


@warns(RuntimeWarning)
def test_too_many_n_pca2():
    build_graph(data[: data.shape[1] - 1], n_pca=data.shape[1] - 1)


@warns(RuntimeWarning)
def test_precomputed_with_pca():
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
    assert_raises(ValueError, G.transform, G.data[:, 0])
    assert_raises(ValueError, G.transform, G.data[:, None, :15])
    assert_raises(ValueError, G.transform, G.data[:, :15])


def test_transform_dense_no_pca():
    G = build_graph(data, n_pca=None)
    assert np.all(G.data_nu == G.transform(G.data))
    assert_raises(ValueError, G.transform, G.data[:, 0])
    assert_raises(ValueError, G.transform, G.data[:, None, :15])
    assert_raises(ValueError, G.transform, G.data[:, :15])


def test_transform_sparse_pca():
    G = build_graph(data, sparse=True, n_pca=20)
    assert np.all(G.data_nu == G.transform(G.data))
    assert_raises(ValueError, G.transform, sp.csr_matrix(G.data)[:, 0])
    assert_raises(ValueError, G.transform, sp.csr_matrix(G.data)[:, :15])


def test_transform_sparse_no_pca():
    G = build_graph(data, sparse=True, n_pca=None)
    assert np.sum(G.data_nu != G.transform(G.data)) == 0
    assert_raises(ValueError, G.transform, sp.csr_matrix(G.data)[:, 0])
    assert_raises(ValueError, G.transform, sp.csr_matrix(G.data)[:, :15])


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
    assert_raises(IndexError, G.inverse_transform, G.data_nu, columns=data.shape[1])
    assert_raises(ValueError, G.inverse_transform, G.data[:, 0])
    assert_raises(ValueError, G.inverse_transform, G.data[:, None, :15])
    assert_raises(ValueError, G.inverse_transform, G.data[:, :15])


def test_inverse_transform_sparse_svd():
    G = build_graph(data, sparse=True, n_pca=data.shape[1] - 1)
    np.testing.assert_allclose(data, G.inverse_transform(G.data_nu), atol=1e-12)
    np.testing.assert_allclose(
        data[:, -1, None], G.inverse_transform(G.data_nu, columns=-1), atol=1e-12
    )
    np.testing.assert_allclose(
        data[:, 5:7], G.inverse_transform(G.data_nu, columns=[5, 6]), atol=1e-12
    )
    assert_raises(IndexError, G.inverse_transform, G.data_nu, columns=data.shape[1])
    assert_raises(TypeError, G.inverse_transform, sp.csr_matrix(G.data)[:, 0])
    assert_raises(TypeError, G.inverse_transform, sp.csr_matrix(G.data)[:, :15])
    assert_raises(ValueError, G.inverse_transform, data[:, 0])
    assert_raises(ValueError, G.inverse_transform, data[:, :15])


def test_inverse_transform_dense_no_pca():
    G = build_graph(data, n_pca=None)
    np.testing.assert_allclose(
        data[:, 5:7], G.inverse_transform(G.data_nu, columns=[5, 6]), atol=1e-12
    )
    assert np.all(G.data == G.inverse_transform(G.data_nu))
    assert_raises(ValueError, G.inverse_transform, G.data[:, 0])
    assert_raises(ValueError, G.inverse_transform, G.data[:, None, :15])
    assert_raises(ValueError, G.inverse_transform, G.data[:, :15])


def test_inverse_transform_sparse_no_pca():
    G = build_graph(data, sparse=True, n_pca=None)
    assert np.sum(G.data != G.inverse_transform(G.data_nu)) == 0
    assert_raises(ValueError, G.inverse_transform, sp.csr_matrix(G.data)[:, 0])
    assert_raises(ValueError, G.inverse_transform, sp.csr_matrix(G.data)[:, :15])


#####################################################
# Check adaptive PCA with rank thresholding
#####################################################


def test_transform_adaptive_pca():
    G = build_graph(data, n_pca=True, random_state=42)
    assert np.all(G.data_nu == G.transform(G.data))
    assert_raises(ValueError, G.transform, G.data[:, 0])
    assert_raises(ValueError, G.transform, G.data[:, None, :15])
    assert_raises(ValueError, G.transform, G.data[:, :15])

    G2 = build_graph(data, n_pca=True, rank_threshold=G.rank_threshold, random_state=42)
    assert np.allclose(G2.data_nu, G2.transform(G2.data))
    assert np.allclose(G2.data_nu, G.transform(G.data))

    G3 = build_graph(data, n_pca=G2.n_pca, random_state=42,svd_iters=200)

    assert np.allclose(G3.data_nu, G3.transform(G3.data))
    assert np.allclose((G2.data_pca.components_.T*G2.data_pca.singular_values_)@G2.data_pca.components_,(G3.data_pca.components_.T*G3.data_pca.singular_values_)@G3.data_pca.components_)


def test_transform_sparse_adaptive_pca():
    G = build_graph(data, sparse=True, n_pca=True, random_state=42)
    assert np.all(G.data_nu == G.transform(G.data))
    assert_raises(ValueError, G.transform, sp.csr_matrix(G.data)[:, 0])
    assert_raises(ValueError, G.transform, sp.csr_matrix(G.data)[:, :15])

    G2 = build_graph(
        data, sparse=True, n_pca=True, rank_threshold=G.rank_threshold, random_state=42
    )
    assert np.allclose(G2.data_nu, G2.transform(G2.data))
    assert np.allclose(G2.data_nu, G.transform(G.data))

    G3 = build_graph(data, sparse=True, n_pca=G2.n_pca, random_state=42, svd_iters = 200)
    assert np.allclose(G3.data_nu, G3.transform(G3.data))
    #Note: there are some numerical stability issues with random svd that seem to be related to the number of components taken
    #and the powers used.  Note that G2 and G1 have their singular vectors computed by doing the full SVD whereas G3 only takes
    # a subspace.
    #I have opted to compare the LRAs themselves rather than the PCA implementation.
    #If you take a different number of components here I think you would be able to recover the previous tests of PC approx= PC
    #You can see this by computing the singular values.  They diverge for lower order singular vecs when you are just computing a few.
    assert np.allclose((G2.data_pca.components_.T*G2.data_pca.singular_values_)@G2.data_pca.components_,(G3.data_pca.components_.T*G3.data_pca.singular_values_)@G3.data_pca.components_)

def test_marcenkopastur_thresholding_exact():
    """ We can test Marcenko-Pastur threshold out to 4 decimal places by using the table defined in 
            "The Optimal Hard Threshold for Singular Values is 4/sqrt(3)" (2014)
            Matan Gavish, David L. Donoho 
            https://arxiv.org/abs/1305.5870" """

    #Our algorithm will compute more digits though!
    #To get any more testing accuracy we need a reference implementation (such as their matlab one)
    x = np.arange(0.05, 1.0, 0.05)
    x = np.round(x,2)

    for v in x:
        table_mp_omega = (osvth['omega'][v])
        assert np.isclose(graphtools.base.LowRankApproximation._exact_hard_threshold(v),table_mp_omega, 1e-4)

def test_marcenkopastur_thresholding_approximate():
    """ We can test Marcenko-Pastur threshold out to 2 decimal places by using the approximation defined in
            "The Optimal Hard Threshold for Singular Values is 4/sqrt(3)" (2014)
            Matan Gavish, David L. Donoho 
            https://arxiv.org/abs/1305.5870" """

    #Our algorithm will compute more digits though!
    #To get any more testing accuracy we need a reference implementation (such as their matlab one)
    x = np.linspace(0.001,1,100)

    for v in x:
        assert np.isclose(np.round(graphtools.base.LowRankApproximation._exact_hard_threshold(v),2),np.round(graphtools.base.LowRankApproximation._approximate_hard_threshold(v),2), 1e-2)

#############
# Test API
#############


def test_set_params():
    G = graphtools.base.Data(data, n_pca=20)
    assert G.get_params() == {"n_pca": 20, "random_state": None}
    G.set_params(random_state=13)
    assert G.random_state == 13
    assert_raises(ValueError, G.set_params, n_pca=10)
    G.set_params(n_pca=G.n_pca)
