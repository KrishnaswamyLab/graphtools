import graphtools.matrix
import graphtools.utils
from parameterized import parameterized
from scipy import sparse
import numpy as np
import graphtools
from load_tests import data
from load_tests import assert_warns_message


@parameterized(
    [
        (np.array,),
        (sparse.csr_matrix,),
        (sparse.csc_matrix,),
        (sparse.bsr_matrix,),
        (sparse.lil_matrix,),
        (sparse.coo_matrix,),
    ]
)
def test_nonzero_discrete(matrix_class):
    X = np.random.choice([0, 1, 2], p=[0.95, 0.025, 0.025], size=(100, 100))
    X = matrix_class(X)
    assert graphtools.matrix.nonzero_discrete(X, [1, 2])
    assert not graphtools.matrix.nonzero_discrete(X, [1, 3])


@parameterized([(0,), (1e-4,)])
def test_nonzero_discrete_knngraph(thresh):
    G = graphtools.Graph(data, n_pca=10, knn=5, decay=None, thresh=thresh)
    assert graphtools.matrix.nonzero_discrete(G.K, [0.5, 1])


@parameterized([(0,), (1e-4,)])
def test_nonzero_discrete_decay_graph(thresh):
    G = graphtools.Graph(data, n_pca=10, knn=5, decay=15, thresh=thresh)
    assert not graphtools.matrix.nonzero_discrete(G.K, [0.5, 1])


def test_nonzero_discrete_constant():
    assert graphtools.matrix.nonzero_discrete(2, [1, 2])
    assert not graphtools.matrix.nonzero_discrete(2, [1, 3])


def test_if_sparse_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) if_sparse. (Use graphtools.matrix.if_sparse instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.if_sparse(lambda x: x, lambda x: x, np.zeros((4, 4)))


def test_sparse_minimum_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) sparse_minimum. (Use graphtools.matrix.sparse_minimum instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.sparse_minimum(
            sparse.csr_matrix((4, 4)), sparse.bsr_matrix((4, 4))
        )


def test_sparse_maximum_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) sparse_maximum. (Use graphtools.matrix.sparse_maximum instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.sparse_maximum(
            sparse.csr_matrix((4, 4)), sparse.bsr_matrix((4, 4))
        )


def test_elementwise_minimum_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) elementwise_minimum. (Use graphtools.matrix.elementwise_minimum instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.elementwise_minimum(
            sparse.csr_matrix((4, 4)), sparse.bsr_matrix((4, 4))
        )


def test_elementwise_maximum_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) elementwise_maximum. (Use graphtools.matrix.elementwise_maximum instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.elementwise_maximum(
            sparse.csr_matrix((4, 4)), sparse.bsr_matrix((4, 4))
        )


def test_dense_set_diagonal_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) dense_set_diagonal. (Use graphtools.matrix.dense_set_diagonal instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.dense_set_diagonal(np.zeros((4, 4)), 1)


def test_sparse_set_diagonal_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) sparse_set_diagonal. (Use graphtools.matrix.sparse_set_diagonal instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.sparse_set_diagonal(sparse.csr_matrix((4, 4)), 1)


def test_set_diagonal_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) set_diagonal. (Use graphtools.matrix.set_diagonal instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.set_diagonal(np.zeros((4, 4)), 1)


def test_set_submatrix_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) set_submatrix. (Use graphtools.matrix.set_submatrix instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.set_submatrix(
            sparse.lil_matrix((4, 4)), [1, 2], [0, 1], np.array([[1, 2], [3, 4]])
        )


def test_sparse_nonzero_discrete_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) sparse_nonzero_discrete. (Use graphtools.matrix.sparse_nonzero_discrete instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.sparse_nonzero_discrete(sparse.csr_matrix((4, 4)), [1])


def test_dense_nonzero_discrete_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) dense_nonzero_discrete. (Use graphtools.matrix.dense_nonzero_discrete instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.dense_nonzero_discrete(np.zeros((4, 4)), [1])


def test_nonzero_discrete_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) nonzero_discrete. (Use graphtools.matrix.nonzero_discrete instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.nonzero_discrete(np.zeros((4, 4)), [1])


def test_to_array_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) to_array. (Use graphtools.matrix.to_array instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.to_array([1])


def test_matrix_is_equivalent_deprecated():
    with assert_warns_message(
        DeprecationWarning,
        "Call to deprecated function (or staticmethod) matrix_is_equivalent. (Use graphtools.matrix.matrix_is_equivalent instead) -- Deprecated since version 1.5.0.",
    ):
        graphtools.utils.matrix_is_equivalent(
            sparse.csr_matrix((4, 4)), sparse.bsr_matrix((4, 4))
        )
