import numpy as np
import numbers

from scipy import sparse


def if_sparse(sparse_func, dense_func, *args, **kwargs):
    if sparse.issparse(args[0]):
        for arg in args[1:]:
            assert sparse.issparse(arg)
        return sparse_func(*args, **kwargs)
    else:
        return dense_func(*args, **kwargs)


def sparse_minimum(X, Y):
    return X.minimum(Y)


def sparse_maximum(X, Y):
    return X.maximum(Y)


def elementwise_minimum(X, Y):
    return if_sparse(sparse_minimum, np.minimum, X, Y)


def elementwise_maximum(X, Y):
    return if_sparse(sparse_maximum, np.maximum, X, Y)


def dense_set_diagonal(X, diag):
    X[np.diag_indices(X.shape[0])] = diag
    return X


def sparse_set_diagonal(X, diag):
    cls = type(X)
    if not isinstance(X, (sparse.lil_matrix, sparse.dia_matrix)):
        X = X.tocoo()
    X.setdiag(diag)
    return cls(X)


def set_diagonal(X, diag):
    return if_sparse(sparse_set_diagonal, dense_set_diagonal, X, diag=diag)


def set_submatrix(X, i, j, values):
    X[np.ix_(i, j)] = values
    return X


def sparse_nonzero_discrete(X, values):
    if isinstance(
        X, (sparse.bsr_matrix, sparse.dia_matrix, sparse.dok_matrix, sparse.lil_matrix)
    ):
        X = X.tocsr()
    return dense_nonzero_discrete(X.data, values)


def dense_nonzero_discrete(X, values):
    result = np.full_like(X, False, dtype=bool)
    for value in values:
        result = np.logical_or(result, X == value)
    return np.all(result)


def nonzero_discrete(X, values):
    if isinstance(values, numbers.Number):
        values = [values]
    if 0 not in values:
        values.append(0)
    return if_sparse(sparse_nonzero_discrete, dense_nonzero_discrete, X, values=values)


def to_array(X):
    if sparse.issparse(X):
        X = X.toarray()
    elif isinstance(X, np.matrix):
        X = X.A
    return X


def matrix_is_equivalent(X, Y):
    """
    Checks matrix equivalence with numpy, scipy and pandas
    """
    return X is Y or (
        isinstance(X, Y.__class__)
        and X.shape == Y.shape
        and np.sum((X != Y).sum()) == 0
    )
