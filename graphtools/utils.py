import numpy as np
from scipy import sparse


def if_sparse(sparse_func, dense_func, *args, **kwargs):
    if sparse.issparse(args[0]):
        for arg in args[1:]:
            assert(sparse.issparse(arg))
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


def to_dense(X):
    if sparse.issparse(X):
        X = X.toarray()
    return X
