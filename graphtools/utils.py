import numpy as np
from scipy import sparse
from scipy.integrate import quad as definite_integral
import numbers


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


class MarcenkoPastur(object):
    ''' A collection of methods for computing statistics
        from the Marcenko-Pastur distribution
        Translated to Python by Jay Stanley from
        Code Supplement for
        "The Optimal Hard Threshold for Singular Values is 4/sqrt(3)" (2014)
        Matan Gavish, David L. Donoho
        https://arxiv.org/abs/1305.5870
    '''
    @staticmethod
    def median(beta):
        f = MarcenkoPastur._median_exact
        return f(beta)

    @staticmethod
    def _median_exact(beta):
        """
        Numerically evaluate the median of the Marcenko-Pastur distribution
        """
        def mar_pas(x): 1 - MarcenkoPastur._incremental(x, beta, 0)
        low_bound = (1 - np.sqrt(beta)) ** 2
        high_bound = (1 + np.sqrt(beta)) ** 2
        iterating = True

        while (iterating and (high_bound - low_bound) > 0.001):
            iterating = False
            x = np.linspace(low_bound, high_bound, num = 5)
            y = np.array([mar_pas(x_i) for x_i in x])
            low = y < 0.5
            high = y > 0.5
            if low.any():
                low_bound = np.max(x[low])
                iterating = True
            if high.any():
                high_bound = np.min(x[high])
                iterating = True
            median = (high_bound + low_bound)/2
        return median

    @staticmethod
    def _incremental(x0, beta, gamma):
        if beta > 1:
            raise ValueError("Beta must be less than 1 for the MP distribution")

        top_spec = (1 + np.sqrt(beta)) ** 2
        bottom_spec = (1 - np.sqrt(beta)) ** 2

        def mar_pas(x):
            x_shift = (top_spec - x) * (x - bottom_spec)
            Q = x_shift > 0
            y = np.sqrt(np.where(Q, x_shift, 0)) / (2 * np.pi * beta * x)
            return y

        if gamma != 0:
            def function(x): (x ** gamma * mar_pas(x))
        else:
            def function(x): mar_pas(x)

        y = definite_integral(function, 0, top_spec)[0]

        return y