import numpy as np
import numbers
import warnings

from scipy import sparse
from functools import partial

try:
    import pandas as pd
except ImportError:
    # pandas not installed
    pass


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


def check_positive(**params):
    """Check that parameters are positive as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], numbers.Number) or params[p] <= 0:
            raise ValueError("Expected {} > 0, got {}".format(p, params[p]))


def check_int(**params):
    """Check that parameters are integers as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], numbers.Integral):
            raise ValueError("Expected {} integer, got {}".format(p, params[p]))


def check_if_not(x, *checks, **params):
    """Run checks only if parameters are not equal to a specified value

    Parameters
    ----------

    x : excepted value
        Checks not run if parameters equal x

    checks : function
        Unnamed arguments, check functions to be run

    params : object
        Named arguments, parameters to be checked

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] is not x and params[p] != x:
            [check(**{p: params[p]}) for check in checks]


def check_in(choices, **params):
    """Checks parameters are in a list of allowed parameters

    Parameters
    ----------

    choices : array-like, accepted values

    params : object
        Named arguments, parameters to be checked

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] not in choices:
            raise ValueError(
                "{} value {} not recognized. Choose from {}".format(
                    p, params[p], choices
                )
            )


def check_between(v_min, v_max, **params):
    """Checks parameters are in a specified range

    Parameters
    ----------

    v_min : float, minimum allowed value (inclusive)

    v_max : float, maximum allowed value (inclusive)

    params : object
        Named arguments, parameters to be checked

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] < v_min or params[p] > v_max:
            raise ValueError(
                "Expected {} between {} and {}, "
                "got {}".format(p, v_min, v_max, params[p])
            )


def attribute(attr, default=None, doc=None, on_set=None):
    def getter(self, attr):
        try:
            return getattr(self, "_" + attr)
        except AttributeError:
            return default

    def setter(self, value, attr, on_set=None):
        if on_set is not None:
            if callable(on_set):
                on_set = [on_set]
            for fn in on_set:
                fn(**{attr: value})
        setattr(self, "_" + attr, value)

    return property(
        fget=partial(getter, attr=attr),
        fset=partial(setter, attr=attr, on_set=on_set),
        doc=doc,
    )


def is_SparseDataFrame(X):
    try:
        pd
    except NameError:
        # pandas not installed
        return False
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version",
            FutureWarning,
        )
        try:
            return isinstance(X, pd.SparseDataFrame)
        except AttributeError:
            return False
