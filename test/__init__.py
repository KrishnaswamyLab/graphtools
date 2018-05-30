from sklearn.decomposition import PCA
from sklearn import datasets
from scipy.spatial.distance import pdist, cdist, squareform
import pygsp
import graphtools
import numpy as np
import scipy.sparse as sp
import warnings
import pandas as pd

import nose2
from nose.tools import raises, assert_raises, make_decorator
warnings.filterwarnings("error")

global digits
global data
digits = datasets.load_digits()
data = digits['data']


def generate_swiss_roll(n_samples=1000, noise=0.5, seed=42):
    generator = np.random.RandomState(seed)
    t = 1.5 * np.pi * (1 + 2 * generator.rand(1, n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    sample_idx = np.random.choice([0, 1], n_samples, replace=True)
    z = sample_idx
    t = np.squeeze(t)
    X = np.concatenate((x, y))
    X += noise * generator.randn(2, n_samples)
    X = X.T[np.argsort(t)]
    X = np.hstack((X, z.reshape(3000, 1)))
    return X, sample_idx


def build_graph(data, n_pca=20, thresh=0,
                decay=10, knn=3,
                random_state=42,
                sparse=False,
                graph_class=graphtools.Graph,
                **kwargs):
    if sparse:
        data = sp.coo_matrix(data)
    return graph_class(data, thresh=thresh, n_pca=n_pca,
                       decay=decay, knn=knn,
                       random_state=42, **kwargs)


def warns(*warns):
    """Test must raise one of expected warnings to pass.
    Example use::
      @warns(RuntimeWarning, UserWarning)
      def test_raises_type_error():
          warnings.warn("This test passes", RuntimeWarning)
      @warns(ImportWarning)
      def test_that_fails_by_passing():
          pass
    """
    valid = ' or '.join([w.__name__ for w in warns])

    def decorate(func):
        name = func.__name__

        def newfunc(*arg, **kw):
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                func(*arg, **kw)
                warnings.filterwarnings("error")
            try:
                for warn in w:
                    raise warn.category
            except warns:
                pass
            else:
                message = "%s() did not raise %s" % (name, valid)
                raise AssertionError(message)
        newfunc = make_decorator(func)(newfunc)
        return newfunc
    return decorate
