from nose.tools import assert_raises_regex
from nose.tools import assert_warns_regex
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import graphtools
import nose2
import numpy as np
import pandas as pd
import pygsp
import re
import scipy.sparse as sp
import warnings


def assert_warns_message(expected_warning, expected_message, *args, **kwargs):
    expected_regex = re.escape(expected_message)
    return assert_warns_regex(expected_warning, expected_regex, *args, **kwargs)


def assert_raises_message(expected_warning, expected_message, *args, **kwargs):
    expected_regex = re.escape(expected_message)
    return assert_raises_regex(expected_warning, expected_regex, *args, **kwargs)


def reset_warnings():
    warnings.resetwarnings()
    warnings.simplefilter("error")
    ignore_numpy_warning()
    ignore_igraph_warning()
    ignore_joblib_warning()


def ignore_numpy_warning():
    warnings.filterwarnings(
        "ignore",
        category=PendingDeprecationWarning,
        message="the matrix subclass is not the recommended way to represent "
        "matrices or deal with linear algebra ",
    )


def ignore_igraph_warning():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="The SafeConfigParser class has been renamed to ConfigParser "
        "in Python 3.2. This alias will be removed in future versions. Use "
        "ConfigParser directly instead",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="Using or importing the ABCs from 'collections' instead of from "
        "'collections.abc' is deprecated since Python 3.3, and in 3.9 it will stop working",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="Using or importing the ABCs from 'collections' instead of from "
        "'collections.abc' is deprecated since Python 3.3,and in 3.9 it will stop working",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="Using or importing the ABCs from 'collections' instead of from "
        "'collections.abc' is deprecated, and in 3.8 it will stop working",
    )


def ignore_joblib_warning():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="check_pickle is deprecated in joblib 0.12 and will be removed"
        " in 0.13",
    )


reset_warnings()

global digits
global data
digits = datasets.load_digits()
data = digits["data"]


def generate_swiss_roll(n_samples=1000, noise=0.5, seed=42):
    generator = np.random.RandomState(seed)
    t = 1.5 * np.pi * (1 + 2 * generator.rand(1, n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    sample_idx = generator.choice([0, 1], n_samples, replace=True)
    z = sample_idx
    t = np.squeeze(t)
    X = np.concatenate((x, y))
    X += noise * generator.randn(2, n_samples)
    X = X.T[np.argsort(t)]
    X = np.hstack((X, z.reshape(n_samples, 1)))
    return X, sample_idx


def build_graph(
    data,
    n_pca=20,
    thresh=0,
    decay=10,
    knn=3,
    random_state=42,
    sparse=False,
    graph_class=graphtools.Graph,
    verbose=0,
    **kwargs,
):
    if sparse:
        data = sp.coo_matrix(data)
    return graph_class(
        data,
        thresh=thresh,
        n_pca=n_pca,
        decay=decay,
        knn=knn,
        random_state=42,
        verbose=verbose,
        **kwargs,
    )
