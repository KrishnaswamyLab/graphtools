from future.utils import with_metaclass
from builtins import super
from copy import copy as shallow_copy
import numpy as np
import abc
import pygsp
from inspect import signature
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.utils.graph import graph_shortest_path
from scipy import sparse
import warnings
import numbers
import pickle
import sys
import tasklogger

from . import matrix, utils

_logger = tasklogger.get_tasklogger("graphtools")


class Base(object):
    """Class that deals with key-word arguments but is otherwise
    just an object.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        # Extract and sort argument names excluding 'self'
        parameters = set([p.name for p in parameters])

        # recurse
        for superclass in cls.__bases__:
            try:
                parameters.update(superclass._get_param_names())
            except AttributeError:
                # object and pygsp.graphs.Graph don't have this method
                pass

        return parameters

    def set_params(self, **kwargs):
        # for k in kwargs:
        #     raise TypeError("set_params() got an unexpected "
        #                     "keyword argument '{}'".format(k))
        return self


class Data(Base):
    """Parent class that handles the import and dimensionality reduction of data

    Parameters
    ----------
    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`.
        `pandas.DataFrame`, `pandas.SparseDataFrame`.

    n_pca : {`int`, `None`, `bool`, 'auto'}, optional (default: `None`)
        number of PC dimensions to retain for graph building.
        If n_pca in `[None, False, 0]`, uses the original data.
        If 'auto' or `True` then estimate using a singular value threshold
        Note: if data is sparse, uses SVD instead of PCA
        TODO: should we subtract and store the mean?

    rank_threshold : `float`, 'auto', optional (default: 'auto')
        threshold to use when estimating rank for
        `n_pca in [True, 'auto']`.
        If 'auto', this threshold is
        s_max * eps * max(n_samples, n_features)
        where s_max is the maximum singular value of the data matrix
        and eps is numerical precision. [press2007]_.

    random_state : `int` or `None`, optional (default: `None`)
        Random state for random PCA

    Attributes
    ----------
    data : array-like, shape=[n_samples,n_features]
        Original data matrix

    n_pca : int or `None`

    data_nu : array-like, shape=[n_samples,n_pca]
        Reduced data matrix

    data_pca : sklearn.decomposition.PCA or sklearn.decomposition.TruncatedSVD
        sklearn PCA operator
    """

    def __init__(
        self, data, n_pca=None, rank_threshold=None, random_state=None, **kwargs
    ):

        self._check_data(data)
        n_pca, rank_threshold = self._parse_n_pca_threshold(data, n_pca, rank_threshold)

        if utils.is_SparseDataFrame(data):
            data = data.to_coo()
        elif utils.is_DataFrame(data):
            try:
                # sparse data
                data = data.sparse.to_coo()
            except AttributeError:
                # dense data
                data = np.array(data)
        elif utils.is_Anndata(data):
            data = data.X

        self.data = data
        self.n_pca = n_pca
        self.rank_threshold = rank_threshold
        self.random_state = random_state
        self.data_nu = self._reduce_data()
        super().__init__(**kwargs)

    def _parse_n_pca_threshold(self, data, n_pca, rank_threshold):
        if isinstance(n_pca, str):
            n_pca = n_pca.lower()
            if n_pca != "auto":
                raise ValueError(
                    "n_pca must be an integer "
                    "0 <= n_pca < min(n_samples,n_features), "
                    "or in [None, False, True, 'auto']."
                )
        if isinstance(n_pca, numbers.Number):
            if not float(n_pca).is_integer():  # cast it to integer
                n_pcaR = np.round(n_pca).astype(int)
                warnings.warn(
                    "Cannot perform PCA to fractional {} dimensions. "
                    "Rounding to {}".format(n_pca, n_pcaR),
                    RuntimeWarning,
                )
                n_pca = n_pcaR

            if n_pca < 0:
                raise ValueError(
                    "n_pca cannot be negative. "
                    "Please supply an integer "
                    "0 <= n_pca < min(n_samples,n_features) or None"
                )
            elif np.min(data.shape) <= n_pca:
                warnings.warn(
                    "Cannot perform PCA to {} dimensions on "
                    "data with min(n_samples, n_features) = {}".format(
                        n_pca, np.min(data.shape)
                    ),
                    RuntimeWarning,
                )
                n_pca = 0

        if n_pca in [0, False, None]:  # cast 0, False to None.
            n_pca = None
        elif n_pca is True:  # notify that we're going to estimate rank.
            n_pca = "auto"
            _logger.info(
                "Estimating n_pca from matrix rank. "
                "Supply an integer n_pca "
                "for fixed amount."
            )
        if not any([isinstance(n_pca, numbers.Number), n_pca is None, n_pca == "auto"]):
            raise ValueError(
                "n_pca was not an instance of numbers.Number, "
                "could not be cast to False, and not None. "
                "Please supply an integer "
                "0 <= n_pca < min(n_samples,n_features) or None"
            )
        if rank_threshold is not None and n_pca != "auto":
            warnings.warn(
                "n_pca = {}, therefore rank_threshold of {} "
                "will not be used. To use rank thresholding, "
                "set n_pca = True".format(n_pca, rank_threshold),
                RuntimeWarning,
            )
        if n_pca == "auto":
            if isinstance(rank_threshold, str):
                rank_threshold = rank_threshold.lower()
            if rank_threshold is None:
                rank_threshold = "auto"
            if isinstance(rank_threshold, numbers.Number):
                if rank_threshold <= 0:
                    raise ValueError(
                        "rank_threshold must be positive float or 'auto'. "
                    )
            else:
                if rank_threshold != "auto":
                    raise ValueError(
                        "rank_threshold must be positive float or 'auto'. "
                    )
        return n_pca, rank_threshold

    def _check_data(self, data):
        if len(data.shape) != 2:
            msg = "Expected 2D array, got {}D array " "instead (shape: {}.) ".format(
                len(data.shape), data.shape
            )
            if len(data.shape) < 2:
                msg += (
                    "\nReshape your data either using array.reshape(-1, 1) "
                    "if your data has a single feature or array.reshape(1, -1) if "
                    "it contains a single sample."
                )
            raise ValueError(msg)

    def _reduce_data(self):
        """Private method to reduce data dimension.

        If data is dense, uses randomized PCA. If data is sparse, uses
        randomized SVD.
        TODO: should we subtract and store the mean?
        TODO: Fix the rank estimation so we do not compute the full SVD.

        Returns
        -------
        Reduced data matrix
        """
        if self.n_pca is not None and (
            self.n_pca == "auto" or self.n_pca < self.data.shape[1]
        ):
            with _logger.task("PCA"):
                n_pca = self.data.shape[1] - 1 if self.n_pca == "auto" else self.n_pca
                if sparse.issparse(self.data):
                    if (
                        isinstance(self.data, sparse.coo_matrix)
                        or isinstance(self.data, sparse.lil_matrix)
                        or isinstance(self.data, sparse.dok_matrix)
                    ):
                        self.data = self.data.tocsr()
                    self.data_pca = TruncatedSVD(n_pca, random_state=self.random_state)
                else:
                    self.data_pca = PCA(
                        n_pca, svd_solver="randomized", random_state=self.random_state
                    )
                self.data_pca.fit(self.data)
                if self.n_pca == "auto":
                    s = self.data_pca.singular_values_
                    smax = s.max()
                    if self.rank_threshold == "auto":
                        threshold = (
                            smax * np.finfo(self.data.dtype).eps * max(self.data.shape)
                        )
                        self.rank_threshold = threshold
                    threshold = self.rank_threshold
                    gate = np.where(s >= threshold)[0]
                    self.n_pca = gate.shape[0]
                    if self.n_pca == 0:
                        raise ValueError(
                            "Supplied threshold {} was greater than "
                            "maximum singular value {} "
                            "for the data matrix".format(threshold, smax)
                        )
                    _logger.info(
                        "Using rank estimate of {} as n_pca".format(self.n_pca)
                    )
                    # reset the sklearn operator
                    op = self.data_pca  # for line-width brevity..
                    op.components_ = op.components_[gate, :]
                    op.explained_variance_ = op.explained_variance_[gate]
                    op.explained_variance_ratio_ = op.explained_variance_ratio_[gate]
                    op.singular_values_ = op.singular_values_[gate]
                    self.data_pca = (
                        op  # im not clear if this is needed due to assignment rules
                    )
                data_nu = self.data_pca.transform(self.data)
            return data_nu
        else:
            data_nu = self.data
            if sparse.issparse(data_nu) and not isinstance(
                data_nu, (sparse.csr_matrix, sparse.csc_matrix, sparse.bsr_matrix)
            ):
                data_nu = data_nu.tocsr()
            return data_nu

    def get_params(self):
        """Get parameters from this object
        """
        return {"n_pca": self.n_pca, "random_state": self.random_state}

    def set_params(self, **params):
        """Set parameters on this object

        Safe setter method - attributes should not be modified directly as some
        changes are not valid.
        Valid parameters:
        - n_pca
        - random_state

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        if "n_pca" in params and params["n_pca"] != self.n_pca:
            raise ValueError("Cannot update n_pca. Please create a new graph")
        if "random_state" in params:
            self.random_state = params["random_state"]
        super().set_params(**params)
        return self

    def transform(self, Y):
        """Transform input data `Y` to reduced data space defined by `self.data`

        Takes data in the same ambient space as `self.data` and transforms it
        to be in the same reduced space as `self.data_nu`.

        Parameters
        ----------
        Y : array-like, shape=[n_samples_y, n_features]
            n_features must be the same as `self.data`.

        Returns
        -------
        Transformed data, shape=[n_samples_y, n_pca]

        Raises
        ------
        ValueError : if Y.shape[1] != self.data.shape[1]
        """
        try:
            # try PCA first
            return self.data_pca.transform(Y)
        except ValueError:
            # shape is wrong
            raise ValueError(
                "data of shape {0} cannot be transformed"
                " to graph built on data of shape {1}. "
                "Expected shape ({2}, {3})".format(
                    Y.shape, self.data.shape, Y.shape[0], self.data.shape[1]
                )
            )
        except AttributeError:  # no pca, try to return data
            if len(Y.shape) < 2 or Y.shape[1] != self.data.shape[1]:
                # shape is wrong
                raise ValueError(
                    "data of shape {0} cannot be transformed"
                    " to graph built on data of shape {1}. "
                    "Expected shape ({2}, {3})".format(
                        Y.shape, self.data.shape, Y.shape[0], self.data.shape[1]
                    )
                )
            else:
                return Y

    def inverse_transform(self, Y, columns=None):
        """Transform input data `Y` to ambient data space defined by `self.data`

        Takes data in the same reduced space as `self.data_nu` and transforms
        it to be in the same ambient space as `self.data`.

        Parameters
        ----------
        Y : array-like, shape=[n_samples_y, n_pca]
            n_features must be the same as `self.data_nu`.

        columns : list-like
            list of integers referring to column indices in the original data
            space to be returned. Avoids recomputing the full matrix where only
            a few dimensions of the ambient space are of interest

        Returns
        -------
        Inverse transformed data, shape=[n_samples_y, n_features]

        Raises
        ------
        ValueError : if Y.shape[1] != self.data_nu.shape[1]
        """
        try:
            if not hasattr(self, "data_pca"):
                # no pca performed
                try:
                    if Y.shape[1] != self.data_nu.shape[1]:
                        # shape is wrong
                        raise ValueError
                except IndexError:
                    # len(Y.shape) < 2
                    raise ValueError
                if columns is None:
                    return Y
                else:
                    columns = np.array([columns]).flatten()
                    return Y[:, columns]
            else:
                if columns is None:
                    return self.data_pca.inverse_transform(Y)
                else:
                    # only return specific columns
                    columns = np.array([columns]).flatten()
                    Y_inv = np.dot(Y, self.data_pca.components_[:, columns])
                    if hasattr(self.data_pca, "mean_"):
                        Y_inv += self.data_pca.mean_[columns]
                    return Y_inv
        except ValueError:
            # more informative error
            raise ValueError(
                "data of shape {0} cannot be inverse transformed"
                " from graph built on reduced data of shape ({1}, {2}). Expected shape ({3}, {2})".format(
                    Y.shape, self.data_nu.shape[0], self.data_nu.shape[1], Y.shape[0]
                )
            )


class BaseGraph(with_metaclass(abc.ABCMeta, Base)):
    """Parent graph class

    Parameters
    ----------

    kernel_symm : string, optional (default: '+')
        Defines method of kernel symmetrization.
        '+'  : additive
        '*'  : multiplicative
        'mnn' : min-max MNN symmetrization
        'none' : no symmetrization

    theta: float (default: 1)
        Min-max symmetrization constant.
        K = `theta * min(K, K.T) + (1 - theta) * max(K, K.T)`

    anisotropy : float, optional (default: 0)
        Level of anisotropy between 0 and 1
        (alpha in Coifman & Lafon, 2006)

    initialize : `bool`, optional (default : `True`)
        if false, don't create the kernel matrix.

    Attributes
    ----------
    K : array-like, shape=[n_samples, n_samples]
        kernel matrix defined as the adjacency matrix with
        ones down the diagonal

    kernel : synonym for `K`

    P : array-like, shape=[n_samples, n_samples] (cached)
        diffusion operator defined as a row-stochastic form
        of the kernel matrix

    diff_op : synonym for `P`
    """

    def __init__(
        self,
        kernel_symm="+",
        theta=None,
        anisotropy=0,
        gamma=None,
        initialize=True,
        **kwargs
    ):
        if gamma is not None:
            warnings.warn(
                "gamma is deprecated. " "Setting theta={}".format(gamma), FutureWarning
            )
            theta = gamma
        if kernel_symm == "gamma":
            warnings.warn(
                "kernel_symm='gamma' is deprecated. " "Setting kernel_symm='mnn'",
                FutureWarning,
            )
            kernel_symm = "mnn"
        if kernel_symm == "theta":
            warnings.warn(
                "kernel_symm='theta' is deprecated. " "Setting kernel_symm='mnn'",
                FutureWarning,
            )
            kernel_symm = "mnn"
        self.kernel_symm = kernel_symm
        self.theta = theta
        self._check_symmetrization(kernel_symm, theta)
        if not (isinstance(anisotropy, numbers.Real) and 0 <= anisotropy <= 1):
            raise ValueError(
                "Expected 0 <= anisotropy <= 1. " "Got {}".format(anisotropy)
            )
        self.anisotropy = anisotropy

        if initialize:
            _logger.debug("Initializing kernel...")
            self.K
        else:
            _logger.debug("Not initializing kernel.")
        super().__init__(**kwargs)

    def _check_symmetrization(self, kernel_symm, theta):
        if kernel_symm not in ["+", "*", "mnn", None]:
            raise ValueError(
                "kernel_symm '{}' not recognized. Choose from "
                "'+', '*', 'mnn', or 'none'.".format(kernel_symm)
            )
        elif kernel_symm != "mnn" and theta is not None:
            warnings.warn(
                "kernel_symm='{}' but theta is not None. "
                "Setting kernel_symm='mnn'.".format(kernel_symm)
            )
            self.kernel_symm = kernel_symm = "mnn"

        if kernel_symm == "mnn":
            if theta is None:
                self.theta = theta = 1
                warnings.warn(
                    "kernel_symm='mnn' but theta not given. "
                    "Defaulting to theta={}.".format(self.theta)
                )
            elif not isinstance(theta, numbers.Number) or theta < 0 or theta > 1:
                raise ValueError(
                    "theta {} not recognized. Expected "
                    "a float between 0 and 1".format(theta)
                )

    def _build_kernel(self):
        """Private method to build kernel matrix

        Runs public method to build kernel matrix and runs
        additional checks to ensure that the result is okay

        Returns
        -------
        Kernel matrix, shape=[n_samples, n_samples]

        Raises
        ------
        RuntimeWarning : if K is not symmetric
        """
        kernel = self.build_kernel()
        kernel = self.symmetrize_kernel(kernel)
        kernel = self.apply_anisotropy(kernel)
        if (kernel - kernel.T).max() > 1e-5:
            warnings.warn("K should be symmetric", RuntimeWarning)
        if np.any(kernel.diagonal() == 0):
            warnings.warn("K should have a non-zero diagonal", RuntimeWarning)
        return kernel

    def symmetrize_kernel(self, K):
        # symmetrize
        if self.kernel_symm == "+":
            _logger.debug("Using addition symmetrization.")
            K = (K + K.T) / 2
        elif self.kernel_symm == "*":
            _logger.debug("Using multiplication symmetrization.")
            K = K.multiply(K.T)
        elif self.kernel_symm == "mnn":
            _logger.debug("Using mnn symmetrization (theta = {}).".format(self.theta))
            K = self.theta * matrix.elementwise_minimum(K, K.T) + (
                1 - self.theta
            ) * matrix.elementwise_maximum(K, K.T)
        elif self.kernel_symm is None:
            _logger.debug("Using no symmetrization.")
            pass
        else:
            raise NotImplementedError
        return K

    def apply_anisotropy(self, K):
        if self.anisotropy == 0:
            # do nothing
            return K
        else:
            if sparse.issparse(K):
                d = np.array(K.sum(1)).flatten()
                K = K.tocoo()
                K.data = K.data / ((d[K.row] * d[K.col]) ** self.anisotropy)
                K = K.tocsr()
            else:
                d = K.sum(1)
                K = K / (np.outer(d, d) ** self.anisotropy)
        return K

    def get_params(self):
        """Get parameters from this object
        """
        return {
            "kernel_symm": self.kernel_symm,
            "theta": self.theta,
            "anisotropy": self.anisotropy,
        }

    def set_params(self, **params):
        """Set parameters on this object

        Safe setter method - attributes should not be modified directly as some
        changes are not valid.
        Valid parameters:
        Invalid parameters: (these would require modifying the kernel matrix)
        - kernel_symm
        - theta

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        if "theta" in params and params["theta"] != self.theta:
            raise ValueError("Cannot update theta. Please create a new graph")
        if "anisotropy" in params and params["anisotropy"] != self.anisotropy:
            raise ValueError("Cannot update anisotropy. Please create a new graph")
        if "kernel_symm" in params and params["kernel_symm"] != self.kernel_symm:
            raise ValueError("Cannot update kernel_symm. Please create a new graph")
        super().set_params(**params)
        return self

    @property
    def P(self):
        """Diffusion operator (cached)

        Return or calculate the diffusion operator

        Returns
        -------

        P : array-like, shape=[n_samples, n_samples]
            diffusion operator defined as a row-stochastic form
            of the kernel matrix
        """
        try:
            return self._diff_op
        except AttributeError:
            self._diff_op = normalize(self.kernel, "l1", axis=1)
            return self._diff_op

    @property
    def kernel_degree(self):
        """Weighted degree vector (cached)

        Return or calculate the degree vector from the affinity matrix

        Returns
        -------

        degrees : array-like, shape=[n_samples]
            Row sums of graph kernel
        """
        try:
            return self._kernel_degree
        except AttributeError:
            self._kernel_degree = matrix.to_array(self.kernel.sum(axis=1)).reshape(
                -1, 1
            )
            return self._kernel_degree

    @property
    def diff_aff(self):
        """Symmetric diffusion affinity matrix

        Return or calculate the symmetric diffusion affinity matrix

        .. math:: A(x,y) = K(x,y) (d(x) d(y))^{-1/2}

        where :math:`d` is the degrees (row sums of the kernel.)

        Returns
        -------

        diff_aff : array-like, shape=[n_samples, n_samples]
            symmetric diffusion affinity matrix defined as a
            doubly-stochastic form of the kernel matrix
        """
        row_degrees = self.kernel_degree
        if sparse.issparse(self.kernel):
            # diagonal matrix
            degrees = sparse.csr_matrix(
                (
                    1 / np.sqrt(row_degrees.flatten()),
                    np.arange(len(row_degrees)),
                    np.arange(len(row_degrees) + 1),
                )
            )
            return degrees @ self.kernel @ degrees
        else:
            col_degrees = row_degrees.T
            return (self.kernel / np.sqrt(row_degrees)) / np.sqrt(col_degrees)

    @property
    def diff_op(self):
        """Synonym for P
        """
        return self.P

    @property
    def K(self):
        """Kernel matrix

        Returns
        -------
        K : array-like, shape=[n_samples, n_samples]
            kernel matrix defined as the adjacency matrix with
            ones down the diagonal
        """
        try:
            return self._kernel
        except AttributeError:
            self._kernel = self._build_kernel()
            return self._kernel

    @property
    def kernel(self):
        """Synonym for K
        """
        return self.K

    @property
    def weighted(self):
        return self.decay is not None

    @abc.abstractmethod
    def build_kernel(self):
        """Build the kernel matrix

        Abstract method that all child classes must implement.
        Must return a symmetric matrix

        Returns
        -------
        K : kernel matrix, shape=[n_samples, n_samples]
            symmetric matrix with ones down the diagonal
            with no non-negative entries.
        """
        raise NotImplementedError

    def to_pygsp(self, **kwargs):
        """Convert to a PyGSP graph

        For use only when the user means to create the graph using
        the flag `use_pygsp=True`, and doesn't wish to recompute the kernel.
        Creates a graphtools.graphs.TraditionalGraph with a precomputed
        affinity matrix which also inherits from pygsp.graphs.Graph.

        Parameters
        ----------
        kwargs
            keyword arguments for graphtools.Graph

        Returns
        -------
        G : graphtools.base.PyGSPGraph, graphtools.graphs.TraditionalGraph
        """
        from . import api

        if "precomputed" in kwargs:
            if kwargs["precomputed"] != "affinity":
                warnings.warn(
                    "Cannot build PyGSPGraph with precomputed={}. "
                    "Using 'affinity' instead.".format(kwargs["precomputed"]),
                    UserWarning,
                )
            del kwargs["precomputed"]
        if "use_pygsp" in kwargs:
            if kwargs["use_pygsp"] is not True:
                warnings.warn(
                    "Cannot build PyGSPGraph with use_pygsp={}. "
                    "Use True instead.".format(kwargs["use_pygsp"]),
                    UserWarning,
                )
            del kwargs["use_pygsp"]
        return api.Graph(self.K, precomputed="affinity", use_pygsp=True, **kwargs)

    def to_igraph(self, attribute="weight", **kwargs):
        """Convert to an igraph Graph

        Uses the igraph.Graph constructor

        Parameters
        ----------
        attribute : str, optional (default: "weight")
        kwargs : additional arguments for igraph.Graph
        """
        try:
            import igraph as ig
        except ImportError:  # pragma: no cover
            raise ImportError(
                "Please install igraph with " "`pip install --user python-igraph`."
            )
        try:
            W = self.W
        except AttributeError:
            # not a pygsp graph
            W = self.K.copy()
            W = matrix.set_diagonal(W, 0)
        sources, targets = W.nonzero()
        edgelist = list(zip(sources, targets))
        g = ig.Graph(W.shape[0], edgelist, **kwargs)
        weights = W[W.nonzero()]
        weights = matrix.to_array(weights)
        g.es[attribute] = weights.flatten().tolist()
        return g

    def to_pickle(self, path):
        """Save the current Graph to a pickle.

        Parameters
        ----------
        path : str
            File path where the pickled object will be stored.
        """
        pickle_obj = shallow_copy(self)
        is_oldpygsp = all(
            [isinstance(self, pygsp.graphs.Graph), int(sys.version.split(".")[1]) < 7]
        )
        if is_oldpygsp:
            pickle_obj.logger = pickle_obj.logger.name
        with open(path, "wb") as f:
            pickle.dump(pickle_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _check_shortest_path_distance(self, distance):
        if distance == "data" and self.weighted:
            raise NotImplementedError(
                "Graph shortest path with constant or data distance only "
                "implemented for unweighted graphs. "
                "For weighted graphs, use `distance='affinity'`."
            )
        elif distance == "constant" and self.weighted:
            raise NotImplementedError(
                "Graph shortest path with constant distance only "
                "implemented for unweighted graphs. "
                "For weighted graphs, use `distance='affinity'`."
            )
        elif distance == "affinity" and not self.weighted:
            raise ValueError(
                "Graph shortest path with affinity distance only "
                "valid for weighted graphs. "
                "For unweighted graphs, use `distance='constant'` "
                "or `distance='data'`."
            )

    def _default_shortest_path_distance(self):
        if not self.weighted:
            distance = "data"
            _logger.info("Using ambient data distances.")
        else:
            distance = "affinity"
            _logger.info("Using negative log affinity distances.")
        return distance

    def shortest_path(self, method="auto", distance=None):
        """
        Find the length of the shortest path between every pair of vertices on the graph

        Parameters
        ----------
        method : string ['auto'|'FW'|'D']
            method to use.  Options are
            'auto' : attempt to choose the best method for the current problem
            'FW' : Floyd-Warshall algorithm.  O[N^3]
            'D' : Dijkstra's algorithm with Fibonacci stacks.  O[(k+log(N))N^2]
        distance : {'constant', 'data', 'affinity'}, optional (default: 'data')
            Distances along kNN edges.
            'constant' gives constant edge lengths.
            'data' gives distances in ambient data space.
            'affinity' gives distances as negative log affinities.
        Returns
        -------
        D : np.ndarray, float, shape = [N,N]
            D[i,j] gives the shortest distance from point i to point j
            along the graph. If no path exists, the distance is np.inf
        Notes
        -----
        Currently, shortest paths can only be calculated on kNNGraphs with
        `decay=None`
        """
        if distance is None:
            distance = self._default_shortest_path_distance()

        self._check_shortest_path_distance(distance)

        if distance == "constant":
            D = self.K
        elif distance == "data":
            D = sparse.coo_matrix(self.K)
            D.data = np.sqrt(
                np.sum((self.data_nu[D.row] - self.data_nu[D.col]) ** 2, axis=1)
            )
        elif distance == "affinity":
            D = sparse.csr_matrix(self.K)
            D.data = -1 * np.log(D.data)
        else:
            raise ValueError(
                "Expected `distance` in ['constant', 'data', 'affinity']. "
                "Got {}".format(distance)
            )

        P = graph_shortest_path(D, method=method)
        # symmetrize for numerical error
        P = (P + P.T) / 2
        # sklearn returns 0 if no path exists
        P[np.where(P == 0)] = np.inf
        # diagonal should actually be zero
        P[(np.arange(P.shape[0]), np.arange(P.shape[0]))] = 0
        return P


class PyGSPGraph(with_metaclass(abc.ABCMeta, pygsp.graphs.Graph, Base)):
    """Interface between BaseGraph and PyGSP.

    All graphs should possess these matrices. We inherit a lot
    of functionality from pygsp.graphs.Graph.

    There is a lot of overhead involved in having both a weight and
    kernel matrix
    """

    def __init__(self, lap_type="combinatorial", coords=None, plotting=None, **kwargs):
        if plotting is None:
            plotting = {}
        W = self._build_weight_from_kernel(self.K)

        super().__init__(
            W, lap_type=lap_type, coords=coords, plotting=plotting, **kwargs
        )

    @property
    @abc.abstractmethod
    def K():
        """Kernel matrix

        Returns
        -------
        K : array-like, shape=[n_samples, n_samples]
            kernel matrix defined as the adjacency matrix with
            ones down the diagonal
        """
        raise NotImplementedError

    def _build_weight_from_kernel(self, kernel):
        """Private method to build an adjacency matrix from
        a kernel matrix

        Just puts zeroes down the diagonal in-place, since the
        kernel matrix is ultimately not stored.

        Parameters
        ----------
        kernel : array-like, shape=[n_samples, n_samples]
            Kernel matrix.

        Returns
        -------
        Adjacency matrix, shape=[n_samples, n_samples]
        """

        weight = kernel.copy()
        self._diagonal = weight.diagonal().copy()
        weight = matrix.set_diagonal(weight, 0)
        return weight


class DataGraph(with_metaclass(abc.ABCMeta, Data, BaseGraph)):
    """Abstract class for graphs built from a dataset

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`.

    n_pca : {`int`, `None`, `bool`, 'auto'}, optional (default: `None`)
        number of PC dimensions to retain for graph building.
        If n_pca in `[None,False,0]`, uses the original data.
        If `True` then estimate using a singular value threshold
        Note: if data is sparse, uses SVD instead of PCA
        TODO: should we subtract and store the mean?

    rank_threshold : `float`, 'auto', optional (default: 'auto')
        threshold to use when estimating rank for
        `n_pca in [True, 'auto']`.
        Note that the default kwarg is `None` for this parameter.
        It is subsequently parsed to 'auto' if necessary.
        If 'auto', this threshold is
        smax * np.finfo(data.dtype).eps * max(data.shape)
        where smax is the maximum singular value of the data matrix.
        For reference, see, e.g.
        W. Press, S. Teukolsky, W. Vetterling and B. Flannery,
        “Numerical Recipes (3rd edition)”,
        Cambridge University Press, 2007, page 795.

    random_state : `int` or `None`, optional (default: `None`)
        Random state for random PCA and graph building

    verbose : `bool`, optional (default: `True`)
        Verbosity.

    n_jobs : `int`, optional (default : 1)
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used
    """

    def __init__(self, data, verbose=True, n_jobs=1, **kwargs):
        # kwargs are ignored
        self.n_jobs = n_jobs
        self.verbose = verbose
        _logger.set_level(verbose)
        super().__init__(data, **kwargs)

    def get_params(self):
        """Get parameters from this object
        """
        params = Data.get_params(self)
        params.update(BaseGraph.get_params(self))
        return params

    @abc.abstractmethod
    def build_kernel_to_data(self, Y):
        """Build a kernel from new input data `Y` to the `self.data`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_dimensions]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        Returns
        -------

        K_yx: array-like, [n_samples_y, n_samples]
            kernel matrix where each row represents affinities of a single
            sample in `Y` to all samples in `self.data`.

        Raises
        ------

        ValueError: if this Graph is not capable of extension or
        if the supplied data is the wrong shape
        """
        raise NotImplementedError

    def _check_extension_shape(self, Y):
        """Private method to check if new data matches `self.data`

        Parameters
        ----------
        Y : array-like, shape=[n_samples_y, n_features_y]
            Input data

        Returns
        -------
        Y : array-like, shape=[n_samples_y, n_pca]
            (Potentially transformed) input data

        Raises
        ------
        ValueError : if `n_features_y` is not either `self.data.shape[1]` or
        `self.n_pca`.
        """
        if len(Y.shape) != 2:
            raise ValueError("Expected a 2D matrix. Y has shape {}".format(Y.shape))
        if not Y.shape[1] == self.data_nu.shape[1]:
            # try PCA transform
            if Y.shape[1] == self.data.shape[1]:
                Y = self.transform(Y)
            else:
                # wrong shape
                if self.data.shape[1] != self.data_nu.shape[1]:
                    # PCA is possible
                    msg = ("Y must be of shape either " "(n, {}) or (n, {})").format(
                        self.data.shape[1], self.data_nu.shape[1]
                    )
                else:
                    # no PCA, only one choice of shape
                    msg = "Y must be of shape (n, {})".format(self.data.shape[1])
                raise ValueError(msg)
        return Y

    def extend_to_data(self, Y):
        """Build transition matrix from new data to the graph

        Creates a transition matrix such that `Y` can be approximated by
        a linear combination of samples in `self.data`. Any
        transformation of `self.data` can be trivially applied to `Y` by
        performing

        `transform_Y = self.interpolate(transform, transitions)`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_dimensions]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        Returns
        -------

        transitions : array-like, shape=[n_samples_y, self.data.shape[0]]
            Transition matrix from `Y` to `self.data`
        """
        Y = self._check_extension_shape(Y)
        kernel = self.build_kernel_to_data(Y)
        transitions = normalize(kernel, norm="l1", axis=1)
        return transitions

    def interpolate(self, transform, transitions=None, Y=None):
        """Interpolate new data onto a transformation of the graph data

        One of either transitions or Y should be provided

        Parameters
        ----------

        transform : array-like, shape=[n_samples, n_transform_features]

        transitions : array-like, optional, shape=[n_samples_y, n_samples]
            Transition matrix from `Y` (not provided) to `self.data`

        Y: array-like, optional, shape=[n_samples_y, n_dimensions]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        Returns
        -------

        Y_transform : array-like, [n_samples_y, n_features or n_pca]
            Transition matrix from `Y` to `self.data`

        Raises
        ------
        ValueError: if neither `transitions` nor `Y` is provided
        """
        if transitions is None:
            if Y is None:
                raise ValueError("Either `transitions` or `Y` must be provided.")
            else:
                transitions = self.extend_to_data(Y)
        Y_transform = transitions.dot(transform)
        return Y_transform

    def set_params(self, **params):
        """Set parameters on this object

        Safe setter method - attributes should not be modified directly as some
        changes are not valid.
        Valid parameters:
        - n_jobs
        - verbose

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        if "n_jobs" in params:
            self.n_jobs = params["n_jobs"]
        if "verbose" in params:
            self.verbose = params["verbose"]
            _logger.set_level(self.verbose)
        super().set_params(**params)
        return self
