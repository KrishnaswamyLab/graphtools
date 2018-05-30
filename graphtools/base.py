from future.utils import with_metaclass
from builtins import super
import numpy as np
import abc
import pygsp
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from scipy import sparse
import numbers
import warnings

from .utils import (elementwise_minimum,
                    elementwise_maximum,
                    set_diagonal,
                    set_submatrix)
from .logging import (set_logging,
                      log_start,
                      log_complete,
                      log_warning,
                      log_debug)


class Base(object):
    """Class that deals with key-word arguments but is otherwise
    just an object.
    """

    def __init__(self, **kwargs):
        super().__init__()


class Data(Base):
    """Parent class that handles the import and dimensionality reduction of data

    Parameters
    ----------
    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`.
        TODO: accept pandas dataframes

    n_pca : `int` or `None`, optional (default: `None`)
        number of PC dimensions to retain for graph building.
        If `None`, uses the original data.
        Note: if data is sparse, uses SVD instead of PCA
        TODO: should we subtract and store the mean?

    random_state : `int` or `None`, optional (default: `None`)
        Random state for random PCA

    Attributes
    ----------
    data : array-like, shape=[n_samples,n_features]
        Original data matrix

    n_pca : int or `None`

    data_nu : array-like, shape=[n_samples,n_pca]
        Reduced data matrix

    U : array-like, shape=[n_samples, n_pca]
        Left singular vectors from PCA calculation

    S : array-like, shape=[n_pca]
        Singular values from PCA calculation

    V : array-like, shape=[n_features, n_pca]
        Right singular vectors from SVD calculation
    """

    def __init__(self, data, n_pca=None, random_state=None, **kwargs):

        if len(data.shape) != 2:
            raise ValueError("Expected a 2D matrix. data has shape {}".format(
                data.shape))
        if n_pca is not None and data.shape[1] <= n_pca:
            warnings.warn("Cannot perform PCA to {} dimensions on "
                          "data with {} dimensions".format(n_pca,
                                                           data.shape[1]),
                          RuntimeWarning)
            n_pca = None
        self.data = data
        self.n_pca = n_pca
        self.random_state = random_state

        self.data_nu = self._reduce_data()
        super().__init__(**kwargs)

    def _reduce_data(self):
        """Private method to reduce data dimension.

        If data is dense, uses randomized PCA. If data is sparse, uses
        randomized SVD.
        TODO: should we subtract and store the mean?

        Returns
        -------
        Reduced data matrix
        """
        if self.n_pca is not None and self.n_pca < self.data.shape[1]:
            log_start("PCA")
            if sparse.issparse(self.data):
                _, _, VT = randomized_svd(self.data, self.n_pca,
                                          random_state=self.random_state)
                V = VT.T
                self._right_singular_vectors = V
                data_nu = self.data.dot(V)
            else:
                self.pca = PCA(self.n_pca,
                               svd_solver='randomized',
                               random_state=self.random_state)
                self.pca.fit(self.data)
                data_nu = self.pca.transform(self.data)
            log_complete("PCA")
            return data_nu
        else:
            return self.data

    def get_params(self):
        """Get parameters from this object
        """
        return {'n_pca': self.n_pca,
                'random_state': self.random_state}

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
        if 'n_pca' in params and params['n_pca'] != self.n_pca:
            raise ValueError("Cannot update n_pca. Please create a new graph")
        if 'random_state' in params:
            self.random_state = params['random_state']
        return self

    @property
    def U(self):
        """Left singular vectors

        Returns
        -------
        Left singular vectors from PCA calculation, shape=[n_samples, n_pca]

        Raises
        ------
        AttributeError : PCA was not performed
        """
        try:
            return self.pca.components_
        except AttributeError:
            return None

    @property
    def S(self):
        """Singular values

        Returns
        -------
        Singular values from PCA calculation, shape=[n_pca]

        Raises
        ------
        AttributeError : PCA was not performed
        """
        try:
            return self.pca.singular_values_
        except AttributeError:
            return None

    @property
    def V(self):
        """Right singular vectors

        TODO: can we get this from PCA as well?

        Returns
        -------
        Right singular values from SVD calculation, shape=[n_features, n_pca]

        Raises
        ------
        AttributeError : SVD was not performed
        """
        try:
            return self._right_singular_vectors
        except AttributeError:
            return None

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
            return self.pca.transform(Y)
        except AttributeError:
            # no PCA - try SVD instead
            try:
                return Y.dot(self._right_singular_vectors)
            except AttributeError:
                # no SVD either - check if we can just return as is
                try:
                    if Y.shape[1] != self.data.shape[1]:
                        # shape is wrong
                        raise ValueError
                    return Y
                except IndexError:
                    # len(Y.shape) < 2
                    raise ValueError
        except ValueError:
            # more informative error
            raise ValueError("data of shape {} cannot be transformed"
                             " to graph built on data of shape {}".format(
                                 Y.shape, self.data.shape))

    def inverse_transform(self, Y):
        """Transform input data `Y` to ambient data space defined by `self.data`

        Takes data in the same reduced space as `self.data_nu` and transforms
        it to be in the same ambient space as `self.data`.

        Parameters
        ----------
        Y : array-like, shape=[n_samples_y, n_pca]
            n_features must be the same as `self.data_nu`.

        Returns
        -------
        Inverse transformed data, shape=[n_samples_y, n_features]

        Raises
        ------
        ValueError : if Y.shape[1] != self.data_nu.shape[1]
        """
        try:
            # try PCA first
            return self.pca.inverse_transform(Y)
        except AttributeError:
            # no PCA - try SVD instead
            try:
                return Y.dot(self._right_singular_vectors.T)
            except AttributeError:
                # no SVD either - check if we can just return as is
                try:
                    if Y.shape[1] != self.data_nu.shape[1]:
                        # shape is wrong
                        raise ValueError
                    return Y
                except IndexError:
                    # len(Y.shape) < 2
                    raise ValueError
        except ValueError:
            # more informative error
            raise ValueError("data of shape {} cannot be inverse transformed"
                             " from graph built on data of shape {}".format(
                                 Y.shape, self.data_nu.shape))


class BaseGraph(with_metaclass(abc.ABCMeta, Base)):
    """Parent graph class

    Parameters
    ----------

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

    def __init__(self, initialize=True, **kwargs):
        if initialize:
            log_debug("Initializing kernel...")
            self.K
        else:
            log_debug("Not initializing kernel.")
        super().__init__(**kwargs)

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
        if (kernel - kernel.T).max() > 1e-5:
            warnings.warn("K should be symmetric", RuntimeWarning)
        if np.any(kernel.diagonal == 0):
            warnings.warn("K should have a non-zero diagonal", RuntimeWarning)
        return kernel

    def get_params(self):
        """Get parameters from this object
        """
        return {}

    def set_params(self, **params):
        """Set parameters on this object

        Safe setter method - attributes should not be modified directly as some
        changes are not valid.
        Valid parameters:

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
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
            self._diff_op = normalize(self.kernel, 'l1', axis=1)
            return self._diff_op

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


class PyGSPGraph(with_metaclass(abc.ABCMeta, pygsp.graphs.Graph)):
    """Interface between BaseGraph and PyGSP.

    All graphs should possess these matrices. We inherit a lot
    of functionality from pygsp.graphs.Graph.

    There is a lot of overhead involved in having both a weight and
    kernel matrix
    """

    def __init__(self, **kwargs):
        W = self._build_weight_from_kernel(self.K)

        # delete non-pygsp keywords
        # TODO: is there a better way?
        keywords = [k for k in kwargs.keys()]
        for kw in keywords:
            if kw not in ['gtype', 'lap_type', 'coords', 'plotting']:
                del kwargs[kw]

        super().__init__(W=W, **kwargs)

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
        weight = set_diagonal(weight, 0)
        return weight


class DataGraph(with_metaclass(abc.ABCMeta, Data, BaseGraph)):
    """Abstract class for graphs built from a dataset

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`.
        TODO: accept pandas dataframes

    n_pca : `int` or `None`, optional (default: `None`)
        number of PC dimensions to retain for graph building.
        If `None`, uses the original data.
        Note: if data is sparse, uses SVD instead of PCA
        TODO: should we subtract and store the mean?

    random_state : `int` or `None`, optional (default: `None`)
        Random state for random PCA and graph building

    verbose : `bool`, optional (default: `True`)
        Verbosity.
        TODO: should this be an integer instead to allow multiple
        levels of verbosity?

    n_jobs : `int`, optional (default : 1)
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used
    """

    def __init__(self, data,
                 verbose=True,
                 n_jobs=1, **kwargs):
        # kwargs are ignored
        self.n_jobs = n_jobs
        self.verbose = verbose
        set_logging(verbose)
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
            raise ValueError("Expected a 2D matrix. Y has shape {}".format(
                Y.shape))
        if not Y.shape[1] == self.data_nu.shape[1]:
            # try PCA transform
            if Y.shape[1] == self.data.shape[1]:
                Y = self.transform(Y)
            else:
                # wrong shape
                if self.data.shape[1] != self.data_nu.shape[1]:
                    # PCA is possible
                    msg = ("Y must be of shape either "
                           "(n, {}) or (n, {})").format(
                        self.data.shape[1], self.data_nu.shape[1])
                else:
                    # no PCA, only one choice of shape
                    msg = "Y must be of shape (n, {})".format(
                        self.data.shape[1])
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
        transitions = normalize(kernel, norm='l1', axis=1)
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
                raise ValueError(
                    "Either `transitions` or `Y` must be provided.")
            else:
                transitions = self.extend_to_data(Y)
        Y_transform = transitions.dot(transform)
        return Y_transform
