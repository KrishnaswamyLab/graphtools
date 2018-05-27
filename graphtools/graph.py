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


class Data(object):
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

    def __init__(self, data, n_pca=None, random_state=None):

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

        self.data_nu = self._reduce_data()
        super().__init__()

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


class BaseGraph(with_metaclass(abc.ABCMeta, pygsp.graphs.Graph)):
    """Parent graph class

    All graphs should possess these matrices. We inherit a lot
    of functionality from pygsp.graphs.Graph.

    TODO: should we only optionally inherit from pygsp?
    There is a lot of overhead involved in having both a weight and
    kernel matrix

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

    def __init__(self, initialize=True, pygsp_kws=None, **kwargs):
        if initialize:
            kernel = self._build_kernel()
            W = self._build_weight_from_kernel(kernel)
        else:
            W = np.array([[0]])
        if pygsp_kws is None:
            pygsp_kws = {}
        super().__init__(W, **pygsp_kws)

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

        weight = kernel
        self._diagonal = weight.diagonal().copy()
        weight = set_diagonal(weight, 0)
        return weight

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

        TODO: set W as a lil / dok matrix to avoid sparsity changes

        Returns
        -------
        K : array-like, shape=[n_samples, n_samples]
            kernel matrix defined as the adjacency matrix with
            ones down the diagonal
        """
        kernel = self.W.copy()
        kernel = set_diagonal(kernel, self._diagonal)
        return kernel

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


class DataGraph(with_metaclass(abc.ABCMeta, BaseGraph, Data)):
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

    def __init__(self, data, n_pca=None, random_state=None,
                 verbose=True, n_jobs=1, **kwargs):
        # kwargs are ignored
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        set_logging(verbose)
        log_debug("set logging to debug")
        Data.__init__(self, data, n_pca=n_pca,
                      random_state=random_state)
        BaseGraph.__init__(self, **kwargs)

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


class kNNGraph(DataGraph):
    """
    K nearest neighbors graph

    TODO: kNNGraph with thresh=0 is just a TraditionalGraph. Should this
    be resolved?

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`.
        TODO: accept pandas dataframes

    knn : `int`, optional (default: 5)
        Number of nearest neighbors (including self) to use to build the graph

    decay : `int` or `None`, optional (default: `None`)
        Rate of alpha decay to use. If `None`, alpha decay is not used.

    distance : `str`, optional (default: `'euclidean'`)
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph.
        TODO: actually sklearn.neighbors has even more choices

    thresh : `float`, optional (default: `1e-5`)
        Threshold above which to calculate alpha decay kernel.
        All affinities below `thresh` will be set to zero in order to save
        on time and memory constraints.

    Attributes
    ----------

    knn_tree : `sklearn.neighbors.NearestNeighbors`
        The fitted KNN tree. (cached)
        TODO: can we be more clever than sklearn when it comes to choosing
        between KD tree, ball tree and brute force?
    """

    def __init__(self, data, knn=5, decay=None,
                 distance='euclidean',
                 thresh=1e-5, **kwargs):
        self.knn = knn
        self.decay = decay
        self.distance = distance
        self.thresh = thresh

        if decay is not None and thresh <= 0:
            raise ValueError("Cannot instantiate a kNNGraph with `decay=None` "
                             "and `thresh=0`. Use a TraditionalGraph instead.")

        super().__init__(data, **kwargs)

    def get_params(self):
        """Get parameters from this object
        """
        params = super().get_params()
        params.update({'knn': self.knn,
                       'decay': self.decay,
                       'distance': self.distance,
                       'thresh': self.thresh,
                       'n_jobs': self.n_jobs,
                       'random_state': self.random_state,
                       'verbose': self.verbose})
        return params

    def set_params(self, **params):
        """Set parameters on this object

        Safe setter method - attributes should not be modified directly as some
        changes are not valid.
        Valid parameters:
        - n_jobs
        - random_state
        - verbose
        Invalid parameters: (these would require modifying the kernel matrix)
        - knn
        - decay
        - distance
        - thresh

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        if 'knn' in params and params['knn'] != self.knn:
            raise ValueError("Cannot update knn. Please create a new graph")
        if 'decay' in params and params['decay'] != self.decay:
            raise ValueError("Cannot update decay. Please create a new graph")
        if 'distance' in params and params['distance'] != self.distance:
            raise ValueError("Cannot update distance. "
                             "Please create a new graph")
        if 'thresh' in params and params['thresh'] != self.thresh \
                and self.decay != 0:
            raise ValueError("Cannot update thresh. Please create a new graph")
        if 'n_jobs' in params:
            self.n_jobs = params['n_jobs']
        if 'random_state' in params:
            self.random_state = params['random_state']
        if 'verbose' in params:
            self.verbose = params['verbose']
        # update superclass parameters
        super().set_params(**params)
        return self

    @property
    def knn_tree(self):
        """KNN tree object (cached)

        Builds or returns the fitted KNN tree.
        TODO: can we be more clever than sklearn when it comes to choosing
        between KD tree, ball tree and brute force?

        Returns
        -------
        knn_tree : `sklearn.neighbors.NearestNeighbors`
        """
        try:
            return self._knn_tree
        except AttributeError:
            try:
                self._knn_tree = NearestNeighbors(
                    n_neighbors=self.knn,
                    algorithm='ball_tree',
                    metric=self.distance,
                    n_jobs=self.n_jobs).fit(self.data_nu)
            except ValueError:
                # invalid metric
                log_warning(
                    "Metric {} not valid for `sklearn.neighbors.BallTree`. "
                    "Graph instantiation may be slower than normal.")
                self._knn_tree = NearestNeighbors(
                    n_neighbors=self.knn,
                    algorithm='auto',
                    metric=self.distance,
                    n_jobs=self.n_jobs).fit(self.data_nu)
            return self._knn_tree

    def build_kernel(self):
        """Build the KNN kernel.

        Build a k nearest neighbors kernel, optionally with alpha decay.
        Must return a symmetric matrix

        Returns
        -------
        K : kernel matrix, shape=[n_samples, n_samples]
            symmetric matrix with ones down the diagonal
            with no non-negative entries.
        """
        if self.decay is None or self.thresh == 1:
            # binary connectivity matrix
            # sklearn has a function for this
            log_start("KNN search")
            K = kneighbors_graph(self.knn_tree,
                                 n_neighbors=self.knn,
                                 metric=self.distance,
                                 mode='connectivity',
                                 include_self=True)
            log_complete("KNN search")
        else:
            # sparse fast alpha decay
            K = self.build_kernel_to_data(self.data_nu)
        # symmetrize
        K = (K + K.T) / 2
        return K

    def build_kernel_to_data(self, Y, knn=None):
        """Build a kernel from new input data `Y` to the `self.data`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_dimensions]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        knn : `int` or `None`, optional (default: `None`)
            If `None`, defaults to `self.knn`

        Returns
        -------

        K_yx: array-like, [n_samples_y, n_samples]
            kernel matrix where each row represents affinities of a single
            sample in `Y` to all samples in `self.data`.

        Raises
        ------

        ValueError: if the supplied data is the wrong shape
        """
        if knn is None:
            knn = self.knn
        Y = self._check_extension_shape(Y)
        log_start("KNN search")
        if self.decay is None or self.thresh == 1:
            # binary connectivity matrix
            K = self.knn_tree.kneighbors_graph(
                Y, n_neighbors=knn,
                mode='connectivity')
            log_complete("KNN search")
        else:
            # sparse fast alpha decay
            knn_tree = self.knn_tree
            search_knn = min(knn * 20, len(self.data_nu))
            distances, indices = knn_tree.kneighbors(
                Y, n_neighbors=search_knn)
            log_complete("KNN search")
            log_start("affinities")
            bandwidth = distances[:, knn - 1]
            radius = bandwidth * np.power(-1 * np.log(self.thresh),
                                          1 / self.decay)
            update_idx = np.argwhere(
                np.max(distances, axis=1) < radius).reshape(-1)
            if len(update_idx) > 0 and search_knn < len(self.data_nu):
                distances = [d for d in distances]
                indices = [i for i in indices]
            while len(update_idx) > len(Y) // 10 and \
                    search_knn < len(self.data_nu) / 2:
                # increase the knn search
                search_knn = min(search_knn * 20, len(self.data_nu))
                dist_new, ind_new = knn_tree.kneighbors(
                    Y[update_idx], n_neighbors=search_knn)
                for i, idx in enumerate(update_idx):
                    distances[idx] = dist_new[i]
                    indices[idx] = ind_new[i]
                update_idx = [i for i, d in enumerate(distances)
                              if np.max(d) < radius[i]]
            if search_knn > len(self.data_nu) / 2:
                knn_tree = NearestNeighbors(knn, algorithm='brute',
                                            n_jobs=-1).fit(self.data_nu)
            if len(update_idx) > 0:
                # give up - radius search
                dist_new, ind_new = knn_tree.radius_neighbors(
                    Y[update_idx, :],
                    radius=np.max(radius[update_idx]))
                for i, idx in enumerate(update_idx):
                    distances[idx] = dist_new[i]
                    indices[idx] = ind_new[i]
            data = np.concatenate([distances[i] / bandwidth[i]
                                   for i in range(len(distances))])
            indices = np.concatenate(indices)
            indptr = np.concatenate(
                [[0], np.cumsum([len(d) for d in distances])])
            K = sparse.csr_matrix((data, indices, indptr))
            K.data = np.exp(-1 * np.power(K.data, self.decay))
            # TODO: should we zero values that are below thresh?
            K.data[K.data < self.thresh] = 0
            K = K.tocoo()
            K.eliminate_zeros()
            K = K.tocsr()
            log_complete("affinities")
        return K


class LandmarkGraph(DataGraph):
    """Landmark graph

    Adds landmarking feature to any data graph by taking spectral clusters
    and building a 'landmark operator' from clusters to samples and back to
    clusters.
    Any transformation on the landmark kernel is trivially extended to the
    data space by multiplying by the transition matrix.

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`.
        TODO: accept pandas dataframes

    n_landmark : `int`, optional (default: 2000)
        number of landmarks to use

    n_svd : `int`, optional (default: 100)
        number of SVD components to use for spectral clustering

    Attributes
    ----------
    landmark_op : array-like, shape=[n_landmark, n_landmark]
        Landmark operator.
        Can be treated as a diffusion operator between landmarks.

    transitions : array-like, shape=[n_samples, n_landmark]
        Transition probabilities between samples and landmarks.

    _clusters : array-like, shape=[n_samples]
        Private attribute. Cluster assignments for each sample.
    """

    def __init__(self, data, n_landmark=2000, n_svd=100, **kwargs):
        """Initialize a landmark graph.

        Raises
        ------
        RuntimeWarning : if too many SVD dimensions or
        too few landmarks are used
        """
        if n_landmark >= data.shape[0]:
            raise ValueError(
                "n_landmark ({}) >= n_samples ({}). Use "
                "kNNGraph instead".format(n_landmark, data.shape[0]))
        if n_svd >= data.shape[0]:
            warnings.warn("n_svd ({}) >= n_samples ({}) Consider "
                          "using kNNGraph or lower n_svd".format(
                              n_svd, data.shape[0]),
                          RuntimeWarning)
        self.n_landmark = n_landmark
        self.n_svd = n_svd
        super().__init__(data, **kwargs)

    def get_params(self):
        """Get parameters from this object
        """
        params = super().get_params()
        params.update({'n_landmark': self.n_landmark,
                       'n_pca': self.n_pca})
        return params

    def set_params(self, **params):
        """Set parameters on this object

        Safe setter method - attributes should not be modified directly as some
        changes are not valid.
        Valid parameters:
        - n_landmark
        - n_svd

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        # update parameters
        reset_landmarks = False
        if 'n_landmark' in params and params['n_landmark'] != self.n_landmark:
            self.n_landmark = params['n_landmark']
            reset_landmarks = True
        if 'n_svd' in params and params['n_svd'] != self.n_svd:
            self.n_svd = params['n_svd']
            reset_landmarks = True
        # update superclass parameters
        super().set_params(**params)
        # reset things that changed
        if reset_landmarks:
            self._reset_landmarks()
        return self

    def _reset_landmarks(self):
        """Reset landmark data

        Landmarks can be recomputed without recomputing the kernel
        """
        try:
            del self._landmark_op
            del self._transitions
            del self._clusters
        except AttributeError:
            # landmarks aren't currently defined
            pass

    @property
    def landmark_op(self):
        """Landmark operator

        Compute or return the landmark operator

        Returns
        -------
        landmark_op : array-like, shape=[n_landmark, n_landmark]
            Landmark operator. Can be treated as a diffusion operator between
            landmarks.
        """
        try:
            return self._landmark_op
        except AttributeError:
            self.build_landmark_op()
            return self._landmark_op

    @property
    def transitions(self):
        """Transition matrix from samples to landmarks

        Compute the landmark operator if necessary, then return the
        transition matrix.

        Returns
        -------
        transitions : array-like, shape=[n_samples, n_landmark]
            Transition probabilities between samples and landmarks.
        """
        try:
            return self._transitions
        except AttributeError:
            self.build_landmark_op()
            return self._transitions

    def build_landmark_op(self):
        """Build the landmark operator

        Calculates spectral clusters on the kernel, and calculates transition
        probabilities between cluster centers by using transition probabilities
        between samples assigned to each cluster.
        """
        is_sparse = sparse.issparse(self.kernel)
        # spectral clustering
        log_start("SVD")
        _, _, VT = randomized_svd(self.diff_op,
                                  n_components=self.n_svd,
                                  random_state=self.random_state)
        log_complete("SVD")
        log_start("KMeans")
        kmeans = MiniBatchKMeans(
            self.n_landmark,
            init_size=3 * self.n_landmark,
            batch_size=10000,
            random_state=self.random_state)
        self._clusters = kmeans.fit_predict(
            self.diff_op.dot(VT.T))
        # some clusters are not assigned
        landmarks = np.unique(self._clusters)
        log_complete("KMeans")

        # transition matrices
        if is_sparse:
            pmn = sparse.vstack(
                [sparse.csr_matrix(self.kernel[self._clusters == i, :].sum(
                    axis=0)) for i in landmarks])
        else:
            pmn = np.array([np.sum(self.kernel[self._clusters == i, :], axis=0)
                            for i in landmarks])
        # row normalize
        pnm = pmn.transpose()
        pmn = normalize(pmn, norm='l1', axis=1)
        pnm = normalize(pnm, norm='l1', axis=1)
        diff_op = pmn.dot(pnm)  # sparsity agnostic matrix multiplication
        if is_sparse:
            # no need to have a sparse landmark operator
            diff_op = diff_op.toarray()
        # store output
        self._landmark_op = diff_op
        self._transitions = pnm

    def extend_to_data(self, data, **kwargs):
        """Build transition matrix from new data to the graph

        Creates a transition matrix such that `Y` can be approximated by
        a linear combination of landmarks. Any
        transformation of the landmarks can be trivially applied to `Y` by
        performing

        `transform_Y = transitions.dot(transform)`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_dimensions]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        Returns
        -------

        transitions : array-like, [n_samples_y, self.data.shape[0]]
            Transition matrix from `Y` to `self.data`
        """
        kernel = self.build_kernel_to_data(data, **kwargs)
        if sparse.issparse(kernel):
            pnm = sparse.hstack(
                [sparse.csr_matrix(kernel[:, self._clusters == i].sum(
                    axis=1)) for i in np.unique(self._clusters)])
        else:
            pnm = np.array([np.sum(
                kernel[:, self._clusters == i],
                axis=1).T for i in np.unique(self._clusters)]).transpose()
        pnm = normalize(pnm, norm='l1', axis=1)
        return pnm

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
        """
        if transitions is None and Y is None:
            # assume Y is self.data and use standard landmark transitions
            transitions = self.transitions
        return super().interpolate(transform, transitions=transitions, Y=Y)


class TraditionalGraph(DataGraph):
    """Traditional weighted adjacency graph

    TODO: kNNGraph with thresh=0 is just a TraditionalGraph. Should this
    be resolved?

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`.
        If `precomputed` is not `None`, data should be an
        [n_samples, n_samples] matrix denoting pairwise distances,
        affinities, or edge weights.
        TODO: accept pandas dataframes

    knn : `int`, optional (default: 5)
        Number of nearest neighbors (including self) to use to build the graph

    decay : `int` or `None`, optional (default: `None`)
        Rate of alpha decay to use. If `None`, alpha decay is not used.

    distance : `str`, optional (default: `'euclidean'`)
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph.
        TODO: actually sklearn.neighbors has even more choices

    n_pca : `int` or `None`, optional (default: `None`)
        number of PC dimensions to retain for graph building.
        If `None`, uses the original data.
        Note: if data is sparse, uses SVD instead of PCA.
        Only one of `precomputed` and `n_pca` can be set.

    precomputed : {'distance', 'affinity', 'adjacency', `None`}, optional (default: `None`)
        If the graph is precomputed, this variable denotes which graph
        matrix is provided as `data`.
        Only one of `precomputed` and `n_pca` can be set.
    """

    def __init__(self, data, knn=5, decay=10,
                 distance='euclidean', n_pca=None,
                 precomputed=None, **kwargs):
        if precomputed is not None and n_pca is not None:
            # the data itself is a matrix of distances / affinities
            n_pca = None
            warnings.warn("n_pca cannot be given on a precomputed graph."
                          " Setting n_pca=None", RuntimeWarning)
        self.knn = knn
        self.decay = decay
        self.distance = distance
        self.precomputed = precomputed

        super().__init__(data, n_pca=n_pca,
                         **kwargs)

    def get_params(self):
        """Get parameters from this object
        """
        params = super().get_params()
        params.update({'knn': self.knn,
                       'decay': self.decay,
                       'distance': self.distance,
                       'precomputed': self.precomputed})
        return params

    def set_params(self, **params):
        """Set parameters on this object

        Safe setter method - attributes should not be modified directly as some
        changes are not valid.
        Invalid parameters: (these would require modifying the kernel matrix)
        - precomputed
        - distance
        - knn
        - decay

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        if 'precomputed' in params and \
                params['precomputed'] != self.precomputed:
            raise ValueError("Cannot update precomputed. "
                             "Please create a new graph")
        if 'distance' in params and params['distance'] != self.distance and \
                self.precomputed is not None:
            raise ValueError("Cannot update distance. "
                             "Please create a new graph")
        if 'knn' in params and params['knn'] != self.knn and \
                self.precomputed is not None:
            raise ValueError("Cannot update knn. Please create a new graph")
        if 'decay' in params and params['decay'] != self.decay and \
                self.precomputed is not None:
            raise ValueError("Cannot update decay. Please create a new graph")
        # update superclass parameters
        super().set_params(**params)
        return self

    def build_kernel(self):
        """Build the KNN kernel.

        Build a k nearest neighbors kernel, optionally with alpha decay.
        If `precomputed` is not `None`, the appropriate steps in the kernel
        building process are skipped.
        Must return a symmetric matrix

        Returns
        -------
        K : kernel matrix, shape=[n_samples, n_samples]
            symmetric matrix with ones down the diagonal
            with no non-negative entries.

        Raises
        ------
        ValueError: if `precomputed` is not an acceptable value
        """
        if self.precomputed is not None:
            if self.precomputed not in ["distance", "affinity", "adjacency"]:
                raise ValueError("Precomputed value {} not recognized. "
                                 "Choose from ['distance', 'affinity', "
                                 "'adjacency']")
            elif self.data_nu.shape[0] != self.data_nu.shape[1]:
                raise ValueError("Precomputed {} must be a square matrix. "
                                 "{} was given".format(self.precomputed,
                                                       self.data_nu.shape))
        if self.precomputed is "affinity":
            # already done
            # TODO: should we check that precomputed matrices look okay?
            # e.g. check the diagonal
            K = self.data_nu
        elif self.precomputed is "adjacency":
            # need to set diagonal to one to make it an affinity matrix
            K = self.data_nu
            K = set_diagonal(K, 1)
        else:
            log_start("affinities")
            if sparse.issparse(self.data_nu):
                self.data_nu = self.data_nu.toarray()
            if self.precomputed is "distance":
                pdx = self.data_nu
            elif self.precomputed is None:
                pdx = squareform(pdist(self.data_nu, metric=self.distance))
            knn_dist = np.partition(pdx, self.knn, axis=1)[:, :self.knn]
            epsilon = np.max(knn_dist, axis=1)
            pdx = (pdx.T / epsilon).T
            K = np.exp(-1 * np.power(pdx, self.decay))
            log_complete("affinities")
        # symmetrize
        K = (K + K.T) / 2
        return K

    def build_kernel_to_data(self, Y, knn=None):
        """Build transition matrix from new data to the graph

        Creates a transition matrix such that `Y` can be approximated by
        a linear combination of landmarks. Any
        transformation of the landmarks can be trivially applied to `Y` by
        performing

        `transform_Y = transitions.dot(transform)`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_dimensions]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        Returns
        -------

        transitions : array-like, [n_samples_y, self.data.shape[0]]
            Transition matrix from `Y` to `self.data`

        Raises
        ------

        ValueError: if `precomputed` is not `None`, then the graph cannot
        be extended.
        """
        if knn is None:
            knn = self.knn
        if self.precomputed is not None:
            raise ValueError("Cannot extend kernel on precomputed graph")
        else:
            Y = self._check_extension_shape(Y)
            pdx = cdist(Y, self.data_nu, metric=self.distance)
            knn_dist = np.partition(pdx, knn, axis=1)[:, :knn]
            epsilon = np.max(knn_dist, axis=1)
            pdx = (pdx.T / epsilon).T
            K = np.exp(-1 * pdx**self.decay)
        return K


class MNNGraph(DataGraph):
    """Mutual nearest neighbors graph

    Performs batch correction by forcing connections between batches, but
    only when the connection is mutual (i.e. x is a neighbor of y _and_
    y is a neighbor of x).

    Parameters
    ----------
    sample_idx: array-like, shape=[n_samples]
        Batch index

    beta: `float`, optional (default: 1)
        Downweight within-batch affinities by beta

    gamma: `float` or {'+', '*'} (default: 0.99)
        Symmetrization method.
        If '+', use `(K + K.T) / 2`;
        if '*', use `K * K.T`;
        if a float, use
        `gamma * min(K, K.T) + (1 - gamma) * max(K, K.T)`

    adaptive_k : `{'min', 'mean', 'sqrt', 'none'}` (default: 'sqrt')
        Weights MNN kernel adaptively using the number of cells in
        each sample according to the selected method.

    Attributes
    ----------
    subgraphs : list of `kNNGraph`s
        Graphs representing each batch separately
    """

    def __init__(self, data, sample_idx,
                 knn=5, beta=1, gamma=0.99, n_pca=None,
                 adaptive_k='sqrt',
                 **kwargs):
        self.beta = beta
        self.gamma = gamma
        self.sample_idx = sample_idx
        self.samples, self.n_cells = np.unique(
            self.sample_idx, return_counts=True)
        self.adaptive_k = adaptive_k
        self.knn = knn
        self.weighted_knn = self._weight_knn()

        self.knn_args = kwargs

        if sample_idx is None:
            raise ValueError("sample_idx must be given. For a graph without"
                             " batch correction, use kNNGraph.")
        elif len(sample_idx) != len(data):
            raise ValueError("sample_idx ({}) must be the same length as "
                             "data ({})".format(len(sample_idx), len(data)))
        elif len(self.samples) == 1:
            raise ValueError(
                "sample_idx must contain more than one unique value")

        if isinstance(gamma, str):
            if gamma not in ['+', '*']:
                raise ValueError(
                    "gamma '{}' not recognized. Choose from "
                    "'+', '*', a float between 0 and 1, "
                    "or a matrix of floats between 0 "
                    "and 1.".format(gamma))
        elif isinstance(gamma, numbers.Number):
            if (gamma < 0 or gamma > 1):
                raise ValueError(
                    "gamma '{}' invalid. Choose from "
                    "'+', '*', a float between 0 and 1, "
                    "or a matrix of floats between 0 "
                    "and 1.".format(gamma))
        else:
            # matrix
            if not np.shape(self.gamma) == (len(self.samples),
                                            len(self.samples)):
                raise ValueError(
                    "Matrix gamma must be of shape "
                    "({}), got ({})".format(
                        (len(self.samples),
                         len(self.samples)), gamma.shape))
            elif np.max(gamma) > 1 or np.min(gamma) < 0:
                raise ValueError(
                    "Values in matrix gamma must be between"
                    " 0 and 1, got values between {} and {}".format(
                        np.max(gamma), np.min(gamma)))
            elif np.any(gamma != gamma.T):
                raise ValueError("gamma must be a symmetric matrix")

        super().__init__(data, n_pca=n_pca, **kwargs)

    def _weight_knn(self, sample_size=None):
        """Select adaptive values of knn

        Parameters
        ----------

        sample_size : `int` or `None`
            Number of cells in the sample in question. Used only for
            out-of-sample extension. If `None`, calculates within-sample
            knn values.

        Returns
        -------

        knn : array-like or `int`, weighted knn values
        """
        if sample_size is None:
            # calculate within sample knn values
            sample_size = self.n_cells
        if self.adaptive_k == 'min':
            # the smallest sample has k
            knn_weight = self.n_cells / np.min(self.n_cells)
        elif self.adaptive_k == 'mean':
            # the average sample has k
            knn_weight = self.n_cells / np.mean(self.n_cells)
        elif self.adaptive_k == 'sqrt':
            # the samples are sqrt'd first, then smallest has k
            knn_weight = np.sqrt(self.n_cells / np.min(self.n_cells))
        elif self.adaptive_k == 'none':
            knn_weight = np.repeat(1, len(self.n_cells))
        weighted_knn = np.round(self.knn * knn_weight).astype(np.int32)
        if len(weighted_knn) == 1:
            weighted_knn = weighted_knn[0]
        return weighted_knn

    def get_params(self):
        """Get parameters from this object
        """
        params = super().get_params()
        params.update({'beta': self.beta,
                       'gamma': self.gamma})
        params.update(self.knn_args)
        return params

    def set_params(self, **params):
        """Set parameters on this object

        Safe setter method - attributes should not be modified directly as some
        changes are not valid.
        Valid parameters:
        - n_jobs
        - random_state
        - verbose
        Invalid parameters: (these would require modifying the kernel matrix)
        - knn
        - adaptive_k
        - decay
        - distance
        - thresh
        - beta
        - gamma

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        # mnn specific arguments
        if 'beta' in params and params['beta'] != self.beta:
            raise ValueError("Cannot update beta. Please create a new graph")
        if 'gamma' in params and params['gamma'] != self.gamma:
            raise ValueError("Cannot update gamma. Please create a new graph")
        if 'adaptive_k' in params and params['adaptive_k'] != self.adaptive_k:
            raise ValueError(
                "Cannot update adaptive_k. Please create a new graph")

        # knn arguments
        knn_kernel_args = ['knn', 'decay', 'distance', 'thresh']
        knn_other_args = ['n_jobs', 'random_state', 'verbose']
        for arg in knn_kernel_args:
            if arg in params and (arg not in self.knn_args or
                                  params[arg] != self.knn_args[arg]):
                raise ValueError("Cannot update {}. "
                                 "Please create a new graph".format(arg))
        for arg in knn_other_args:
            if arg in params:
                self.knn_args[arg] = params[arg]

        # update subgraph parameters
        [g.set_params(**knn_other_args) for g in self.subgraphs]

        # update superclass parameters
        super().set_params(**params)
        return self

    def build_kernel(self):
        """Build the MNN kernel.

        Build a mutual nearest neighbors kernel.

        Returns
        -------
        K : kernel matrix, shape=[n_samples, n_samples]
            symmetric matrix with ones down the diagonal
            with no non-negative entries.
        """
        log_start("subgraphs")
        self.subgraphs = []
        if 'n_landmark' in self.knn_args:
            del self.knn_args['n_landmark']
        # iterate through sample ids
        for i, idx in enumerate(self.samples):
            log_debug("subgraph {}: sample {}".format(i, idx))
            # select data for sample
            data = self.data_nu[self.sample_idx == idx]
            # build a kNN graph for cells within sample
            graph = Graph(data, n_pca=None,
                          knn=self.weighted_knn[i],
                          initialize=False,
                          **(self.knn_args))
            self.subgraphs.append(graph)  # append to list of subgraphs
        log_complete("subgraphs")

        if isinstance(self.subgraphs[0], kNNGraph):
            K = sparse.lil_matrix(
                (self.data_nu.shape[0], self.data_nu.shape[0]))
        else:
            K = np.zeros([self.data_nu.shape[0], self.data_nu.shape[0]])
        for i, X in enumerate(self.subgraphs):
            for j, Y in enumerate(self.subgraphs):
                log_start(
                    "kernel from sample {} to {}".format(self.samples[i],
                                                         self.samples[j]))
                Kij = Y.build_kernel_to_data(
                    X.data_nu,
                    knn=self.weighted_knn[i])
                if i == j:
                    # downweight within-batch affinities by beta
                    Kij = Kij * self.beta
                K = set_submatrix(K, self.sample_idx == i,
                                  self.sample_idx == j, Kij)
                log_complete(
                    "kernel from sample {} to {}".format(self.samples[i],
                                                         self.samples[j]))

        if not (isinstance(self.gamma, str) or
                isinstance(self.gamma, numbers.Number)):
            # matrix gamma
            # Gamma can be a matrix with specific values transitions for
            # each batch. This allows for technical replicates and
            # experimental samples to be corrected simultaneously
            for i in range(len(self.samples)):
                for j in range(i, len(self.samples)):
                    Kij = K[self.sample_idx == i, :][:, self.sample_idx == j]
                    Kji = K[self.sample_idx == j, :][:, self.sample_idx == i]
                    Kij_symm = self.gamma[i, j] * \
                        elementwise_minimum(Kij, Kji.T) + \
                        (1 - self.gamma[i, j]) * \
                        elementwise_maximum(Kij, Kji.T)
                    K = set_submatrix(K, self.sample_idx == i,
                                      self.sample_idx == j, Kij_symm)
                    if not i == j:
                        K = set_submatrix(K, self.sample_idx == j,
                                          self.sample_idx == i, Kij_symm.T)
        else:
            # symmetrize
            if isinstance(self.gamma, str):
                if self.gamma == "+":
                    K = (K + K.T) / 2
                elif self.gamma == "*":
                    K = K.multiply(K.T)
            elif isinstance(self.gamma, numbers.Number):
                K = self.gamma * elementwise_minimum(K, K.T) + \
                    (1 - self.gamma) * elementwise_maximum(K, K.T)
            else:
                # this should never happen
                raise ValueError("invalid gamma")
        return K

    def build_kernel_to_data(self, Y, gamma=None):
        """Build transition matrix from new data to the graph

        Creates a transition matrix such that `Y` can be approximated by
        a linear combination of landmarks. Any
        transformation of the landmarks can be trivially applied to `Y` by
        performing

        TODO: test this.

        `transform_Y = transitions.dot(transform)`

        Parameters
        ----------

        Y : array-like, [n_samples_y, n_dimensions]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        gamma : array-like or `None`, optional (default: `None`)
            if `self.gamma` is a matrix, gamma values must be explicitly
            specified between `Y` and each sample in `self.data`

        Returns
        -------

        transitions : array-like, [n_samples_y, self.data.shape[0]]
            Transition matrix from `Y` to `self.data`
        """
        log_warning("building MNN kernel to gamma is experimental")
        if not isinstance(self.gamma, str) and \
                not isinstance(self.gamma, numbers.Number):
            if gamma is None:
                raise ValueError(
                    "self.gamma is a matrix but gamma is not provided.")
            elif len(gamma) != len(self.samples):
                raise ValueError(
                    "gamma should have one value for every sample")

        Y = self._check_extension_shape(Y)
        kernel_xy = []
        kernel_yx = []
        # don't really need within Y kernel
        Y_graph = kNNGraph(Y, n_pca=None, knn=0, **(self.knn_args))
        y_knn = self._weight_knn(sample_size=len(Y))
        for i, X in enumerate(self.subgraphs):
            kernel_xy.append(X.build_kernel_to_data(
                Y, knn=self.weighted_knn[i]))  # kernel X -> Y
            kernel_yx.append(Y_graph.build_kernel_to_data(
                X.data_nu, knn=y_knn))  # kernel Y -> X
        kernel_xy = sparse.hstack(kernel_xy)  # n_cells_y x n_cells_x
        kernel_yx = sparse.vstack(kernel_yx)  # n_cells_x x n_cells_y

        # symmetrize
        if gamma is not None:
            # Gamma can be a vector with specific values transitions for
            # each batch. This allows for technical replicates and
            # experimental samples to be corrected simultaneously
            K = np.empty_like(kernel_xy)
            for i, sample in enumerate(self.samples):
                sample_idx = self.sample_idx == sample
                K[:, sample_idx] = gamma[i] * \
                    kernel_xy[:, sample_idx].minimum(
                        kernel_yx[sample_idx, :].T) + \
                    (1 - gamma[i]) * \
                    kernel_xy[:, sample_idx].maximum(
                        kernel_yx[sample_idx, :].T)
        if self.gamma == "+":
            K = (kernel_xy + kernel_yx.T) / 2
        elif self.gamma == "*":
            K = kernel_xy.multiply(kernel_yx.T)
        else:
            K = self.gamma * kernel_xy.minimum(kernel_yx.T) + \
                (1 - self.gamma) * kernel_xy.maximum(kernel_yx.T)
        return K


def Graph(data,
          n_pca=None,
          sample_idx=None,
          adaptive_k='sqrt',
          precomputed=None,
          knn=5,
          decay=None,
          distance='euclidean',
          thresh=0,
          n_landmark=None,
          n_svd=100,
          beta=1,
          gamma=0.5,
          n_jobs=-1,
          verbose=False,
          random_state=None,
          graphtype='auto',
          **kwargs):
    """Create a graph built on data.

    Automatically selects the appropriate DataGraph subclass based on
    chosen parameters.
    Selection criteria:
    - if `graphtype` is given, this will be respected
    - otherwise:
    -- if `sample_idx` is given, an MNNGraph will be created
    -- if `precomputed` is not given, and either `decay` is `None` or `thresh`
    is given, a kNNGraph will be created
    - otherwise, a TraditionalGraph will be created.

    Incompatibilities:
    - MNNGraph and kNNGraph cannot be precomputed
    - kNNGraph and TraditionalGraph do not accept sample indices

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

    knn : `int`, optional (default: 5)
        Number of nearest neighbors (including self) to use to build the graph

    decay : `int` or `None`, optional (default: `None`)
        Rate of alpha decay to use. If `None`, alpha decay is not used.

    distance : `str`, optional (default: `'euclidean'`)
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph.
        TODO: actually sklearn.neighbors has even more choices

    thresh : `float`, optional (default: `1e-5`)
        Threshold above which to calculate alpha decay kernel.
        All affinities below `thresh` will be set to zero in order to save
        on time and memory constraints.

    precomputed : {'distance', 'affinity', 'adjacency', `None`}, optional (default: `None`)
        If the graph is precomputed, this variable denotes which graph
        matrix is provided as `data`.
        Only one of `precomputed` and `n_pca` can be set.

    beta: float, optional(default: 1)
        Multiply within - batch connections by(1 - beta)

    gamma: float or {'+', '*'} (default: 0.99)
        Symmetrization method. If '+', use `(K + K.T) / 2`,
        if '*', use `K * K.T`, if a float, use
        `gamma * min(K, K.T) + (1 - gamma) * max(K, K.T)`

    sample_idx: array-like
        Batch index for MNN kernel

    adaptive_k : `{'min', 'mean', 'sqrt', 'none'}` (default: 'sqrt')
        Weights MNN kernel adaptively using the number of cells in
        each sample according to the selected method.

    n_landmark : `int`, optional (default: 2000)
        number of landmarks to use

    n_svd : `int`, optional (default: 100)
        number of SVD components to use for spectral clustering

    random_state : `int` or `None`, optional (default: `None`)
        Random state for random PCA

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

    Returns
    -------
    G : `DataGraph`

    Raises
    ------
    ValueError : if selected parameters are incompatible.
    """
    if sample_idx is not None and len(np.unique(sample_idx)) == 1:
        warnings.warn("Only one unique sample. "
                      "Not using MNNGraph")
        sample_idx = None
        if graphtype == 'mnn':
            graphtype = 'auto'
    if graphtype == 'auto':
        # automatic graph selection
        if sample_idx is not None:
            # only mnn does batch correction
            graphtype = "mnn"
        elif precomputed is None and (decay is None or thresh > 0):
            # precomputed requires exact graph
            # no decay or threshold decay require knngraph
            graphtype = "knn"
        else:
            graphtype = "exact"

    # set base graph type
    if graphtype == "knn":
        base = kNNGraph
        if precomputed is not None:
            raise ValueError("kNNGraph does not support precomputed "
                             "values. Use `graphtype='exact'` or "
                             "`precomputed=None`")
        if sample_idx is not None:
            raise ValueError("kNNGraph does not support batch "
                             "correction. Use `graphtype='mnn'` or "
                             "`sample_idx=None`")

    elif graphtype == "mnn":
        base = MNNGraph
        if precomputed is not None:
            raise ValueError("MNNGraph does not support precomputed "
                             "values. Use `graphtype='exact'` and "
                             "`sample_idx=None` or `precomputed=None`")
    elif graphtype == "exact":
        base = TraditionalGraph
        if sample_idx is not None:
            raise ValueError("TraditionalGraph does not support batch "
                             "correction. Use `graphtype='mnn'` or "
                             "`sample_idx=None`")
    else:
        raise ValueError("graphtype '{}' not recognized. Choose from "
                         "['knn', 'mnn', 'exact', 'auto']")

    # set add landmarks if necessary
    if n_landmark is not None:
        log_debug("Building {} graph with landmarks".format(graphtype))

        class Graph(base, LandmarkGraph):
            pass
    else:
        log_debug("Building {} graph".format(graphtype))

        class Graph(base):
            pass

    # build graph and return
    return Graph(data,
                 n_pca=n_pca,
                 sample_idx=sample_idx,
                 adaptive_k=adaptive_k,
                 precomputed=precomputed,
                 knn=knn,
                 decay=decay,
                 distance=distance,
                 thresh=thresh,
                 n_landmark=n_landmark,
                 n_svd=n_svd,
                 beta=beta,
                 gamma=gamma,
                 n_jobs=n_jobs,
                 verbose=verbose,
                 random_state=random_state,
                 **kwargs)
