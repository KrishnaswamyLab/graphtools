from builtins import super
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from scipy import sparse
import numbers
import warnings
import tasklogger

from .utils import (set_diagonal,
                    elementwise_minimum,
                    elementwise_maximum,
                    set_submatrix)
from .base import DataGraph, PyGSPGraph


class kNNGraph(DataGraph):
    """
    K nearest neighbors graph

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`,
        `pandas.DataFrame`, `pandas.SparseDataFrame`.

    knn : `int`, optional (default: 5)
        Number of nearest neighbors (including self) to use to build the graph

    decay : `int` or `None`, optional (default: `None`)
        Rate of alpha decay to use. If `None`, alpha decay is not used.

    distance : `str`, optional (default: `'euclidean'`)
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph.
        TODO: actually sklearn.neighbors has even more choices

    thresh : `float`, optional (default: `1e-4`)
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
                 thresh=1e-4, n_pca=None, **kwargs):
        self.knn = knn
        self.decay = decay
        self.distance = distance
        self.thresh = thresh

        if decay is not None and thresh <= 0:
            raise ValueError("Cannot instantiate a kNNGraph with `decay=None` "
                             "and `thresh=0`. Use a TraditionalGraph instead.")
        if knn > data.shape[0]:
            warnings.warn("Cannot set knn ({k}) to be greater than "
                          "data.shape[0] ({n}). Setting knn={n}".format(
                              k=knn, n=data.shape[0]))
        if n_pca is None and data.shape[1] > 500:
            warnings.warn("Building a kNNGraph on data of shape {} is "
                          "expensive. Consider setting n_pca.".format(
                              data.shape), UserWarning)

        super().__init__(data, n_pca=n_pca, **kwargs)

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
            if hasattr(self, "_knn_tree"):
                self.knn_tree.set_params(n_jobs=self.n_jobs)
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
                warnings.warn(
                    "Metric {} not valid for `sklearn.neighbors.BallTree`. "
                    "Graph instantiation may be slower than normal.".format(
                        self.distance),
                    UserWarning)
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
        K = self.build_kernel_to_data(self.data_nu)
        return K

    def build_kernel_to_data(self, Y, knn=None):
        """Build a kernel from new input data `Y` to the `self.data`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_features]
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
        if knn > self.data.shape[0]:
            warnings.warn("Cannot set knn ({k}) to be greater than "
                          "data.shape[0] ({n}). Setting knn={n}".format(
                              k=knn, n=self.data.shape[0]))

        Y = self._check_extension_shape(Y)
        tasklogger.log_start("KNN search")
        if self.decay is None or self.thresh == 1:
            # binary connectivity matrix
            K = self.knn_tree.kneighbors_graph(
                Y, n_neighbors=knn,
                mode='connectivity')
            tasklogger.log_complete("KNN search")
        else:
            # sparse fast alpha decay
            knn_tree = self.knn_tree
            search_knn = min(knn * 20, self.data_nu.shape[0])
            distances, indices = knn_tree.kneighbors(
                Y, n_neighbors=search_knn)
            if np.any(distances[:, 1] == 0):
                has_duplicates = distances[:, 1] == 0
                idx = np.argwhere((distances == 0) & has_duplicates[:, None])
                duplicate_ids = np.array(
                    [[indices[i[0], i[1]], i[0]]
                     for i in idx if indices[i[0], i[1]] < i[0]])
                duplicate_ids = duplicate_ids[np.argsort(duplicate_ids[:, 0])]
                duplicate_names = ", ".join(["{} and {}".format(i[0], i[1])
                                             for i in duplicate_ids])
                warnings.warn(
                    "Detected zero distance between samples {}. "
                    "Consider removing duplicates to avoid errors in "
                    "downstream processing.".format(duplicate_names),
                    RuntimeWarning)
            tasklogger.log_complete("KNN search")
            tasklogger.log_start("affinities")
            bandwidth = distances[:, knn - 1]
            radius = bandwidth * np.power(-1 * np.log(self.thresh),
                                          1 / self.decay)
            update_idx = np.argwhere(
                np.max(distances, axis=1) < radius).reshape(-1)
            tasklogger.log_debug("search_knn = {}; {} remaining".format(
                search_knn, len(update_idx)))
            if len(update_idx) > 0:
                distances = [d for d in distances]
                indices = [i for i in indices]
            while len(update_idx) > Y.shape[0] // 10 and \
                    search_knn < self.data_nu.shape[0] / 2:
                # increase the knn search
                search_knn = min(search_knn * 20, self.data_nu.shape[0])
                dist_new, ind_new = knn_tree.kneighbors(
                    Y[update_idx], n_neighbors=search_knn)
                for i, idx in enumerate(update_idx):
                    distances[idx] = dist_new[i]
                    indices[idx] = ind_new[i]
                update_idx = [i for i, d in enumerate(distances)
                              if np.max(d) < radius[i]]
                tasklogger.log_debug("search_knn = {}; {} remaining".format(
                    search_knn,
                    len(update_idx)))
            if search_knn > self.data_nu.shape[0] / 2:
                knn_tree = NearestNeighbors(
                    knn, algorithm='brute',
                    n_jobs=self.n_jobs).fit(self.data_nu)
            if len(update_idx) > 0:
                tasklogger.log_debug(
                    "radius search on {}".format(len(update_idx)))
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
            K = sparse.csr_matrix((data, indices, indptr),
                                  shape=(Y.shape[0], self.data_nu.shape[0]))
            K.data = np.exp(-1 * np.power(K.data, self.decay))
            # handle nan
            K.data = np.where(np.isnan(K.data), 1, K.data)
            # TODO: should we zero values that are below thresh?
            K.data[K.data < self.thresh] = 0
            K = K.tocoo()
            K.eliminate_zeros()
            K = K.tocsr()
            tasklogger.log_complete("affinities")
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
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`.,
        `pandas.DataFrame`, `pandas.SparseDataFrame`.

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

    def _landmarks_to_data(self):
        landmarks = np.unique(self._clusters)
        if sparse.issparse(self.kernel):
            pmn = sparse.vstack(
                [sparse.csr_matrix(self.kernel[self._clusters == i, :].sum(
                    axis=0)) for i in landmarks])
        else:
            pmn = np.array([np.sum(self.kernel[self._clusters == i, :], axis=0)
                            for i in landmarks])
        return pmn

    def _data_transitions(self):
        return normalize(self._landmarks_to_data(), 'l1', axis=1)

    def build_landmark_op(self):
        """Build the landmark operator

        Calculates spectral clusters on the kernel, and calculates transition
        probabilities between cluster centers by using transition probabilities
        between samples assigned to each cluster.
        """
        tasklogger.log_start("landmark operator")
        is_sparse = sparse.issparse(self.kernel)
        # spectral clustering
        tasklogger.log_start("SVD")
        _, _, VT = randomized_svd(self.diff_aff,
                                  n_components=self.n_svd,
                                  random_state=self.random_state)
        tasklogger.log_complete("SVD")
        tasklogger.log_start("KMeans")
        kmeans = MiniBatchKMeans(
            self.n_landmark,
            init_size=3 * self.n_landmark,
            batch_size=10000,
            random_state=self.random_state)
        self._clusters = kmeans.fit_predict(
            self.diff_op.dot(VT.T))
        # some clusters are not assigned
        tasklogger.log_complete("KMeans")

        # transition matrices
        pmn = self._landmarks_to_data()

        # row normalize
        pnm = pmn.transpose()
        pmn = normalize(pmn, norm='l1', axis=1)
        pnm = normalize(pnm, norm='l1', axis=1)
        landmark_op = pmn.dot(pnm)  # sparsity agnostic matrix multiplication
        if is_sparse:
            # no need to have a sparse landmark operator
            landmark_op = landmark_op.toarray()
        # store output
        self._landmark_op = landmark_op
        self._transitions = pnm
        tasklogger.log_complete("landmark operator")

    def extend_to_data(self, data, **kwargs):
        """Build transition matrix from new data to the graph

        Creates a transition matrix such that `Y` can be approximated by
        a linear combination of landmarks. Any
        transformation of the landmarks can be trivially applied to `Y` by
        performing

        `transform_Y = transitions.dot(transform)`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_features]
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

        Y: array-like, optional, shape=[n_samples_y, n_features]
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

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`,
        `pandas.DataFrame`, `pandas.SparseDataFrame`.
        If `precomputed` is not `None`, data should be an
        [n_samples, n_samples] matrix denoting pairwise distances,
        affinities, or edge weights.

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

    thresh : `float`, optional (default: `1e-4`)
        Threshold above which to calculate alpha decay kernel.
        All affinities below `thresh` will be set to zero in order to save
        on time and memory constraints.

    precomputed : {'distance', 'affinity', 'adjacency', `None`},
        optional (default: `None`)
        If the graph is precomputed, this variable denotes which graph
        matrix is provided as `data`.
        Only one of `precomputed` and `n_pca` can be set.
    """

    def __init__(self, data, knn=5, decay=10,
                 distance='euclidean', n_pca=None,
                 thresh=1e-4,
                 precomputed=None, **kwargs):
        if precomputed is not None and n_pca is not None:
            # the data itself is a matrix of distances / affinities
            n_pca = None
            warnings.warn("n_pca cannot be given on a precomputed graph."
                          " Setting n_pca=None", RuntimeWarning)
        if decay is None and precomputed not in ['affinity', 'adjacency']:
            # decay high enough is basically a binary kernel
            raise ValueError("`decay` must be provided for a TraditionalGraph"
                             ". For kNN kernel, use kNNGraph.")
        if precomputed is not None:
            if precomputed not in ["distance", "affinity", "adjacency"]:
                raise ValueError("Precomputed value {} not recognized. "
                                 "Choose from ['distance', 'affinity', "
                                 "'adjacency']")
            elif data.shape[0] != data.shape[1]:
                raise ValueError("Precomputed {} must be a square matrix. "
                                 "{} was given".format(precomputed,
                                                       data.shape))
            elif (data < 0).sum() > 0:
                raise ValueError("Precomputed {} should be "
                                 "non-negative".format(precomputed))
        self.knn = knn
        self.decay = decay
        self.distance = distance
        self.thresh = thresh
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
                self.precomputed is None:
            raise ValueError("Cannot update distance. "
                             "Please create a new graph")
        if 'knn' in params and params['knn'] != self.knn and \
                self.precomputed is None:
            raise ValueError("Cannot update knn. Please create a new graph")
        if 'decay' in params and params['decay'] != self.decay and \
                self.precomputed is None:
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
        if self.precomputed == "affinity":
            # already done
            # TODO: should we check that precomputed matrices look okay?
            # e.g. check the diagonal
            K = self.data_nu
        elif self.precomputed == "adjacency":
            # need to set diagonal to one to make it an affinity matrix
            K = self.data_nu
            if sparse.issparse(K) and \
                not (isinstance(K, sparse.dok_matrix) or
                     isinstance(K, sparse.lil_matrix)):
                K = K.tolil()
            K = set_diagonal(K, 1)
        else:
            tasklogger.log_start("affinities")
            if sparse.issparse(self.data_nu):
                self.data_nu = self.data_nu.toarray()
            if self.precomputed == "distance":
                pdx = self.data_nu
            elif self.precomputed is None:
                pdx = pdist(self.data_nu, metric=self.distance)
                if np.any(pdx == 0):
                    pdx = squareform(pdx)
                    duplicate_ids = np.array(
                        [i for i in np.argwhere(pdx == 0)
                         if i[1] > i[0]])
                    duplicate_names = ", ".join(["{} and {}".format(i[0], i[1])
                                                 for i in duplicate_ids])
                    warnings.warn(
                        "Detected zero distance between samples {}. "
                        "Consider removing duplicates to avoid errors in "
                        "downstream processing.".format(duplicate_names),
                        RuntimeWarning)
                else:
                    pdx = squareform(pdx)
            else:
                raise ValueError(
                    "precomputed='{}' not recognized. "
                    "Choose from ['affinity', 'adjacency', 'distance', "
                    "None]".format(self.precomputed))
            knn_dist = np.partition(pdx, self.knn, axis=1)[:, :self.knn]
            epsilon = np.max(knn_dist, axis=1)
            pdx = (pdx.T / epsilon).T
            K = np.exp(-1 * np.power(pdx, self.decay))
            # handle nan
            K = np.where(np.isnan(K), 1, K)
            tasklogger.log_complete("affinities")
        # truncate
        if sparse.issparse(K):
            if not (isinstance(K, sparse.csr_matrix) or
                    isinstance(K, sparse.csc_matrix) or
                    isinstance(K, sparse.bsr_matrix)):
                K = K.tocsr()
            K.data[K.data < self.thresh] = 0
            K = K.tocoo()
            K.eliminate_zeros()
            K = K.tocsr()
        else:
            K[K < self.thresh] = 0
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

        Y: array-like, [n_samples_y, n_features]
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
            tasklogger.log_start("affinities")
            Y = self._check_extension_shape(Y)
            pdx = cdist(Y, self.data_nu, metric=self.distance)
            knn_dist = np.partition(pdx, knn, axis=1)[:, :knn]
            epsilon = np.max(knn_dist, axis=1)
            pdx = (pdx.T / epsilon).T
            K = np.exp(-1 * pdx**self.decay)
            # handle nan
            K = np.where(np.isnan(K), 1, K)
            K[K < self.thresh] = 0
            tasklogger.log_complete("affinities")
        return K


class MNNGraph(DataGraph):
    """Mutual nearest neighbors graph

    Performs batch correction by forcing connections between batches, but
    only when the connection is mutual (i.e. x is a neighbor of y _and_
    y is a neighbor of x).

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`,
        `scipy.sparse.spmatrix`.,
        `pandas.DataFrame`, `pandas.SparseDataFrame`.

    sample_idx: array-like, shape=[n_samples]
        Batch index

    beta: `float`, optional (default: 1)
        Downweight within-batch affinities by beta

    adaptive_k : {'min', 'mean', 'sqrt', `None`} (default: 'sqrt')
        Weights MNN kernel adaptively using the number of cells in
        each sample according to the selected method.

    Attributes
    ----------
    subgraphs : list of `graphtools.graphs.kNNGraph`s
        Graphs representing each batch separately
    """

    def __init__(self, data, sample_idx,
                 knn=5, beta=1, n_pca=None,
                 adaptive_k='sqrt',
                 decay=None,
                 distance='euclidean',
                 thresh=1e-4,
                 n_jobs=1,
                 **kwargs):
        self.beta = beta
        self.sample_idx = sample_idx
        self.samples, self.n_cells = np.unique(
            self.sample_idx, return_counts=True)
        self.adaptive_k = adaptive_k
        self.knn = knn
        self.decay = decay
        self.distance = distance
        self.thresh = thresh
        self.n_jobs = n_jobs
        self.weighted_knn = self._weight_knn()

        if sample_idx is None:
            raise ValueError("sample_idx must be given. For a graph without"
                             " batch correction, use kNNGraph.")
        elif len(sample_idx) != data.shape[0]:
            raise ValueError("sample_idx ({}) must be the same length as "
                             "data ({})".format(len(sample_idx),
                                                data.shape[0]))
        elif len(self.samples) == 1:
            raise ValueError(
                "sample_idx must contain more than one unique value")

        super().__init__(data, n_pca=n_pca, **kwargs)

    def _check_symmetrization(self, kernel_symm, gamma):
        if kernel_symm == 'gamma' and gamma is not None and \
                not isinstance(gamma, numbers.Number):
            # matrix gamma
            try:
                gamma.shape
            except AttributeError:
                raise ValueError("gamma {} not recognized. "
                                 "Expected a float between 0 and 1 "
                                 "or a [n_batch,n_batch] matrix of "
                                 "floats between 0 and 1".format(gamma))
            if not np.shape(gamma) == (len(self.samples),
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
        else:
            super()._check_symmetrization(kernel_symm, gamma)

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
        elif self.adaptive_k is None:
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
                       'adaptive_k': self.adaptive_k,
                       'knn': self.knn,
                       'decay': self.decay,
                       'distance': self.distance,
                       'thresh': self.thresh,
                       'n_jobs': self.n_jobs})
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
        if 'adaptive_k' in params and params['adaptive_k'] != self.adaptive_k:
            raise ValueError(
                "Cannot update adaptive_k. Please create a new graph")

        # knn arguments
        knn_kernel_args = ['knn', 'decay', 'distance', 'thresh']
        knn_other_args = ['n_jobs', 'random_state', 'verbose']
        for arg in knn_kernel_args:
            if arg in params and params[arg] != getattr(self, arg):
                raise ValueError("Cannot update {}. "
                                 "Please create a new graph".format(arg))
        for arg in knn_other_args:
            if arg in params:
                self.__setattr__(arg, params[arg])
                for g in self.subgraphs:
                    g.set_params(**{arg: params[arg]})

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
        tasklogger.log_start("subgraphs")
        self.subgraphs = []
        from .api import Graph
        # iterate through sample ids
        for i, idx in enumerate(self.samples):
            tasklogger.log_debug("subgraph {}: sample {}, "
                                 "n = {}, knn = {}".format(
                                     i, idx, np.sum(self.sample_idx == idx),
                                     self.weighted_knn[i]))
            # select data for sample
            data = self.data_nu[self.sample_idx == idx]
            # build a kNN graph for cells within sample
            graph = Graph(data, n_pca=None,
                          knn=self.weighted_knn[i],
                          decay=self.decay,
                          distance=self.distance,
                          thresh=self.thresh,
                          verbose=self.verbose,
                          random_state=self.random_state,
                          n_jobs=self.n_jobs,
                          initialize=False)
            self.subgraphs.append(graph)  # append to list of subgraphs
        tasklogger.log_complete("subgraphs")

        if self.thresh > 0 or self.decay is None:
            K = sparse.lil_matrix(
                (self.data_nu.shape[0], self.data_nu.shape[0]))
        else:
            K = np.zeros([self.data_nu.shape[0], self.data_nu.shape[0]])
        for i, X in enumerate(self.subgraphs):
            for j, Y in enumerate(self.subgraphs):
                tasklogger.log_start(
                    "kernel from sample {} to {}".format(self.samples[i],
                                                         self.samples[j]))
                Kij = Y.build_kernel_to_data(
                    X.data_nu,
                    knn=self.weighted_knn[i])
                if i == j:
                    # downweight within-batch affinities by beta
                    Kij = Kij * self.beta
                K = set_submatrix(K, self.sample_idx == self.samples[i],
                                  self.sample_idx == self.samples[j], Kij)
                tasklogger.log_complete(
                    "kernel from sample {} to {}".format(self.samples[i],
                                                         self.samples[j]))
        return K

    def symmetrize_kernel(self, K):
        if self.kernel_symm == 'gamma' and self.gamma is not None and \
                not isinstance(self.gamma, numbers.Number):
            # matrix gamma
            # Gamma can be a matrix with specific values transitions for
            # each batch. This allows for technical replicates and
            # experimental samples to be corrected simultaneously
            tasklogger.log_debug("Using gamma symmetrization. "
                                 "Gamma:\n{}".format(self.gamma))
            for i, sample_i in enumerate(self.samples):
                for j, sample_j in enumerate(self.samples):
                    if j < i:
                        continue
                    Kij = K[np.ix_(self.sample_idx == sample_i,
                                   self.sample_idx == sample_j)]
                    Kji = K[np.ix_(self.sample_idx == sample_j,
                                   self.sample_idx == sample_i)]
                    Kij_symm = self.gamma[i, j] * \
                        elementwise_minimum(Kij, Kji.T) + \
                        (1 - self.gamma[i, j]) * \
                        elementwise_maximum(Kij, Kji.T)
                    K = set_submatrix(K, self.sample_idx == sample_i,
                                      self.sample_idx == sample_j, Kij_symm)
                    if not i == j:
                        K = set_submatrix(K, self.sample_idx == sample_j,
                                          self.sample_idx == sample_i,
                                          Kij_symm.T)
        else:
            K = super().symmetrize_kernel(K)
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

        Y : array-like, [n_samples_y, n_features]
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
        raise NotImplementedError
        tasklogger.log_warning("building MNN kernel to gamma is experimental")
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
        y_knn = self._weight_knn(sample_size=Y.shape[0])
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


class kNNLandmarkGraph(kNNGraph, LandmarkGraph):
    pass


class MNNLandmarkGraph(MNNGraph, LandmarkGraph):
    pass


class TraditionalLandmarkGraph(TraditionalGraph, LandmarkGraph):
    pass


class kNNPyGSPGraph(kNNGraph, PyGSPGraph):
    pass


class MNNPyGSPGraph(MNNGraph, PyGSPGraph):
    pass


class TraditionalPyGSPGraph(TraditionalGraph, PyGSPGraph):
    pass


class kNNLandmarkPyGSPGraph(kNNGraph, LandmarkGraph, PyGSPGraph):
    pass


class MNNLandmarkPyGSPGraph(MNNGraph, LandmarkGraph, PyGSPGraph):
    pass


class TraditionalLandmarkPyGSPGraph(TraditionalGraph, LandmarkGraph, PyGSPGraph):
    pass
