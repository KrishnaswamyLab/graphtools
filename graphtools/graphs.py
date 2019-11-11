from __future__ import division
from builtins import super
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform
from scipy import sparse
import numbers
import warnings
import tasklogger

from . import utils
from .base import DataGraph, PyGSPGraph

_logger = tasklogger.get_tasklogger("graphtools")


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

    bandwidth : `float`, list-like,`callable`, or `None`,
                optional (default: `None`)
        Fixed bandwidth to use. If given, overrides `knn`. Can be a single
        bandwidth, or a list-like (shape=[n_samples]) of bandwidths for each
        sample

    bandwidth_scale : `float`, optional (default : 1.0)
        Rescaling factor for bandwidth.

    distance : `str`, optional (default: `'euclidean'`)
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph. Custom distance
        functions of form `f(x, y) = d` are also accepted.
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

    def __init__(
        self,
        data,
        knn=5,
        decay=None,
        knn_max=None,
        search_multiplier=20,
        bandwidth=None,
        bandwidth_scale=1.0,
        distance="euclidean",
        thresh=1e-4,
        n_pca=None,
        **kwargs
    ):

        if decay is not None:
            if thresh <= 0 and knn_max is None:
                raise ValueError(
                    "Cannot instantiate a kNNGraph with `decay=None`, "
                    "`thresh=0` and `knn_max=None`. Use a TraditionalGraph instead."
                )
            elif thresh < np.finfo(float).eps:
                thresh = np.finfo(float).eps

        if callable(bandwidth):
            raise NotImplementedError(
                "Callable bandwidth is only supported by"
                " graphtools.graphs.TraditionalGraph."
            )
        if knn is None and bandwidth is None:
            raise ValueError("Either `knn` or `bandwidth` must be provided.")
        elif knn is None and bandwidth is not None:
            # implementation requires a knn value
            knn = 5
        if decay is None and bandwidth is not None:
            warnings.warn("`bandwidth` is not used when `decay=None`.", UserWarning)
        if knn > data.shape[0] - 2:
            warnings.warn(
                "Cannot set knn ({k}) to be greater than "
                "n_samples ({n}). Setting knn={n}".format(k=knn, n=data.shape[0] - 2)
            )
            knn = data.shape[0] - 2
        if knn_max is not None and knn_max < knn:
            warnings.warn(
                "Cannot set knn_max ({knn_max}) to be less than "
                "knn ({knn}). Setting knn_max={knn}".format(knn=knn, knn_max=knn_max)
            )
            knn_max = knn
        if n_pca in [None, 0, False] and data.shape[1] > 500:
            warnings.warn(
                "Building a kNNGraph on data of shape {} is "
                "expensive. Consider setting n_pca.".format(data.shape),
                UserWarning,
            )

        self.knn = knn
        self.knn_max = knn_max
        self.search_multiplier = search_multiplier
        self.decay = decay
        self.bandwidth = bandwidth
        self.bandwidth_scale = bandwidth_scale
        self.distance = distance
        self.thresh = thresh
        super().__init__(data, n_pca=n_pca, **kwargs)

    def get_params(self):
        """Get parameters from this object
        """
        params = super().get_params()
        params.update(
            {
                "knn": self.knn,
                "decay": self.decay,
                "bandwidth": self.bandwidth,
                "bandwidth_scale": self.bandwidth_scale,
                "knn_max": self.knn_max,
                "distance": self.distance,
                "thresh": self.thresh,
                "n_jobs": self.n_jobs,
                "random_state": self.random_state,
                "verbose": self.verbose,
            }
        )
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
        - knn_max
        - decay
        - bandwidth
        - bandwidth_scale
        - distance
        - thresh

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        if "knn" in params and params["knn"] != self.knn:
            raise ValueError("Cannot update knn. Please create a new graph")
        if "knn_max" in params and params["knn_max"] != self.knn:
            raise ValueError("Cannot update knn_max. Please create a new graph")
        if "decay" in params and params["decay"] != self.decay:
            raise ValueError("Cannot update decay. Please create a new graph")
        if "bandwidth" in params and params["bandwidth"] != self.bandwidth:
            raise ValueError("Cannot update bandwidth. Please create a new graph")
        if (
            "bandwidth_scale" in params
            and params["bandwidth_scale"] != self.bandwidth_scale
        ):
            raise ValueError("Cannot update bandwidth_scale. Please create a new graph")
        if "distance" in params and params["distance"] != self.distance:
            raise ValueError("Cannot update distance. " "Please create a new graph")
        if "thresh" in params and params["thresh"] != self.thresh and self.decay != 0:
            raise ValueError("Cannot update thresh. Please create a new graph")
        if "n_jobs" in params:
            self.n_jobs = params["n_jobs"]
            if hasattr(self, "_knn_tree"):
                self.knn_tree.set_params(n_jobs=self.n_jobs)
        if "random_state" in params:
            self.random_state = params["random_state"]
        if "verbose" in params:
            self.verbose = params["verbose"]
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
                    n_neighbors=self.knn + 1,
                    algorithm="ball_tree",
                    metric=self.distance,
                    n_jobs=self.n_jobs,
                ).fit(self.data_nu)
            except ValueError:
                # invalid metric
                warnings.warn(
                    "Metric {} not valid for `sklearn.neighbors.BallTree`. "
                    "Graph instantiation may be slower than normal.".format(
                        self.distance
                    ),
                    UserWarning,
                )
                self._knn_tree = NearestNeighbors(
                    n_neighbors=self.knn + 1,
                    algorithm="auto",
                    metric=self.distance,
                    n_jobs=self.n_jobs,
                ).fit(self.data_nu)
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
        knn_max = self.knn_max + 1 if self.knn_max else None
        K = self.build_kernel_to_data(self.data_nu, knn=self.knn + 1, knn_max=knn_max)
        return K

    def _check_duplicates(self, distances, indices):
        if np.any(distances[:, 1] == 0):
            has_duplicates = distances[:, 1] == 0
            if np.sum(distances[:, 1:] == 0) < 20:
                idx = np.argwhere((distances == 0) & has_duplicates[:, None])
                duplicate_ids = np.array(
                    [
                        [indices[i[0], i[1]], i[0]]
                        for i in idx
                        if indices[i[0], i[1]] < i[0]
                    ]
                )
                duplicate_ids = duplicate_ids[np.argsort(duplicate_ids[:, 0])]
                duplicate_names = ", ".join(
                    ["{} and {}".format(i[0], i[1]) for i in duplicate_ids]
                )
                warnings.warn(
                    "Detected zero distance between samples {}. "
                    "Consider removing duplicates to avoid errors in "
                    "downstream processing.".format(duplicate_names),
                    RuntimeWarning,
                )
            else:
                warnings.warn(
                    "Detected zero distance between {} pairs of samples. "
                    "Consider removing duplicates to avoid errors in "
                    "downstream processing.".format(
                        np.sum(np.sum(distances[:, 1:] == 0))
                    ),
                    RuntimeWarning,
                )

    def build_kernel_to_data(
        self, Y, knn=None, knn_max=None, bandwidth=None, bandwidth_scale=None
    ):
        """Build a kernel from new input data `Y` to the `self.data`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_features]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        knn : `int` or `None`, optional (default: `None`)
            If `None`, defaults to `self.knn`

        bandwidth : `float`, `callable`, or `None`, optional (default: `None`)
            If `None`, defaults to `self.bandwidth`

        bandwidth_scale : `float`, optional (default : `None`)
            Rescaling factor for bandwidth.
            If `None`, defaults to self.bandwidth_scale

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
        if bandwidth is None:
            bandwidth = self.bandwidth
        if bandwidth_scale is None:
            bandwidth_scale = self.bandwidth_scale
        if knn > self.data.shape[0]:
            warnings.warn(
                "Cannot set knn ({k}) to be greater than "
                "n_samples ({n}). Setting knn={n}".format(
                    k=knn, n=self.data_nu.shape[0]
                )
            )
            knn = self.data_nu.shape[0]
        if knn_max is None:
            knn_max = self.data_nu.shape[0]

        Y = self._check_extension_shape(Y)
        if self.decay is None or self.thresh == 1:
            with _logger.task("KNN search"):
                # binary connectivity matrix
                K = self.knn_tree.kneighbors_graph(
                    Y, n_neighbors=knn, mode="connectivity"
                )
        else:
            with _logger.task("KNN search"):
                # sparse fast alpha decay
                knn_tree = self.knn_tree
                search_knn = min(knn * self.search_multiplier, knn_max)
                distances, indices = knn_tree.kneighbors(Y, n_neighbors=search_knn)
                self._check_duplicates(distances, indices)
            with _logger.task("affinities"):
                if bandwidth is None:
                    bandwidth = distances[:, knn - 1]

                bandwidth = bandwidth * bandwidth_scale

                # check for zero bandwidth
                bandwidth = np.maximum(bandwidth, np.finfo(float).eps)

                radius = bandwidth * np.power(-1 * np.log(self.thresh), 1 / self.decay)
                update_idx = np.argwhere(np.max(distances, axis=1) < radius).reshape(-1)
                _logger.debug(
                    "search_knn = {}; {} remaining".format(search_knn, len(update_idx))
                )
                if len(update_idx) > 0:
                    distances = [d for d in distances]
                    indices = [i for i in indices]
                # increase the knn search
                search_knn = min(search_knn * self.search_multiplier, knn_max)
                while (
                    len(update_idx) > Y.shape[0] // 10
                    and search_knn < self.data_nu.shape[0] / 2
                    and search_knn < knn_max
                ):
                    dist_new, ind_new = knn_tree.kneighbors(
                        Y[update_idx], n_neighbors=search_knn
                    )
                    for i, idx in enumerate(update_idx):
                        distances[idx] = dist_new[i]
                        indices[idx] = ind_new[i]
                    update_idx = [
                        i
                        for i, d in enumerate(distances)
                        if np.max(d)
                        < (
                            radius
                            if isinstance(bandwidth, numbers.Number)
                            else radius[i]
                        )
                    ]
                    _logger.debug(
                        "search_knn = {}; {} remaining".format(
                            search_knn, len(update_idx)
                        )
                    )
                    # increase the knn search
                    search_knn = min(search_knn * self.search_multiplier, knn_max)
                if search_knn > self.data_nu.shape[0] / 2:
                    knn_tree = NearestNeighbors(
                        search_knn, algorithm="brute", n_jobs=self.n_jobs
                    ).fit(self.data_nu)
                if len(update_idx) > 0:
                    if search_knn == knn_max:
                        _logger.debug(
                            "knn search to knn_max ({}) on {}".format(
                                knn_max, len(update_idx)
                            )
                        )
                        # give up - search out to knn_max
                        dist_new, ind_new = knn_tree.kneighbors(
                            Y[update_idx], n_neighbors=search_knn
                        )
                        for i, idx in enumerate(update_idx):
                            distances[idx] = dist_new[i]
                            indices[idx] = ind_new[i]
                    else:
                        _logger.debug("radius search on {}".format(len(update_idx)))
                        # give up - radius search
                        dist_new, ind_new = knn_tree.radius_neighbors(
                            Y[update_idx, :],
                            radius=radius
                            if isinstance(bandwidth, numbers.Number)
                            else np.max(radius[update_idx]),
                        )
                        for i, idx in enumerate(update_idx):
                            distances[idx] = dist_new[i]
                            indices[idx] = ind_new[i]
                if isinstance(bandwidth, numbers.Number):
                    data = np.concatenate(distances) / bandwidth
                else:
                    data = np.concatenate(
                        [distances[i] / bandwidth[i] for i in range(len(distances))]
                    )

                indices = np.concatenate(indices)
                indptr = np.concatenate([[0], np.cumsum([len(d) for d in distances])])
                K = sparse.csr_matrix(
                    (data, indices, indptr), shape=(Y.shape[0], self.data_nu.shape[0])
                )
                K.data = np.exp(-1 * np.power(K.data, self.decay))
                # handle nan
                K.data = np.where(np.isnan(K.data), 1, K.data)
                # TODO: should we zero values that are below thresh?
                K.data[K.data < self.thresh] = 0
                K = K.tocoo()
                K.eliminate_zeros()
                K = K.tocsr()
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

    clusters : array-like, shape=[n_samples]
        Private attribute. Cluster assignments for each sample.

    Examples
    --------
    >>> G = graphtools.Graph(data, n_landmark=1000)
    >>> X_landmark = transform(G.landmark_op)
    >>> X_full = G.interpolate(X_landmark)
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
                "kNNGraph instead".format(n_landmark, data.shape[0])
            )
        if n_svd >= data.shape[0]:
            warnings.warn(
                "n_svd ({}) >= n_samples ({}) Consider "
                "using kNNGraph or lower n_svd".format(n_svd, data.shape[0]),
                RuntimeWarning,
            )
        self.n_landmark = n_landmark
        self.n_svd = n_svd
        super().__init__(data, **kwargs)

    def get_params(self):
        """Get parameters from this object
        """
        params = super().get_params()
        params.update({"n_landmark": self.n_landmark, "n_pca": self.n_pca})
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
        if "n_landmark" in params and params["n_landmark"] != self.n_landmark:
            self.n_landmark = params["n_landmark"]
            reset_landmarks = True
        if "n_svd" in params and params["n_svd"] != self.n_svd:
            self.n_svd = params["n_svd"]
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
    def clusters(self):
        """Cluster assignments for each sample.

        Compute or return the cluster assignments

        Returns
        -------
        clusters : list-like, shape=[n_samples]
            Cluster assignments for each sample.
        """
        try:
            return self._clusters
        except AttributeError:
            self.build_landmark_op()
            return self._clusters

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
        landmarks = np.unique(self.clusters)
        if sparse.issparse(self.kernel):
            pmn = sparse.vstack(
                [
                    sparse.csr_matrix(self.kernel[self.clusters == i, :].sum(axis=0))
                    for i in landmarks
                ]
            )
        else:
            pmn = np.array(
                [np.sum(self.kernel[self.clusters == i, :], axis=0) for i in landmarks]
            )
        return pmn

    def _data_transitions(self):
        return normalize(self._landmarks_to_data(), "l1", axis=1)

    def build_landmark_op(self):
        """Build the landmark operator

        Calculates spectral clusters on the kernel, and calculates transition
        probabilities between cluster centers by using transition probabilities
        between samples assigned to each cluster.
        """
        with _logger.task("landmark operator"):
            is_sparse = sparse.issparse(self.kernel)
            # spectral clustering
            with _logger.task("SVD"):
                _, _, VT = randomized_svd(
                    self.diff_aff,
                    n_components=self.n_svd,
                    random_state=self.random_state,
                )
            with _logger.task("KMeans"):
                kmeans = MiniBatchKMeans(
                    self.n_landmark,
                    init_size=3 * self.n_landmark,
                    batch_size=10000,
                    random_state=self.random_state,
                )
                self._clusters = kmeans.fit_predict(self.diff_op.dot(VT.T))

            # transition matrices
            pmn = self._landmarks_to_data()

            # row normalize
            pnm = pmn.transpose()
            pmn = normalize(pmn, norm="l1", axis=1)
            pnm = normalize(pnm, norm="l1", axis=1)
            landmark_op = pmn.dot(pnm)  # sparsity agnostic matrix multiplication
            if is_sparse:
                # no need to have a sparse landmark operator
                landmark_op = landmark_op.toarray()
            # store output
            self._landmark_op = landmark_op
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
                [
                    sparse.csr_matrix(kernel[:, self.clusters == i].sum(axis=1))
                    for i in np.unique(self.clusters)
                ]
            )
        else:
            pnm = np.array(
                [
                    np.sum(kernel[:, self.clusters == i], axis=1).T
                    for i in np.unique(self.clusters)
                ]
            ).transpose()
        pnm = normalize(pnm, norm="l1", axis=1)
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

    decay : `int` or `None`, optional (default: 40)
        Rate of alpha decay to use. If `None`, alpha decay is not used.

    bandwidth : `float`, list-like,`callable`, or `None`, optional (default: `None`)
        Fixed bandwidth to use. If given, overrides `knn`. Can be a single
        bandwidth, list-like (shape=[n_samples]) of bandwidths for each
        sample, or a `callable` that takes in a `n x m` matrix and returns a
        a single value or list-like of length n (shape=[n_samples])

    bandwidth_scale : `float`, optional (default : 1.0)
        Rescaling factor for bandwidth.

    distance : `str`, optional (default: `'euclidean'`)
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph.
        TODO: actually sklearn.neighbors has even more choices

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

    def __init__(
        self,
        data,
        knn=5,
        decay=40,
        bandwidth=None,
        bandwidth_scale=1.0,
        distance="euclidean",
        n_pca=None,
        thresh=1e-4,
        precomputed=None,
        **kwargs
    ):
        if decay is None and precomputed not in ["affinity", "adjacency"]:
            # decay high enough is basically a binary kernel
            raise ValueError(
                "`decay` must be provided for a "
                "TraditionalGraph. For kNN kernel, use kNNGraph."
            )
        if precomputed is not None and n_pca not in [None, 0, False]:
            # the data itself is a matrix of distances / affinities
            n_pca = None
            warnings.warn(
                "n_pca cannot be given on a precomputed graph." " Setting n_pca=None",
                RuntimeWarning,
            )
        if knn is None and bandwidth is None:
            raise ValueError("Either `knn` or `bandwidth` must be provided.")
        if knn is not None and knn > data.shape[0] - 2:
            warnings.warn(
                "Cannot set knn ({k}) to be greater than "
                " n_samples - 2 ({n}). Setting knn={n}".format(
                    k=knn, n=data.shape[0] - 2
                )
            )
            knn = data.shape[0] - 2
        if precomputed is not None:
            if precomputed not in ["distance", "affinity", "adjacency"]:
                raise ValueError(
                    "Precomputed value {} not recognized. "
                    "Choose from ['distance', 'affinity', "
                    "'adjacency']"
                )
            elif data.shape[0] != data.shape[1]:
                raise ValueError(
                    "Precomputed {} must be a square matrix. "
                    "{} was given".format(precomputed, data.shape)
                )
            elif (data < 0).sum() > 0:
                raise ValueError(
                    "Precomputed {} should be " "non-negative".format(precomputed)
                )
        self.knn = knn
        self.decay = decay
        self.bandwidth = bandwidth
        self.bandwidth_scale = bandwidth_scale
        self.distance = distance
        self.thresh = thresh
        self.precomputed = precomputed

        super().__init__(data, n_pca=n_pca, **kwargs)

    def get_params(self):
        """Get parameters from this object
        """
        params = super().get_params()
        params.update(
            {
                "knn": self.knn,
                "decay": self.decay,
                "bandwidth": self.bandwidth,
                "bandwidth_scale": self.bandwidth_scale,
                "distance": self.distance,
                "precomputed": self.precomputed,
            }
        )
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
        - bandwidth
        - bandwidth_scale

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        if "precomputed" in params and params["precomputed"] != self.precomputed:
            raise ValueError("Cannot update precomputed. " "Please create a new graph")
        if (
            "distance" in params
            and params["distance"] != self.distance
            and self.precomputed is None
        ):
            raise ValueError("Cannot update distance. " "Please create a new graph")
        if "knn" in params and params["knn"] != self.knn and self.precomputed is None:
            raise ValueError("Cannot update knn. Please create a new graph")
        if (
            "decay" in params
            and params["decay"] != self.decay
            and self.precomputed is None
        ):
            raise ValueError("Cannot update decay. Please create a new graph")
        if (
            "bandwidth" in params
            and params["bandwidth"] != self.bandwidth
            and self.precomputed is None
        ):
            raise ValueError("Cannot update bandwidth. Please create a new graph")
        if (
            "bandwidth_scale" in params
            and params["bandwidth_scale"] != self.bandwidth_scale
        ):
            raise ValueError("Cannot update bandwidth_scale. Please create a new graph")
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
            if sparse.issparse(K) and not (
                isinstance(K, sparse.dok_matrix) or isinstance(K, sparse.lil_matrix)
            ):
                K = K.tolil()
            K = utils.set_diagonal(K, 1)
        else:
            with _logger.task("affinities"):
                if sparse.issparse(self.data_nu):
                    self.data_nu = self.data_nu.toarray()
                if self.precomputed == "distance":
                    pdx = self.data_nu
                elif self.precomputed is None:
                    pdx = pdist(self.data_nu, metric=self.distance)
                    if np.any(pdx == 0):
                        pdx = squareform(pdx)
                        duplicate_ids = np.array(
                            [i for i in np.argwhere(pdx == 0) if i[1] > i[0]]
                        )
                        duplicate_names = ", ".join(
                            ["{} and {}".format(i[0], i[1]) for i in duplicate_ids]
                        )
                        warnings.warn(
                            "Detected zero distance between samples {}. "
                            "Consider removing duplicates to avoid errors in "
                            "downstream processing.".format(duplicate_names),
                            RuntimeWarning,
                        )
                    else:
                        pdx = squareform(pdx)
                else:
                    raise ValueError(
                        "precomputed='{}' not recognized. "
                        "Choose from ['affinity', 'adjacency', 'distance', "
                        "None]".format(self.precomputed)
                    )
                if self.bandwidth is None:
                    knn_dist = np.partition(pdx, self.knn + 1, axis=1)[
                        :, : self.knn + 1
                    ]
                    bandwidth = np.max(knn_dist, axis=1)
                elif callable(self.bandwidth):
                    bandwidth = self.bandwidth(pdx)
                else:
                    bandwidth = self.bandwidth
                bandwidth = bandwidth * self.bandwidth_scale
                pdx = (pdx.T / bandwidth).T
                K = np.exp(-1 * np.power(pdx, self.decay))
                # handle nan
                K = np.where(np.isnan(K), 1, K)
        # truncate
        if sparse.issparse(K):
            if not (
                isinstance(K, sparse.csr_matrix)
                or isinstance(K, sparse.csc_matrix)
                or isinstance(K, sparse.bsr_matrix)
            ):
                K = K.tocsr()
            K.data[K.data < self.thresh] = 0
            K = K.tocoo()
            K.eliminate_zeros()
            K = K.tocsr()
        else:
            K[K < self.thresh] = 0
        return K

    def build_kernel_to_data(self, Y, knn=None, bandwidth=None, bandwidth_scale=None):
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
        if bandwidth is None:
            bandwidth = self.bandwidth
        if bandwidth_scale is None:
            bandwidth_scale = self.bandwidth_scale
        if self.precomputed is not None:
            raise ValueError("Cannot extend kernel on precomputed graph")
        else:
            with _logger.task("affinities"):
                Y = self._check_extension_shape(Y)
                pdx = cdist(Y, self.data_nu, metric=self.distance)
                if bandwidth is None:
                    knn_dist = np.partition(pdx, knn, axis=1)[:, :knn]
                    bandwidth = np.max(knn_dist, axis=1)
                elif callable(bandwidth):
                    bandwidth = bandwidth(pdx)
                bandwidth = bandwidth_scale * bandwidth
                pdx = (pdx.T / bandwidth).T
                K = np.exp(-1 * pdx ** self.decay)
                # handle nan
                K = np.where(np.isnan(K), 1, K)
                K[K < self.thresh] = 0
        return K

    @property
    def weighted(self):
        if self.precomputed is not None:
            return not utils.nonzero_discrete(self.K, [0.5, 1])
        else:
            return super().weighted

    def _check_shortest_path_distance(self, distance):
        if self.precomputed is not None:
            if distance == "data":
                raise ValueError(
                    "Graph shortest path with data distance not "
                    "valid for precomputed graphs. For precomputed graphs, "
                    "use `distance='constant'` for unweighted graphs and "
                    "`distance='affinity'` for weighted graphs."
                )
        super()._check_shortest_path_distance(distance)

    def _default_shortest_path_distance(self):
        if self.precomputed is not None and not self.weighted:
            distance = "constant"
            _logger.info("Using constant distances.")
        else:
            distance = super()._default_shortest_path_distance()
        return distance


class MNNGraph(DataGraph):
    """Mutual nearest neighbors graph

    Performs batch correction by forcing connections between batches, but
    only when the connection is mutual (i.e. x is a neighbor of y _and_
    y is a neighbor of x).

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`,
        `scipy.sparse.spmatrix`,
        `pandas.DataFrame`, `pandas.SparseDataFrame`.

    sample_idx : array-like, shape=[n_samples]
        Batch index

    beta : `float`, optional (default: 1)
        Downweight between-batch affinities by beta

    adaptive_k : {'min', 'mean', 'sqrt', `None`} (default: None)
        Weights MNN kernel adaptively using the number of cells in
        each sample according to the selected method.

    Attributes
    ----------
    subgraphs : list of `graphtools.graphs.kNNGraph`
        Graphs representing each batch separately
    """

    def __init__(
        self,
        data,
        sample_idx,
        knn=5,
        beta=1,
        n_pca=None,
        decay=None,
        adaptive_k=None,
        bandwidth=None,
        distance="euclidean",
        thresh=1e-4,
        n_jobs=1,
        **kwargs
    ):
        self.beta = beta
        self.sample_idx = sample_idx
        self.samples, self.n_cells = np.unique(self.sample_idx, return_counts=True)
        self.knn = knn
        self.decay = decay
        self.distance = distance
        self.bandwidth = bandwidth
        self.thresh = thresh
        self.n_jobs = n_jobs

        if sample_idx is None:
            raise ValueError(
                "sample_idx must be given. For a graph without"
                " batch correction, use kNNGraph."
            )
        elif len(sample_idx) != data.shape[0]:
            raise ValueError(
                "sample_idx ({}) must be the same length as "
                "data ({})".format(len(sample_idx), data.shape[0])
            )
        elif len(self.samples) == 1:
            raise ValueError("sample_idx must contain more than one unique value")
        if adaptive_k is not None:
            warnings.warn(
                "`adaptive_k` has been deprecated. Using fixed knn.", DeprecationWarning
            )

        super().__init__(data, n_pca=n_pca, **kwargs)

    def _check_symmetrization(self, kernel_symm, theta):
        if (
            (kernel_symm == "theta" or kernel_symm == "mnn")
            and theta is not None
            and not isinstance(theta, numbers.Number)
        ):
            raise TypeError(
                "Expected `theta` as a float. " "Got {}.".format(type(theta))
            )
        else:
            super()._check_symmetrization(kernel_symm, theta)

    def get_params(self):
        """Get parameters from this object
        """
        params = super().get_params()
        params.update(
            {
                "beta": self.beta,
                "knn": self.knn,
                "decay": self.decay,
                "bandwidth": self.bandwidth,
                "distance": self.distance,
                "thresh": self.thresh,
                "n_jobs": self.n_jobs,
            }
        )
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
        if "beta" in params and params["beta"] != self.beta:
            raise ValueError("Cannot update beta. Please create a new graph")

        # knn arguments
        knn_kernel_args = ["knn", "decay", "distance", "thresh", "bandwidth"]
        knn_other_args = ["n_jobs", "random_state", "verbose"]
        for arg in knn_kernel_args:
            if arg in params and params[arg] != getattr(self, arg):
                raise ValueError(
                    "Cannot update {}. " "Please create a new graph".format(arg)
                )
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
        with _logger.task("subgraphs"):
            self.subgraphs = []
            from .api import Graph

            # iterate through sample ids
            for i, idx in enumerate(self.samples):
                _logger.debug(
                    "subgraph {}: sample {}, "
                    "n = {}, knn = {}".format(
                        i, idx, np.sum(self.sample_idx == idx), self.knn
                    )
                )
                # select data for sample
                data = self.data_nu[self.sample_idx == idx]
                # build a kNN graph for cells within sample
                graph = Graph(
                    data,
                    n_pca=None,
                    knn=self.knn,
                    decay=self.decay,
                    bandwidth=self.bandwidth,
                    distance=self.distance,
                    thresh=self.thresh,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    kernel_symm="+",
                    initialize=True,
                )
                self.subgraphs.append(graph)  # append to list of subgraphs

        with _logger.task("MNN kernel"):
            if self.thresh > 0 or self.decay is None:
                K = sparse.lil_matrix((self.data_nu.shape[0], self.data_nu.shape[0]))
            else:
                K = np.zeros([self.data_nu.shape[0], self.data_nu.shape[0]])
            for i, X in enumerate(self.subgraphs):
                K = utils.set_submatrix(
                    K,
                    self.sample_idx == self.samples[i],
                    self.sample_idx == self.samples[i],
                    X.K,
                )
                within_batch_norm = np.array(np.sum(X.K, 1)).flatten()
                for j, Y in enumerate(self.subgraphs):
                    if i == j:
                        continue
                    with _logger.task(
                        "kernel from sample {} to {}".format(
                            self.samples[i], self.samples[j]
                        )
                    ):
                        Kij = Y.build_kernel_to_data(X.data_nu, knn=self.knn)
                        between_batch_norm = np.array(np.sum(Kij, 1)).flatten()
                        scale = (
                            np.minimum(1, within_batch_norm / between_batch_norm)
                            * self.beta
                        )
                        if sparse.issparse(Kij):
                            Kij = Kij.multiply(scale[:, None])
                        else:
                            Kij = Kij * scale[:, None]
                        K = utils.set_submatrix(
                            K,
                            self.sample_idx == self.samples[i],
                            self.sample_idx == self.samples[j],
                            Kij,
                        )
        return K

    def build_kernel_to_data(self, Y, theta=None):
        """Build transition matrix from new data to the graph

        Creates a transition matrix such that `Y` can be approximated by
        a linear combination of landmarks. Any
        transformation of the landmarks can be trivially applied to `Y` by
        performing

        `transform_Y = transitions.dot(transform)`

        Parameters
        ----------

        Y : array-like, [n_samples_y, n_features]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        theta : array-like or `None`, optional (default: `None`)
            if `self.theta` is a matrix, theta values must be explicitly
            specified between `Y` and each sample in `self.data`

        Returns
        -------

        transitions : array-like, [n_samples_y, self.data.shape[0]]
            Transition matrix from `Y` to `self.data`
        """
        raise NotImplementedError


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
