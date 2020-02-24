import numpy as np
import tasklogger
import pygsp
import abc

from functools import partial
from scipy import sparse

from . import api, graphs, base, utils, matrix


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


_logger = tasklogger.get_tasklogger("graphtools")


class GraphEstimator(object, metaclass=abc.ABCMeta):
    """Estimator which builds a graphtools Graph

    Parameters
    ----------

    knn : int, optional, default: 5
        number of nearest neighbors on which to build kernel

    decay : int, optional, default: 40
        sets decay rate of kernel tails.
        If None, alpha decaying kernel is not used

    n_landmark : int, optional, default: None
        number of landmarks to use in graph construction

    n_pca : int, optional, default: 100
        Number of principal components to use for calculating
        neighborhoods. For extremely large datasets, using
        n_pca < 20 allows neighborhoods to be calculated in
        roughly log(n_samples) time.

    distance : string, optional, default: 'euclidean'
        recommended values: 'euclidean', 'cosine', 'precomputed'
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph. Custom distance
        functions of form `f(x, y) = d` are also accepted. If 'precomputed',
        `data` should be an n_samples x n_samples distance or
        affinity matrix. Distance matrices are assumed to have zeros
        down the diagonal, while affinity matrices are assumed to have
        non-zero values down the diagonal. This is detected automatically using
        `data[0,0]`. You can override this detection with
        `distance='precomputed_distance'` or `distance='precomputed_affinity'`.

    n_jobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used

    random_state : integer or numpy.RandomState, optional, default: None
        If an integer is given, it fixes the seed
        Defaults to the global `numpy` random number generator

    verbose : `int` or `boolean`, optional (default: 1)
        If `True` or `> 0`, print status messages
        
    n_svd : int, optional (default: 100)
        number of singular vectors to compute for landmarking
    
    thresh : float, optional (default: 1e-4)
        threshold below which to truncate kernel
    
    kwargs : additional arguments for graphtools.Graph
    
    Attributes
    ----------
    
    graph : graphtools.Graph
    """

    X = attribute("X", doc="Stored input data")
    graph = attribute("graph", doc="graphtools Graph object")

    @graph.setter
    def graph(self, G):
        self._graph = G
        if G is None:
            self._reset_graph()

    n_pca = attribute(
        "n_pca",
        default=100,
        on_set=partial(utils.check_if_not, None, utils.check_positive, utils.check_int),
    )
    random_state = attribute("random_state")

    knn = attribute("knn", default=5, on_set=[utils.check_positive, utils.check_int])
    decay = attribute("decay", default=40, on_set=utils.check_positive)
    distance = attribute(
        "distance",
        default="euclidean",
        on_set=partial(
            utils.check_in,
            [
                "euclidean",
                "precomputed",
                "cosine",
                "correlation",
                "cityblock",
                "l1",
                "l2",
                "manhattan",
                "braycurtis",
                "canberra",
                "chebyshev",
                "dice",
                "hamming",
                "jaccard",
                "kulsinski",
                "mahalanobis",
                "matching",
                "minkowski",
                "rogerstanimoto",
                "russellrao",
                "seuclidean",
                "sokalmichener",
                "sokalsneath",
                "sqeuclidean",
                "yule",
                "precomputed_affinity",
                "precomputed_distance",
            ],
        ),
    )
    n_svd = attribute(
        "n_svd",
        default=100,
        on_set=partial(utils.check_if_not, None, utils.check_positive, utils.check_int),
    )
    n_jobs = attribute(
        "n_jobs", on_set=partial(utils.check_if_not, None, utils.check_int)
    )
    verbose = attribute("verbose", default=0)
    thresh = attribute(
        "thresh",
        default=1e-4,
        on_set=partial(utils.check_if_not, 0, utils.check_positive),
    )

    n_landmark = attribute("n_landmark")

    @n_landmark.setter
    def n_landmark(self, n_landmark):
        self._n_landmark = n_landmark
        utils.check_if_not(
            None, utils.check_positive, utils.check_int, n_landmark=n_landmark
        )
        self._update_n_landmark(n_landmark)

    def _update_n_landmark(self, n_landmark):
        if self.graph is not None:
            n_landmark = self._parse_n_landmark(self.graph.data_nu, n_landmark)
            if (
                n_landmark is None and isinstance(self.graph, graphs.LandmarkGraph)
            ) or (
                n_landmark is not None
                and not isinstance(self.graph, graphs.LandmarkGraph)
            ):
                # new graph but the same kernel
                # there may be a better way to do this
                kernel = self.graph.kernel
                self.graph = None
                self.fit(self.X, initialize=False)
                self.graph._kernel = kernel

    def __init__(
        self,
        knn=5,
        decay=40,
        n_pca=100,
        n_landmark=None,
        random_state=None,
        distance="euclidean",
        n_svd=100,
        n_jobs=1,
        verbose=1,
        thresh=1e-4,
        **kwargs
    ):

        if verbose is True:
            verbose = 1
        elif verbose is False:
            verbose = 0

        self.n_pca = n_pca
        self.n_landmark = n_landmark
        self.random_state = random_state
        self.knn = knn
        self.decay = decay
        self.distance = distance
        self.n_svd = n_svd
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.thresh = thresh
        self.kwargs = kwargs
        self.logger = _logger
        _logger.set_level(self.verbose)

    def set_params(self, **params):
        for p in params:
            if not getattr(self, p) == params[p]:
                setattr(self, p, params[p])
        self._set_graph_params(**params)

    def _set_graph_params(self, **params):
        if self.graph is not None:
            try:
                if "n_pca" in params:
                    params["n_pca"] = self._parse_n_pca(
                        self.graph.data_nu, params["n_pca"]
                    )
                if "n_svd" in params:
                    params["n_svd"] = self._parse_n_svd(
                        self.graph.data_nu, params["n_svd"]
                    )
                if "n_landmark" in params:
                    params["n_landmark"] = self._parse_n_landmark(
                        self.graph.data_nu, params["n_landmark"]
                    )
                self.graph.set_params(**params)
            except ValueError as e:
                _logger.debug("Reset graph due to {}".format(str(e)))
                self.graph = None

    @abc.abstractmethod
    def _reset_graph(self):
        """Trigger a reset of self.graph
        
        Any downstream effects of resetting the graph should override this function
        """
        raise NotImplementedError

    def _detect_precomputed_matrix_type(self, X):
        if isinstance(X, (sparse.coo_matrix, sparse.dia_matrix)):
            X = X.tocsr()
        if X[0, 0] == 0:
            return "distance"
        else:
            return "affinity"

    @staticmethod
    def _parse_n_landmark(X, n_landmark):
        if n_landmark is not None and n_landmark >= X.shape[0]:
            return None
        else:
            return n_landmark

    @staticmethod
    def _parse_n_pca(X, n_pca):
        if n_pca is not None and n_pca >= min(X.shape):
            return None
        else:
            return n_pca

    @staticmethod
    def _parse_n_svd(X, n_svd):
        if n_svd is not None and n_svd >= X.shape[0]:
            return X.shape[0] - 1
        else:
            return n_svd

    def _parse_input(self, X):
        # passing graphs as input
        if isinstance(X, base.BaseGraph):
            # we can keep this graph
            self.graph = X
            X = X.data
            # immutable graph properties override operator
            n_pca = self.graph.n_pca
            self.knn = self.graph.knn
            self.decay = self.graph.decay
            self.distance = self.graph.distance
            self.thresh = self.graph.thresh
            update_graph = False
            if isinstance(self.graph, graphs.TraditionalGraph):
                precomputed = self.graph.precomputed
            else:
                precomputed = None
        elif isinstance(X, pygsp.graphs.Graph):
            # convert pygsp to graphtools
            self.graph = None
            X = X.W
            precomputed = "adjacency"
            update_graph = False
            n_pca = None
        else:
            # data matrix
            update_graph = True
            if utils.is_Anndata(X):
                X = X.X
            if not callable(self.distance) and self.distance.startswith("precomputed"):
                if self.distance == "precomputed":
                    # automatic detection
                    precomputed = self._detect_precomputed_matrix_type(X)
                elif self.distance in ["precomputed_affinity", "precomputed_distance"]:
                    precomputed = self.distance.split("_")[1]
                else:
                    raise NotImplementedError
                n_pca = None
            else:
                precomputed = None
                n_pca = self._parse_n_pca(X, self.n_pca)
        return (
            X,
            n_pca,
            self._parse_n_landmark(X, self.n_landmark),
            precomputed,
            update_graph,
        )

    def _update_graph(self, X, precomputed, n_pca, n_landmark, **kwargs):
        if self.X is not None and not matrix.matrix_is_equivalent(X, self.X):
            """
            If the same data is used, we can reuse existing kernel and
            diffusion matrices. Otherwise we have to recompute.
            """
            self.graph = None
        else:
            self._update_n_landmark(n_landmark)
            self._set_graph_params(
                n_pca=n_pca,
                precomputed=precomputed,
                n_landmark=n_landmark,
                random_state=self.random_state,
                knn=self.knn,
                decay=self.decay,
                distance=self.distance,
                n_svd=self._parse_n_svd(self.X, self.n_svd),
                n_jobs=self.n_jobs,
                thresh=self.thresh,
                verbose=self.verbose,
                **(self.kwargs)
            )
            if self.graph is not None:
                _logger.info("Using precomputed graph and diffusion operator...")

    def fit(self, X, **kwargs):
        """Computes the graph

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            input data with `n_samples` samples and `n_dimensions`
            dimensions. Accepted data types: `numpy.ndarray`,
            `scipy.sparse.spmatrix`, `pd.DataFrame`, `anndata.AnnData`. If
            `knn_dist` is 'precomputed', `data` should be a n_samples x
            n_samples distance or affinity matrix

        kwargs : additional arguments for graphtools.Graph

        Returns
        -------
        self : graphtools.estimator.GraphEstimator
        """
        X, n_pca, n_landmark, precomputed, update_graph = self._parse_input(X)

        if precomputed is None:
            _logger.info(
                "Building graph on {} samples and {} features.".format(
                    X.shape[0], X.shape[1]
                )
            )
        else:
            _logger.info(
                "Building graph on precomputed {} matrix with {} samples.".format(
                    precomputed, X.shape[0]
                )
            )

        if self.graph is not None and update_graph:
            self._update_graph(X, precomputed, n_pca, n_landmark)

        self.X = X

        if self.graph is None:
            with _logger.task("graph and diffusion operator"):
                self.graph = api.Graph(
                    X,
                    n_pca=n_pca,
                    precomputed=precomputed,
                    n_landmark=n_landmark,
                    random_state=self.random_state,
                    knn=self.knn,
                    decay=self.decay,
                    distance=self.distance,
                    n_svd=self._parse_n_svd(self.X, self.n_svd),
                    n_jobs=self.n_jobs,
                    thresh=self.thresh,
                    verbose=self.verbose,
                    **(self.kwargs),
                    **kwargs
                )
        return self
