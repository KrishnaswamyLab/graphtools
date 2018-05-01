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
import time


class Data(object):  # parent class than handles PCA / import of data

    def __init__(self, data, n_pca=None, random_state=None):
        self.data = data
        self.n_pca = n_pca
        self.data_nu = self._reduce_data()
        super().__init__()

    def _reduce_data(self):
        if self.n_pca is not None and self.n_pca < self.data.shape[1]:
            if sparse.issparse(self.data):
                _, _, VT = randomized_svd(self.data, self.n_pca,
                                          random_state=self.random_state)
                V = VT.T
                self._right_singular_vectors = V
                return self.data.dot(V)
            else:
                self.pca = PCA(self.n_pca,
                               svd_solver='randomized',
                               random_state=self.random_state)
                return self.pca.fit_transform(self.data)
        else:
            return self.data

    def get_params(self):
        return {'n_pca': self.n_pca,
                'random_state': self.random_state}

    def set_params(self, **params):
        if 'n_pca' in params and params['n_pca'] != self.n_pca:
            raise ValueError("Cannot update n_pca. Please create a new graph")
        if 'random_state' in params:
            self.random_state = params['random_state']
        return self

    @property
    def U(self):
        try:
            return self.pca.components_
        except AttributeError:
            return None

    @property
    def S(self):
        try:
            return self.pca.singular_values_
        except AttributeError:
            return None

    @property
    def V(self):
        try:
            return self._right_singular_vectors
        except AttributeError:
            return None

    def transform(self, data):
        try:
            return self.pca.transform(data)
        except AttributeError:
            return data.dot(self._right_singular_vectors)
        except AttributeError:
            return data


# all graphs should possess these matrices
class BaseGraph(pygsp.graphs.Graph, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        self._kernel = self._build_kernel()
        W = self._build_weight_from_kernel()
        super().__init__(W, **kwargs)

    def _build_kernel(self):
        kernel = self.build_kernel()
        if (kernel - kernel.T).max() > 1e-5:
            raise RuntimeWarning("K should be symmetric")
        return kernel

    def _build_weight_from_kernel(self):
        weight = self._kernel.copy()
        if sparse.issparse(weight):
            weight.setdiag(0)
        else:
            np.fill_diagonal(weight, 0)
        return weight

    def get_params(self):
        return {}

    def set_params(self, **params):
        return self

    @property
    def P(self):
        try:
            return self._diff_op
        except AttributeError:
            self._diff_op = normalize(self.kernel, 'l1', axis=1)
            return self._diff_op

    @property
    def diff_op(self):
        return self.P

    @property
    def K(self):
        return self._kernel

    @property
    def kernel(self):
        return self.K

    @abc.abstractmethod
    def build_kernel(self):
        """Build the kernel matrix

        Must return a symmetric matrix
        """
        raise NotImplementedError
        K = K + K.T
        return K


class DataGraph(BaseGraph, Data, metaclass=abc.ABCMeta):

    def __init__(self, data, n_pca=None, random_state=None,
                 verbose=True, n_jobs=1, **kwargs):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        Data.__init__(self, data, n_pca=n_pca,
                      random_state=random_state)
        BaseGraph.__init__(self)

    def get_params(self):
        params = Data.get_params(self)
        params.update(BaseGraph.get_params(self))
        return params

    @abc.abstractmethod
    def build_kernel_to_data(self, Y):
        """Build a kernel from new input data Y to the Graph data

        Parameters
        ----------

        Y : array-like, [n_samples_y, n_dimensions]
            new data for which an affinity matrix is calculated
            to the existing data. n_dimensions must match
            either the ambient or PCA dimensions

        Returns
        -------

        K_yx : array-like, [n_samples_y, n_samples]

        Raises
        ------

        ValueError : if this Graph is not capable of extension
        """
        raise NotImplementedError

    def _check_extension_shape(self, Y):
        if not Y.shape[1] == self.data_nu.shape[1]:
            if Y.shape[1] == self.data.shape[1]:
                Y = self.transform(Y)
            else:
                if self.data.shape[1] != self.data.shape[1]:
                    msg = "Y must be of shape either (n, {}) or (n, {})".format(
                        self.data.shape[1], self.data_nu.shape[1])
                else:
                    msg = "Y must be of shape (n, {})".format(
                        self.data.shape[1])
                raise ValueError(msg)
        return Y

    def extend_to_data(self, Y):
        Y = self._check_extension_shape(Y)
        kernel = self.build_kernel_to_data(Y)
        transitions = normalize(kernel, norm='l1', axis=1)
        return transitions

    def interpolate(self, data, transitions):
        return transitions.dot(data)


class kNNGraph(DataGraph):  # build a kNN graph

    def __init__(self, data, knn=5, decay=None,
                 distance='euclidean',
                 thresh=1e-5, **kwargs):
        self.knn = knn
        self.decay = decay
        self.distance = distance
        self.thresh = thresh

        super().__init__(data, **kwargs)

    def get_params(self):
        params = super().get_params(self)
        params.update({'knn': self.knn,
                       'decay': self.decay,
                       'distance': self.distance,
                       'thresh': self.thresh,
                       'n_jobs': self.n_jobs,
                       'random_state': self.random_state,
                       'verbose': self.verbose})
        return params

    def set_params(self, **params):
        if 'knn' in params and params['knn'] != self.knn:
            raise ValueError("Cannot update knn. Please create a new graph")
        if 'decay' in params and params['decay'] != self.decay:
            raise ValueError("Cannot update decay. Please create a new graph")
        if 'distance' in params and params['distance'] != self.distance:
            raise ValueError("Cannot update distance. "
                             "Please create a new graph")
        if 'thresh' in params and params['thresh'] != self.thresh and self.decay != 0:
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
        try:
            return self._knn_tree
        except AttributeError:
            self._knn_tree = NearestNeighbors(
                n_neighbors=self.knn,
                metric=self.distance,
                n_jobs=self.n_jobs).fit(self.data_nu)
            return self._knn_tree

    def build_kernel(self):
        if self.decay is None or self.thresh == 1:
            K = kneighbors_graph(self.knn_tree,
                                 n_neighbors=self.knn,
                                 metric=self.distance,
                                 mode='connectivity',
                                 include_self=True)
        elif self.thresh == 0:
            pdx = squareform(pdist(self.data, metric=self.distance))
            knn_dist = np.partition(pdx, self.knn, axis=1)[:, :self.knn]
            epsilon = np.max(knn_dist, axis=1)
            pdx = (pdx / epsilon).T
            K = np.exp(-1 * pdx**self.decay)
        else:
            radius, _ = self.knn_tree.kneighbors(self.data_nu)
            bandwidth = radius[:, -1]
            radius = bandwidth * np.power(-1 * np.log(self.thresh),
                                          1 / self.decay)
            distances = np.empty(shape=self.data_nu.shape[0], dtype=np.object)
            for i in range(self.data_nu.shape[0]):
                # this could be parallelized
                row_distances = self.knn_tree.radius_neighbors_graph(
                    self.data_nu[i, None, :],
                    radius[i],
                    mode='distance')
                row_distances.data = np.exp(
                    -1 * ((row_distances.data / bandwidth[i]) ** self.decay))
                distances[i] = row_distances
            K = sparse.vstack(distances)
        K = K + K.T
        return K

    def build_kernel_to_data(self, Y):
        Y = self._check_extension_shape(Y)
        if self.decay is None or self.thresh == 1:
            K = self.knn_tree.kneighbors_graph(
                Y, n_neighbors=self.knn,
                mode='connectivity')
        elif self.thresh == 0:
            pdx = cdist(Y, self.data, metric=self.distance)
            knn_dist = np.partition(pdx, self.knn, axis=1)[:, :self.knn]
            epsilon = np.max(knn_dist, axis=1)
            pdx = (pdx / epsilon).T
            K = np.exp(-1 * pdx**self.decay)
        else:
            radius, _ = self.knn_tree.kneighbors(Y)
            bandwidth = radius[:, -1]
            radius = bandwidth * np.power(-1 * np.log(self.thresh),
                                          1 / self.decay)
            distances = np.empty(shape=Y.shape[0], dtype=np.object)
            for i in range(Y.shape[0]):
                # this could be parallelized
                row_distances = self.knn_tree.radius_neighbors_graph(
                    Y[i, None, :],
                    radius[i],
                    mode='distance')
                row_distances.data = np.exp(
                    -1 * ((row_distances.data / bandwidth[i]) ** self.decay))
                distances[i] = row_distances
            K = sparse.vstack(distances)
        return K


class LandmarkGraph(DataGraph):

    def __init__(self, data, n_landmark=2000, n_svd=100, **kwargs):
        if n_landmark >= data.shape[0]:
            raise RuntimeWarning(
                "n_landmark ({}) >= n_samples ({}). Consider "
                "using kNNGraph instead".format(n_landmark, data.shape[0]))
        if n_svd >= data.shape[0]:
            raise RuntimeWarning(
                "n_svd ({}) >= n_samples ({}) Consider "
                "using lower n_svd".format(n_svd, data.shape[0]))
        self.n_landmark = n_landmark
        self.n_svd = n_svd
        super().__init__(data, **kwargs)

    def get_params(self):
        params = super().get_params(self)
        params.update({'n_landmark': self.n_landmark,
                       'n_pca': self.n_pca})
        return params

    def set_params(self, **params):
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
        try:
            del self._landmark_op
            del self._transitions
            del self._clusters
        except AttributeError:
            pass

    @property
    def landmark_op(self):
        try:
            return self._landmark_op
        except AttributeError:
            self.build_landmark_op()
            return self._landmark_op

    @property
    def transitions(self):
        try:
            return self._transitions
        except AttributeError:
            self.build_landmark_op()
            return self._transitions

    def build_landmark_op(self):
        is_sparse = sparse.issparse(self.kernel)
        # spectral clustering
        if self.verbose:
            print("Calculating SVD...")
            start = time.time()
        _, _, VT = randomized_svd(self.diff_op,
                                  n_components=self.n_svd,
                                  random_state=self.random_state)
        if self.verbose:
            print("SVD complete in {:.2f} seconds".format(
                time.time() - start))
            start = time.time()
            print("Calculating Kmeans...")
        kmeans = MiniBatchKMeans(
            self.n_landmark,
            init_size=3 * self.n_landmark,
            batch_size=10000,
            random_state=self.random_state)
        self._clusters = kmeans.fit_predict(
            self.diff_op.dot(VT.T))
        landmarks = np.unique(self._clusters)
        if self.verbose:
            print("Kmeans complete in {:.2f} seconds".format(
                time.time() - start))

        # transition matrices
        if is_sparse:
            pmn = sparse.vstack(
                [sparse.csr_matrix(self.kernel[self._clusters == i, :].sum(
                    axis=0)) for i in landmarks])
        else:
            pmn = np.array([np.sum(
                self.kernel[self._clusters == i, :], axis=0) for i in landmarks])
        # row normalize
        pnm = pmn.transpose()
        pmn = normalize(pmn, norm='l1', axis=1)
        pnm = normalize(pnm, norm='l1', axis=1)
        diff_op = pmn.dot(pnm)  # sparsity agnostic matrix multiplication
        if is_sparse:
            diff_op = diff_op.todense()
        self._landmark_op = np.array(diff_op)
        self._transitions = pnm

    def extend_to_data(self, data):
        kernel = self.build_kernel_to_data(data)
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

    def interpolate(self, data, transitions=None):
        if transitions is None:
            transitions = self.transitions
        return super().interpolate(data, transitions)


class TraditionalGraph(DataGraph):

    def __init__(self, data, knn=5, decay=10,
                 distance='euclidean', n_pca=None,
                 precomputed=None, **kwargs):
        if precomputed is not None:
            # the data itself is a matrix of distances / affinities
            n_pca = None
            print("Warning: n_pca cannot be given on a precomputed graph."
                  "Setting n_pca=None")
        self.knn = knn
        self.decay = decay
        self.distance = distance
        self.precomputed = precomputed

        super().__init__(data, n_pca=n_pca,
                         **kwargs)

    def get_params(self):
        params = super().get_params(self)
        params.update({'knn': self.knn,
                       'decay': self.decay,
                       'distance': self.distance,
                       'precomputed': self.precomputed})
        return params

    def set_params(self, **params):
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
        if self.precomputed is not None:
            if self.precomputed not in ["distance", "affinity", "adjacency"]:
                raise ValueError("Precomputed value {} not recognized. "
                                 "Choose from ['distance', 'affinity', 'adjacency']")
        if self.precomputed is "affinity":
            K = self.data_nu
        elif self.precomputed is "adjacency":
            K = self.data_nu
            if sparse.issparse(K):
                K.setdiag(1)
            else:
                np.fill_diagonal(K, 1)
        else:
            if self.precomputed is "distance":
                pdx = self.data_nu
            elif self.precomputed is None:
                pdx = squareform(pdist(self.data, metric=self.distance))
            knn_dist = np.partition(pdx, self.knn, axis=1)[:, :self.knn]
            epsilon = np.max(knn_dist, axis=1)
            pdx = (pdx / epsilon).T
            K = np.exp(-1 * pdx**self.decay)
        K = K + K.T
        return K

    def build_kernel_to_data(self, Y):
        if self.precomputed is not None:
            raise ValueError("Cannot extend kernel on precomputed graph")
        else:
            Y = self._check_extension_shape(Y)
            pdx = cdist(Y, self.data, metric=self.distance)
            knn_dist = np.partition(pdx, self.knn, axis=1)[:, :self.knn]
            epsilon = np.max(knn_dist, axis=1)
            pdx = (pdx / epsilon).T
            K = np.exp(-1 * pdx**self.decay)
        return K


class MNNGraph(DataGraph):

    def __init__(self, data, beta=0, gamma=0.5, n_pca=None,
                 sample_idx=None, **kwargs):
        self.beta = beta
        self.gamma = gamma
        self.sample_idx = sample_idx
        self.knn_args = kwargs

        super().__init__(data, n_pca=n_pca, **kwargs)

    def get_params(self):
        params = super().get_params(self)
        params.update({'beta': self.beta,
                       'gamma': self.gamma})
        params.update(self.knn_args)
        return params

    def set_params(self, **params):
        if 'beta' in params and params['beta'] != self.beta:
            raise ValueError("Cannot update beta. Please create a new graph")
        if 'gamma' in params and params['gamma'] != self.gamma:
            raise ValueError("Cannot update gamma. Please create a new graph")

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

        # update superclass parameters
        super().set_params(**params)
        return self

    def build_kernel(self):
        self.subgraphs = []
        for idx in np.unique(self.sample_idx):
            data = self.data_nu[self.sample_idx == idx]
            graph = kNNGraph(
                data, n_pca=None, **(self.knn_args))
            self.subgraphs.append(graph)
        kernels = []
        for i, X in enumerate(self.subgraphs):
            kernels.append([])
            for j, Y in enumerate(self.subgraphs):
                if i == j:
                    Kij = X.kernel
                    Kij = Kij * self.beta
                else:
                    Kij = X.build_kernel_to_data(Y.data_nu)
                kernels[-1].append(Kij)

        K = sparse.hstack([sparse.vstack(
            kernels[i]) for i in range(len(kernels))])
        K = self.gamma * K.minimum(K.T) + \
            (1 - self.gamma) * K.maximum(K.T)
        return K

    def build_kernel_to_data(self, Y):
        Y = self._check_extension_shape(Y)
        kernel_xy = []
        kernel_yx = []
        Y_graph = kNNGraph(
            Y, n_pca=None, **(self.knn_args))
        for i, X in enumerate(self.subgraphs):
            kernel_xy.append(X.build_kernel_to_data(Y))
            kernel_yx.append(Y_graph.build_kernel_to_data(X.data_nu))
        kernel_xy = sparse.hstack(kernel_xy)
        kernel_yx = sparse.vstack(kernel_yx)
        K = self.gamma * kernel_xy.minimum(kernel_yx.T) + \
            (1 - self.gamma) * kernel_xy.maximum(kernel_yx.T)
        return K


def Graph(data,
          n_pca=None,
          sample_idx=None,
          precomputed=None,
          knn=5,
          decay=None,
          distance='euclidean',
          thresh=1e-5,
          n_landmark=None,
          n_svd=100,
          beta=0,
          gamma=0.5,
          n_jobs=-1,
          verbose=False,
          random_state=None,
          graphtype='auto'):
    base = []
    if graphtype == 'auto':
        if sample_idx is not None:
            graphtype = "mnn"
        elif precomputed is None and (decay is None or thresh > 0):
            graphtype = "knn"
        else:
            graphtype = "exact"
    base = []
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
    if n_landmark is not None:
        class Graph(base, LandmarkGraph):

            def __init__(self, *args, **kwargs):
                base.__init__(self, *args, **kwargs)
                LandmarkGraph.__init__(self, *args, **kwargs)
    else:
        class Graph(base):
            pass

    return Graph(data,
                 n_pca=n_pca,
                 sample_idx=sample_idx,
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
                 random_state=random_state)
