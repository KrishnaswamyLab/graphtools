import numpy as np
import abc
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
from scipy import sparse


class Data(object):  # parent class than handles PCA / import of data

    def __init__(self, data, ndim=None, random_state=None):
        self.data = data
        if ndim is not None and ndim < data.shape[1]:
            pca = PCA(ndim, svd_solver='randomized', random_state=random_state)
            self.data_nu = pca.fit_transform(data)
            self.U = pca.components_
            self.S = pca.singular_values_
        else:
            self.data_nu = data


# all graphs should possess these matrices
class BaseGraph(object, metaclass=abc.ABCMeta):

    def __init__(self):
        self.lap_type = "combinatorial"

    @property
    def K(self):
        try:
            return self._K
        except AttributeError:
            self._K = self.build_K()
            if (self._K - self._K.T).max() > 1e-5:
                raise RuntimeWarning("K should be symmetric")
            return self._K

    @property
    def A(self):
        try:
            return self._A
        except AttributeError:
            self._A = np.fill_diagonal(self.K, 0)
            return self._A

    @property
    def D(self):
        try:
            return self._D
        except AttributeError:
            self._D = self.A.sum(axis=1)[:, None]
            return self._D

    @property
    def diffop(self):
        try:
            return self._diffop
        except AttributeError:
            self._diffop = normalize(self.K, 'l1', axis=1)
            return self._diffop

    @property
    def L(self, lap_type="combinatorial"):
        if not self._L or lap_type is not self.lap_type:
            if lap_type is "combinatorial":
                self._L = self.D - self.A
            elif lap_type is "normalized":
                self._L = np.eye(self.A.shape[0]) - np.diag(self.D) ^ - \
                    5 * self.A * np.diag(self.D) ^ -.5
            elif lap_type is "randomwalk":
                self._L = np.eye(self.A.shape[0]) - self.diffop
            self.lap_type = lap_type
        return self._L

    @abc.abstractmethod
    def build_K(self):
        """Build the kernel matrix

        Must return a symmetric matrix
        """
        raise NotImplementedError
        K = K + K.T
        return K


class kNNGraph(BaseGraph, Data):  # build a kNN graph

    def __init__(self, data, ndim=None, random_state=None,
                 knn=5, decay=0, distance='euclidean',
                 thresh=1e-5, n_jobs=-1):
        self.knn = knn
        self.decay = decay
        self.distance = distance
        self.thresh = thresh
        self.n_jobs = n_jobs

        Data.__init__(self, data, ndim=ndim,
                      random_state=random_state)
        BaseGraph.__init__(self)

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

    def build_K(self):
        if self.decay == 0:
            K = kneighbors_graph(self.knn_tree,
                                 n_neighbors=self.knn,
                                 metric=self.distance,
                                 mode='connectivity',
                                 include_self=True)
        else:
            knn = self.knn
            tmp = kneighbors_graph(self.knn_tree,
                                   n_neighbors=knn,
                                   metric=self.distance,
                                   mode='distance',
                                   include_self=False)
            bandwidth = sparse.diags(1 / np.max(tmp, 1).A.ravel())
            ktmp = np.exp(-1 * (tmp * bandwidth)**self.decay)
            while (np.min(ktmp[np.nonzero(ktmp)]) > self.thresh):
                knn += 5
                tmp = kneighbors_graph(self.knn_tree,
                                       n_neighbors=knn,
                                       metric=self.distance,
                                       mode='distance',
                                       include_self=False)
                ktmp = np.exp(-1 * (tmp * bandwidth)**self.decay)
            K = ktmp
        K = K + K.T
        return K

    def build_kernel_to_data(self, Y):

        if self.decay == 0:
            K = self.knn_tree.kneighbors_graph(
                Y, n_neighbors=self.knn,
                mode='connectivity')
        else:
            knn = self.knn
            tmp = self.knn_tree.kneighbors_graph(
                Y, n_neighbors=knn,
                mode='distance')
            bandwidth = sparse.diags(1 / np.max(tmp, 1).A.ravel())
            ktmp = np.exp(-1 * (tmp * bandwidth)**self.decay)
            while (np.min(ktmp[np.nonzero(ktmp)]) > self.thresh):
                knn += 5
                tmp = self.knn_tree.kneighbors_graph(
                    Y, n_neighbors=knn,
                    mode='distance')
                ktmp = np.exp(-1 * (tmp * bandwidth)**self.decay)
            K = ktmp

        return K


class TraditionalGraph(BaseGraph, Data):

    def __init__(self, data, ndim=None, random_state=None,
                 knn=5, decay=10,  distance='euclidean',
                 precomputed=None):
        if precomputed is None:
            # the data itself is a matrix of distances / affinities
            ndim = None
        self.knn = knn
        self.decay = decay
        self.distance = distance
        self.precomputed = precomputed

        Data.__init__(self, data, ndim=ndim,
                      random_state=random_state)
        BaseGraph.__init__(self)

    def build_K(self):
        if self.precomputed is not None:
            if self.precomputed not in ["distance", "affinity"]:
                raise ValueError("Precomputed value {} not recognised. "
                                 "Choose from ['distance', 'affinity']")
        if self.precomputed is "distance":
            pdx = self.data_nu
        if self.precomputed is None:
            pdx = squareform(pdist(self.data, metric=self.distance))
        if self.precomputed is not "affinity":
            knn_dist = np.partition(pdx, self.knn, axis=1)[:, :self.knn]
            epsilon = np.max(knn_dist, axis=1)
            pdx = (pdx / epsilon).T
            K = np.exp(-1 * pdx**self.decay)
        else:
            K = self.data_nu

        K = K + K.T
        return K


class MNNGraph(BaseGraph, Data):

    def __init__(self, data, ndim=None, random_state=None,
                 knn=5, decay=0, distance='euclidean',
                 thresh=1e-5, beta=0, gamma=0.5,
                 sample_idx=None):
        self.knn = knn
        self.decay = decay
        self.distance = distance
        self.thresh = thresh
        self.beta = beta
        self.gamma = gamma
        self.sample_idx = sample_idx

        Data.__init__(self, data, ndim=ndim,
                      random_state=random_state)
        BaseGraph.__init__(self)

    def build_K(self):
        graphs = []
        for idx in np.unique(self.sample_idx):
            data = self.data_nu[self.sample_idx == idx]
            graph = kNNGraph(
                data, ndim=None,
                knn=self.knn, decay=self.decay,
                distance=self.distance, thresh=self.thresh)
            graphs.append(graph)
        kernels = []
        for i, X in enumerate(graphs):
            kernels.append([])
            for j, Y in enumerate(graphs):
                Kij = X.build_kernel_to_data(Y.data_nu)
                if i == j:
                    Kij = Kij * self.beta
                kernels[-1].append(Kij)

        K = sparse.hstack([sparse.vstack(
            kernels[i]) for i in range(len(kernels))])
        K = self.gamma * K.minimum(K.T) + \
            (1 - self.gamma) * K.maximum(K.T)
        return K


def Graph(graphtype, data, *args, **kwargs):
    if graphtype == "knn":
        base = kNNGraph
    elif graphtype == "exact":
        base = TraditionalGraph

    class Graph(base):

        def __init__(self, data, *args, **kwargs):
            base.__init__(self, data, *args, **kwargs)

    return Graph(data, *args, **kwargs)
