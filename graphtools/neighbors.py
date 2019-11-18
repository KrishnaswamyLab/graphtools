import nmslib
import tasklogger
import numpy as np
import multiprocessing
from scipy import sparse
import scprep
from sklearn.exceptions import NotFittedError


class HNSW:

    _DENSE_TYPES = {"l2": "l2_sparse"}
    _SPARSE_TYPES = {"l2_sparse": "l2"}

    def __init__(self, n_neighbors, space="l2", data_type=None, n_jobs=1):
        n_jobs = int(n_jobs)
        if n_jobs <= 0:
            n_jobs = multiprocessing.cpu_count() + 1 + n_jobs
        tasklogger.log_debug(
            "Init HNSW index with arguments "
            "n_neighbors={}, space={}, data_type={}, n_jobs={}".format(
                n_neighbors, space, data_type, n_jobs
            )
        )
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.space = space
        self.data_type = data_type
        self._fitted = False

    def _to_dense_type(self, space):
        space = self._SPARSE_TYPES[space]
        if space is None:
            raise NotImplementedError
        return space

    def _to_sparse_type(self, space):
        space = self._DENSE_TYPES[space]
        if space is None:
            raise NotImplementedError
        return space

    def _check_data(self, X):
        if self.data_type == nmslib.DataType.SPARSE_VECTOR and not sparse.issparse(X):
            # convert to CSR matrix
            X = sparse.csr_matrix(scprep.utils.to_array_or_spmatrix(X))
        elif self.data_type == nmslib.DataType.DENSE_VECTOR and sparse.issparse(X):
            # convert to dense matrix
            X = scprep.utils.toarray(X)
        else:
            # convert to numpy or scipy matrix
            X = scprep.utils.to_array_or_spmatrix(X)
        if self.data_type is None:
            # set data_type from data
            if sparse.issparse(X):
                self.data_type = nmslib.DataType.SPARSE_VECTOR
            else:
                self.data_type = nmslib.DataType.DENSE_VECTOR
        if self.data_type == nmslib.DataType.SPARSE_VECTOR:
            # make sure sparse matrix is CSR format
            X = sparse.csr_matrix(X)
            # check space is compatible with sparse data
            if self.space in self._DENSE_TYPES:
                self.space = self._to_sparse_type(self.space)
        else:
            # check space is compatible with dense data
            if self.space in self._SPARSE_TYPES:
                self.space = self._to_dense_type(self.space)
        return X

    def fit(self, X, M=15, efConstruction=100, post=0):
        self.X = self._check_data(X)
        tasklogger.log_debug(
            "Building HNSW index with {} of shape {}".format(type(X), X.shape)
        )
        tasklogger.log_debug(
            "Arguments "
            "M={}, efConstruction={}, post={}, data_type={}".format(
                M, efConstruction, post, self.data_type
            )
        )
        self.index = nmslib.init(
            method="hnsw", space=self.space, data_type=self.data_type
        )
        self.index.addDataPointBatch(X)
        self.index.createIndex(
            {
                "M": M,
                "indexThreadQty": self.n_jobs,
                "efConstruction": efConstruction,
                "post": post,
            }
        )
        self._fitted = True
        return self

    def kneighbors(self, X=None, n_neighbors=None, efSearch=None, **kwargs):
        if X is None:
            if self._fitted:
                X = self.X
            else:
                raise NotFittedError
        elif not self._fitted:
            self.fit(X, **kwargs)
        else:
            X = self._check_data(X)
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if efSearch is None:
            efSearch = max(100, n_neighbors)
        tasklogger.log_debug(
            "Querying HNSW index with {} of shape {}".format(type(X), X.shape)
        )
        complete = False
        while not complete:
            tasklogger.log_debug(
                "Arguments n_neighbors={}, efSearch={}".format(n_neighbors, efSearch)
            )
            self.index.setQueryTimeParams({"efSearch": efSearch})
            indices, distances = zip(
                *(self.index.knnQueryBatch(X, k=n_neighbors, num_threads=self.n_jobs))
            )
            try:
                distances, indices = np.vstack(distances), np.vstack(indices)
                tasklogger.log_debug("Complete")
                complete = True
            except ValueError:
                tasklogger.log_debug("Failed")
                efSearch = efSearch * 2
        return distances, indices

    def kneighbors_graph(
        self, X=None, n_neighbors=None, mode="connectivity", efSearch=100, **kwargs
    ):
        if mode != "connectivity":
            raise NotImplementedError
        if X is not None and X is not self.X:
            self.fit(X)
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        _, indices = self.kneighbors(
            X=X, n_neighbors=n_neighbors, efSearch=efSearch, **kwargs
        )
        return sparse.coo_matrix(
            (
                np.ones(self.X.shape[0] * n_neighbors),
                (np.repeat(np.arange(self.X.shape[0]), n_neighbors), indices.flatten()),
            ),
            shape=(self.X.shape[0], self.X.shape[0]),
        )

    def set_params(self, n_neighbors=None, n_jobs=None):
        if n_neighbors is not None:
            self.n_neighbors = n_neighbors
        if n_jobs is not None:
            self.n_jobs = n_jobs
