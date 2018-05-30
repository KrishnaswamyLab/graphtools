import numpy as np
import warnings

from .logging import (set_logging,
                      log_debug)
from .base import PyGSPGraph
from .graphs import kNNGraph, TraditionalGraph, MNNGraph, LandmarkGraph


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
          use_pygsp=False,
          initialize=True,
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

    graphtype : {'exact', 'knn', 'mnn', 'auto'} (Default: 'auto')
        Manually selects graph type. Only recommended for expert users

    use_pygsp : `bool` (Default: `False`)
        If true, inherits from `pygsp.graphs.Graph`.

    initialize : `bool` (Default: `True`)
        If True, initialize the kernel matrix on instantiation

    **kwargs : extra arguments for `pygsp.graphs.Graph`

    Returns
    -------
    G : `DataGraph`

    Raises
    ------
    ValueError : if selected parameters are incompatible.
    """
    set_logging(verbose)
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
    parent_classes = [base]
    msg = "Building {} graph".format(graphtype)
    if n_landmark is not None:
        parent_classes.append(LandmarkGraph)
        msg = msg + " with landmarks"
    if use_pygsp:
        parent_classes.append(PyGSPGraph)
        if len(parent_classes) > 2:
            msg = msg + " with PyGSP inheritance"
        else:
            msg = msg + " and PyGSP inheritance"

    log_debug(msg)

    # Python3 syntax only
    # class Graph(*parent_classes):
    #     pass
    if len(parent_classes) == 1:
        Graph = parent_classes[0]
    elif len(parent_classes) == 2:
        class Graph(parent_classes[0], parent_classes[1]):
            pass
    elif len(parent_classes) == 2:
        class Graph(parent_classes[0], parent_classes[1], parent_classes[2]):
            pass
    else:
        raise RuntimeError("unknown graph classes")

    # build graph and return
    log_debug("Initializing {} with arguments {}".format(
        parent_classes,
        {
            'n_pca': n_pca,
            'sample_idx': sample_idx,
            'adaptive_k': adaptive_k,
            'precomputed': precomputed,
            'knn': knn,
            'decay': decay,
            'distance': distance,
            'thresh': thresh,
            'n_landmark': n_landmark,
            'n_svd': n_svd,
            'beta': beta,
            'gamma': gamma,
            'n_jobs': n_jobs,
            'verbose': verbose,
            'random_state': random_state,
            'initialize': initialize
        }))
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
                 initialize=initialize,
                 **kwargs)
