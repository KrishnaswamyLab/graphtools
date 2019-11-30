import graphtools.utils
from parameterized import parameterized
from scipy import sparse
import numpy as np
import graphtools
from load_tests import data
from load_tests import optimal_singular_value_thresholding_constants as osvth



@parameterized(
    [
        (np.array,),
        (sparse.csr_matrix,),
        (sparse.csc_matrix,),
        (sparse.bsr_matrix,),
        (sparse.lil_matrix,),
        (sparse.coo_matrix,),
    ]
)
def test_nonzero_discrete(matrix_class):
    X = np.random.choice([0, 1, 2], p=[0.95, 0.025, 0.025], size=(100, 100))
    X = matrix_class(X)
    assert graphtools.utils.nonzero_discrete(X, [1, 2])
    assert not graphtools.utils.nonzero_discrete(X, [1, 3])


@parameterized([(0,), (1e-4,)])
def test_nonzero_discrete_knngraph(thresh):
    G = graphtools.Graph(data, n_pca=10, knn=5, decay=None, thresh=thresh)
    assert graphtools.utils.nonzero_discrete(G.K, [0.5, 1])


@parameterized([(0,), (1e-4,)])
def test_nonzero_discrete_decay_graph(thresh):
    G = graphtools.Graph(data, n_pca=10, knn=5, decay=15, thresh=thresh)
    assert not graphtools.utils.nonzero_discrete(G.K, [0.5, 1])

def test_marcenkopastur_median():
    """ We can test Marcenko-Pastur median out to 4 decimal places by using the table defined in 
            "The Optimal Hard Threshold for Singular Values is 4/sqrt(3)" (2014)
            Matan Gavish, David L. Donoho 
            https://arxiv.org/abs/1305.5870" """

    #Our algorithm will compute more digits though!
    x = np.arange(0.05, 1.0, 0.05)
    x = np.round(x,2)

    for v in x:
        table_mp_median = (osvth['lambda'][v]/osvth['omega'][v])**2
        assert np.isclose(graphtools.utils.MarcenkoPastur.median(v),table_mp_median, 1e-4)
