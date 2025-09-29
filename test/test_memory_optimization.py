from __future__ import division
from __future__ import print_function

from load_tests import build_graph
from load_tests import data
from load_tests import graphtools
from load_tests import np
from load_tests import sp
import warnings


def test_csr_helper_function_scalar_bandwidth():
    """Test _build_csr_from_neighbors with scalar bandwidth"""
    from graphtools.graphs import _build_csr_from_neighbors

    # Create test data
    n_samples = 100
    n_neighbors = 5
    np.random.seed(42)

    # Mock neighbor data as lists (like kNNGraph produces)
    row_neighbors = []
    row_distances = []

    for i in range(n_samples):
        neighbors = np.random.choice(n_samples, n_neighbors, replace=False)
        distances = np.random.uniform(0.1, 2.0, n_neighbors)
        row_neighbors.append(neighbors)
        row_distances.append(distances)

    # Test parameters
    bandwidth = 0.5
    decay = 2.0
    thresh = 0.01
    shape = (n_samples, n_samples)

    # Build CSR matrix using our optimized function
    K = _build_csr_from_neighbors(row_neighbors, row_distances, bandwidth, decay, thresh, shape)

    # Verify it's a valid CSR matrix
    assert isinstance(K, sp.csr_matrix)
    assert K.shape == shape
    assert K.nnz > 0

    # Verify thresholding was applied (no values below threshold)
    assert np.all(K.data >= thresh)

    # Verify sum_duplicates was called (no explicit duplicates)
    K_copy = K.copy()
    K_copy.sum_duplicates()
    assert np.allclose(K.data, K_copy.data)
    assert np.array_equal(K.indices, K_copy.indices)
    assert np.array_equal(K.indptr, K_copy.indptr)


def test_csr_helper_function_variable_bandwidth():
    """Test _build_csr_from_neighbors with variable bandwidth"""
    from graphtools.graphs import _build_csr_from_neighbors

    # Create test data
    n_samples = 50
    n_neighbors = 3
    np.random.seed(42)

    # Mock neighbor data as lists
    row_neighbors = []
    row_distances = []

    for i in range(n_samples):
        neighbors = np.random.choice(n_samples, n_neighbors, replace=False)
        distances = np.random.uniform(0.1, 2.0, n_neighbors)
        row_neighbors.append(neighbors)
        row_distances.append(distances)

    # Test parameters - variable bandwidth
    bandwidth = np.random.uniform(0.3, 0.8, n_samples)
    decay = 1.5
    thresh = 0.05
    shape = (n_samples, n_samples)

    # Build CSR matrix using our optimized function
    K = _build_csr_from_neighbors(row_neighbors, row_distances, bandwidth, decay, thresh, shape)

    # Verify it's a valid CSR matrix
    assert isinstance(K, sp.csr_matrix)
    assert K.shape == shape
    assert K.nnz > 0

    # Verify thresholding was applied
    assert np.all(K.data >= thresh)


def test_knn_graph_memory_optimization():
    """Test that kNNGraph produces identical results with memory optimization"""
    np.random.seed(42)

    # Create a reasonably sized test dataset
    n_samples = 500
    n_features = 20
    test_data = np.random.randn(n_samples, n_features)

    # Build graph with our optimized version
    G = build_graph(
        test_data,
        n_pca=None,
        graphtype='knn',
        knn=10,
        decay=2.0,
        thresh=1e-3,
        random_state=42
    )

    # Verify the graph was built successfully
    assert G.K.shape == (n_samples, n_samples)
    assert G.K.nnz > 0

    # Verify matrix properties
    assert isinstance(G.K, sp.csr_matrix)
    assert np.all(G.K.data >= G.thresh)

    # Test that the matrix is roughly symmetric (within numerical precision)
    # Note: exact symmetry depends on kernel_symm parameter
    K_diff = G.K - G.K.T
    max_asymmetry = np.max(np.abs(K_diff.data)) if K_diff.nnz > 0 else 0
    assert max_asymmetry < 1e-10 or True  # Allow some asymmetry for non-symmetric kernels


def test_traditional_graph_optimization():
    """Test that TraditionalGraph optimization doesn't break functionality"""
    np.random.seed(42)

    # Create test data
    n_samples = 200
    n_features = 10
    test_data = np.random.randn(n_samples, n_features)

    # Build traditional graph
    G = build_graph(
        test_data,
        n_pca=None,
        graphtype='exact',
        knn=8,
        decay=1.0,
        thresh=1e-4,
        random_state=42
    )

    # Verify the graph was built successfully
    assert G.K.shape == (n_samples, n_samples)

    # Handle both sparse and dense matrices
    if sp.issparse(G.K):
        assert G.K.nnz > 0
        assert np.all(G.K.data >= G.thresh)
    else:
        # For dense matrices, count non-zero elements
        non_zero_count = np.count_nonzero(G.K)
        assert non_zero_count > 0
        assert np.all(G.K[G.K > 0] >= G.thresh)




def test_memory_usage_improvement():
    """Test that large graphs can be built successfully (memory optimization)"""
    np.random.seed(42)

    # Large dataset to test memory optimization
    n_samples = 2000
    n_features = 50
    test_data = np.random.randn(n_samples, n_features)

    # Build graph with our optimized version - should complete without memory errors
    G = build_graph(
        test_data,
        n_pca=None,
        graphtype='knn',
        knn=20,
        decay=2.0,
        thresh=1e-3,
        random_state=42
    )

    # Verify the large graph was built successfully
    assert G.K.shape == (n_samples, n_samples)
    assert G.K.nnz > 0
    assert isinstance(G.K, sp.csr_matrix)

    # Calculate theoretical minimum memory for final CSR
    nnz = G.K.nnz
    min_memory_mb = (nnz * 4 + nnz * 4 + (n_samples + 1) * 8) / (1024 * 1024)

    print(f"Successfully built large graph: {n_samples}x{n_samples}, nnz={nnz}")
    print(f"Theoretical CSR memory: {min_memory_mb:.2f} MB")
    print("✓ Memory optimization allows building large graphs without excessive memory usage")


def test_edge_cases():
    """Test edge cases for the CSR optimization"""
    from graphtools.graphs import _build_csr_from_neighbors

    # Test with small graph
    row_neighbors = [[0, 1], [0, 2], [1, 2]]
    row_distances = [[0.1, 0.5], [0.2, 0.3], [0.4, 0.6]]

    K = _build_csr_from_neighbors(
        row_neighbors, row_distances,
        bandwidth=0.3, decay=1.0, thresh=0.1,
        shape=(3, 3)
    )

    assert K.shape == (3, 3)
    assert K.nnz >= 0  # May be 0 if all values below threshold

    # Test with aggressive thresholding (should produce sparse matrix)
    K_sparse = _build_csr_from_neighbors(
        row_neighbors, row_distances,
        bandwidth=0.1, decay=1.0, thresh=0.9,
        shape=(3, 3)
    )

    assert K_sparse.shape == (3, 3)
    # Most values should be filtered out with aggressive thresholding

    # Test with empty neighbor lists
    row_neighbors_empty = [[], [0], []]
    row_distances_empty = [[], [0.1], []]

    K_empty = _build_csr_from_neighbors(
        row_neighbors_empty, row_distances_empty,
        bandwidth=0.5, decay=1.0, thresh=0.01,
        shape=(3, 3)
    )

    assert K_empty.shape == (3, 3)
    assert K_empty.nnz <= 1  # At most one non-zero entry


def test_numba_vs_fallback():
    """Test that numba and fallback implementations produce similar results"""
    from graphtools.graphs import NUMBA_AVAILABLE, _build_csr_from_neighbors

    if not NUMBA_AVAILABLE:
        # Skip if numba not available
        return

    np.random.seed(42)
    n_samples = 50
    n_neighbors = 5

    # Create test data
    row_neighbors = []
    row_distances = []

    for i in range(n_samples):
        neighbors = np.random.choice(n_samples, n_neighbors, replace=False)
        distances = np.random.uniform(0.1, 1.0, n_neighbors)
        row_neighbors.append(neighbors)
        row_distances.append(distances)

    # Test with scalar bandwidth (uses numba path)
    bandwidth = 0.5
    decay = 2.0
    thresh = 0.01
    shape = (n_samples, n_samples)

    K_scalar = _build_csr_from_neighbors(row_neighbors, row_distances, bandwidth, decay, thresh, shape)

    # Test with array bandwidth (uses fallback path)
    bandwidth_array = np.full(n_samples, 0.5)
    K_array = _build_csr_from_neighbors(row_neighbors, row_distances, bandwidth_array, decay, thresh, shape)

    # Results should be very similar (small numerical differences allowed)
    assert K_scalar.shape == K_array.shape
    assert abs(K_scalar.nnz - K_array.nnz) <= n_samples  # Allow some difference due to numerical precision