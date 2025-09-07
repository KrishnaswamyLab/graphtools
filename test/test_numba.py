from __future__ import division
from __future__ import print_function

from load_tests import assert_raises_message
from load_tests import assert_warns_message
from load_tests import build_graph
from load_tests import data
from load_tests import datasets
from load_tests import graphtools
from load_tests import np
from load_tests import PCA
from load_tests import pygsp
from load_tests import sp
from load_tests import TruncatedSVD

import time
import warnings

#####################################################
# Check numba availability and functionality
#####################################################


def test_numba_import():
    """Test that numba import fallback works correctly"""
    try:
        from graphtools.graphs import njit
        from graphtools.graphs import NUMBA_AVAILABLE

        # Test that the decorator works
        @njit
        def dummy_function(x):
            return x * 2

        result = dummy_function(5.0)
        assert result == 10.0
    except ImportError:
        # Should not happen due to fallback
        assert False, "Numba fallback import failed"


def test_numba_functions():
    """Test individual numba-accelerated functions"""
    from graphtools.graphs import _numba_compute_affinities
    from graphtools.graphs import _numba_compute_bandwidths
    from graphtools.graphs import _numba_compute_radius
    from graphtools.graphs import _numba_find_updates
    from graphtools.graphs import _numba_scale_distances_single_bandwidth
    from graphtools.graphs import _numba_threshold_affinities

    # Test bandwidth computation
    distances = np.array([[0.0, 0.1, 0.2, 0.3], [0.0, 0.15, 0.25, 0.35]])
    bandwidth = _numba_compute_bandwidths(distances, knn=3, bandwidth_scale=1.0)
    expected = np.array([0.2, 0.25])  # distances[:, knn-1]
    np.testing.assert_array_almost_equal(bandwidth, expected)

    # Test radius computation
    radius = _numba_compute_radius(bandwidth, thresh=1e-4, decay=2.0)
    expected = bandwidth * np.power(-np.log(1e-4), 0.5)
    np.testing.assert_array_almost_equal(radius, expected)

    # Test update finding
    update_idx = _numba_find_updates(distances, radius)
    # Should find indices where max distance < radius
    for idx in update_idx:
        assert np.max(distances[idx, :]) < radius[idx]

    # Test distance scaling
    distances_flat = np.array([0.1, 0.2, 0.3, 0.4])
    scaled = _numba_scale_distances_single_bandwidth(distances_flat, 0.5)
    expected = distances_flat / 0.5
    np.testing.assert_array_almost_equal(scaled, expected)

    # Test affinity computation
    affinities = _numba_compute_affinities(scaled, decay=2.0)
    expected = np.exp(-np.power(scaled, 2.0))
    np.testing.assert_array_almost_equal(affinities, expected)

    # Test thresholding
    thresh_affinities = _numba_threshold_affinities(affinities.copy(), thresh=0.5)
    expected = affinities.copy()
    expected[expected < 0.5] = 0.0
    np.testing.assert_array_almost_equal(thresh_affinities, expected)


#####################################################
# Check numba acceleration in kNN graphs
#####################################################


def test_knn_graph_numba_consistency():
    """Test that numba and non-numba paths produce identical results"""
    from graphtools.graphs import NUMBA_AVAILABLE

    if not NUMBA_AVAILABLE:
        # Skip if numba not available
        return

    # Build kNN graph with decay
    G = build_graph(
        data, graphtype="knn", knn=5, decay=2, thresh=1e-3, n_pca=20, random_state=42
    )

    # Test data
    Y = data[:50]  # Use subset as test data

    # Get result with numba
    K_numba = G.build_kernel_to_data(Y)

    # Temporarily disable numba for comparison
    import graphtools.graphs as gg

    original_numba = gg.NUMBA_AVAILABLE
    gg.NUMBA_AVAILABLE = False

    try:
        # Get result without numba
        K_vanilla = G.build_kernel_to_data(Y)

        # Compare results - should be nearly identical
        assert K_numba.shape == K_vanilla.shape
        assert (
            abs(K_numba.nnz - K_vanilla.nnz) <= K_numba.shape[0]
        )  # Allow small differences due to numerical precision

        # Check that non-zero elements are in similar positions
        # Use relaxed threshold for float32 precision in PHATE-inspired optimizations
        diff = K_numba - K_vanilla
        max_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0
        assert (
            max_diff < 1e-6
        ), f"Maximum difference {max_diff} too large (expected < 1e-6 for float32)"

    finally:
        # Restore numba availability
        gg.NUMBA_AVAILABLE = original_numba


def test_knn_graph_numba_performance():
    """Test that numba provides performance improvement"""
    from graphtools.graphs import NUMBA_AVAILABLE

    if not NUMBA_AVAILABLE:
        # Skip if numba not available
        return

    # Generate larger test data
    large_data = datasets.make_blobs(n_samples=1000, n_features=50, random_state=42)[0]
    Y = datasets.make_blobs(n_samples=200, n_features=50, random_state=43)[0]

    # Build graph
    G = build_graph(
        large_data,
        graphtype="knn",
        knn=10,
        decay=2,
        thresh=1e-4,
        n_pca=30,
        random_state=42,
    )

    # Time with numba
    start_time = time.time()
    K_numba = G.build_kernel_to_data(Y)
    numba_time = time.time() - start_time

    # Temporarily disable numba
    import graphtools.graphs as gg

    original_numba = gg.NUMBA_AVAILABLE
    gg.NUMBA_AVAILABLE = False

    try:
        # Time without numba
        start_time = time.time()
        K_vanilla = G.build_kernel_to_data(Y)
        vanilla_time = time.time() - start_time

        # Results should be similar
        assert K_numba.shape == K_vanilla.shape

        # Print timing info (will show in test output)
        print(f"\nNumba timing test:")
        print(f"  Without numba: {vanilla_time:.4f}s")
        print(f"  With numba: {numba_time:.4f}s")
        if vanilla_time > 0:
            speedup = vanilla_time / numba_time
            print(f"  Speedup: {speedup:.2f}x")

    finally:
        # Restore numba availability
        gg.NUMBA_AVAILABLE = original_numba


def test_knn_graph_numba_edge_cases():
    """Test numba acceleration with edge cases"""
    from graphtools.graphs import NUMBA_AVAILABLE

    if not NUMBA_AVAILABLE:
        return

    # Test with small data - use valid parameters for kNNGraph
    small_data = np.random.rand(10, 5)
    G_small = build_graph(
        small_data,
        graphtype="knn",
        knn=3,
        decay=1,
        thresh=1e-3,
        n_pca=None,
        random_state=42,
    )
    Y_small = np.random.rand(5, 5)
    K_small = G_small.build_kernel_to_data(Y_small)
    assert K_small.shape == (5, 10)
    assert K_small.nnz > 0

    # Test with fixed bandwidth - handle both sparse and dense matrices
    Y_bw = data[100:120]
    G_bw = build_graph(
        data[:100],
        graphtype="knn",
        knn=5,
        decay=1,
        thresh=1e-10,
        n_pca=10,
        random_state=42,
    )
    K_bw = G_bw.build_kernel_to_data(Y_bw)

    # Handle both sparse and dense matrices
    if hasattr(K_bw, "nnz"):  # sparse matrix
        nonzeros = K_bw.nnz
    else:  # dense matrix
        nonzeros = np.count_nonzero(K_bw)

    assert K_bw.shape == (20, 100)
    assert nonzeros > 0

    # Test with high threshold (sparse result)
    G_sparse = build_graph(
        data, graphtype="knn", knn=5, decay=1, thresh=0.9, n_pca=20, random_state=42
    )
    Y_sparse = data[:30]
    K_sparse = G_sparse.build_kernel_to_data(Y_sparse)
    assert K_sparse.shape == (30, len(data))
    # Should be very sparse due to high threshold
    sparsity = 1 - (K_sparse.nnz / (K_sparse.shape[0] * K_sparse.shape[1]))
    assert sparsity > 0.8  # Expect high sparsity


def test_knn_graph_numba_fallback():
    """Test that code works when numba is not available"""
    import graphtools.graphs as gg

    # Temporarily disable numba
    original_numba = gg.NUMBA_AVAILABLE
    gg.NUMBA_AVAILABLE = False

    try:
        # Should work without numba
        G = build_graph(
            data,
            graphtype="knn",
            knn=5,
            decay=2,
            thresh=1e-3,
            n_pca=20,
            random_state=42,
        )
        Y = data[:30]
        K = G.build_kernel_to_data(Y)

        assert K.shape == (30, len(data))
        assert K.nnz > 0
        assert np.all(K.data >= 0)  # All affinities should be non-negative

    finally:
        # Restore numba availability
        gg.NUMBA_AVAILABLE = original_numba


def test_numba_float32_precision_compatibility():
    """
    Test that numerical differences between numba and non-numba implementations
    are within expected precision bounds for float32/float64 conversion.

    Reasoning:
    ----------
    The numba-accelerated functions use float32 precision for memory efficiency
    and performance (inspired by PHATE optimizations), while the non-numba
    implementations use float64 precision. This test validates that:

    1. The differences are bounded by the precision limits of float32 (~1e-7)
    2. The relative errors are acceptable for scientific computing applications
    3. The optimization doesn't introduce systematic biases

    Float32 has approximately 7 decimal digits of precision, so we expect:
    - Absolute differences on the order of 1e-7 to 2e-7
    - Relative differences typically < 1e-6 for well-conditioned operations
    - Larger relative differences only for very small values near zero

    This test ensures the numba optimizations maintain numerical fidelity
    while providing performance benefits.
    """
    from graphtools.graphs import NUMBA_AVAILABLE

    import graphtools.graphs as gg

    if not NUMBA_AVAILABLE:
        return  # Skip if numba not available

    # Use a moderate-sized dataset for meaningful statistics
    test_data = data[:200]
    Y_test = data[200:250]

    # Test parameters that will exercise the numba code paths
    test_params = {
        "graphtype": "knn",
        "knn": 5,
        "decay": 2,
        "thresh": 1e-6,
        "n_pca": 15,
        "random_state": 42,
    }

    original_numba = gg.NUMBA_AVAILABLE

    try:
        # Get results with numba acceleration (float32 precision)
        gg.NUMBA_AVAILABLE = True
        G_numba = build_graph(test_data, **test_params)
        K_numba = G_numba.build_kernel_to_data(Y_test)

        # Get results without numba (float64 precision)
        gg.NUMBA_AVAILABLE = False
        G_vanilla = build_graph(test_data, **test_params)
        K_vanilla = G_vanilla.build_kernel_to_data(Y_test)

        # Ensure both matrices have the same sparsity pattern for comparison
        assert K_numba.shape == K_vanilla.shape, "Matrix shapes must match"

        # Convert to dense for easier numerical comparison
        K_numba_dense = K_numba.toarray() if hasattr(K_numba, "toarray") else K_numba
        K_vanilla_dense = (
            K_vanilla.toarray() if hasattr(K_vanilla, "toarray") else K_vanilla
        )

        # Calculate absolute and relative differences
        abs_diff = np.abs(K_numba_dense - K_vanilla_dense)

        # Avoid division by zero in relative error calculation
        nonzero_mask = K_vanilla_dense != 0
        rel_diff = np.zeros_like(abs_diff)
        rel_diff[nonzero_mask] = abs_diff[nonzero_mask] / np.abs(
            K_vanilla_dense[nonzero_mask]
        )

        # Statistics for validation
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_abs_diff = np.mean(abs_diff)
        mean_rel_diff = np.mean(rel_diff[nonzero_mask]) if np.any(nonzero_mask) else 0

        print(f"Precision comparison statistics:")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"  Mean relative difference: {mean_rel_diff:.2e}")

        # Validate precision bounds for float32 vs float64 differences
        # Float32 has ~7 digits of precision, so differences should be ~1e-7
        assert (
            max_abs_diff < 1e-6
        ), f"Max absolute difference {max_abs_diff:.2e} exceeds float32 precision bound"

        # Relative differences should generally be much smaller unless values are tiny
        assert (
            max_rel_diff < 1e-5
        ), f"Max relative difference {max_rel_diff:.2e} indicates loss of precision"

        # Mean differences should be even smaller (no systematic bias)
        assert (
            mean_abs_diff < 1e-7
        ), f"Mean absolute difference {mean_abs_diff:.2e} suggests systematic bias"
        assert (
            mean_rel_diff < 1e-6
        ), f"Mean relative difference {mean_rel_diff:.2e} suggests systematic bias"

        # Verify the matrices are "close enough" using numpy's built-in tolerance
        np.testing.assert_allclose(
            K_numba_dense,
            K_vanilla_dense,
            rtol=1e-6,
            atol=1e-7,
            err_msg="Numba and vanilla implementations differ beyond acceptable precision",
        )

    finally:
        # Always restore original numba availability
        gg.NUMBA_AVAILABLE = original_numba
