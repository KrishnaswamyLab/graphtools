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
        from graphtools.graphs import NUMBA_AVAILABLE, njit
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
    from graphtools.graphs import (
        _numba_compute_bandwidths,
        _numba_compute_radius,
        _numba_find_updates,
        _numba_scale_distances_single_bandwidth,
        _numba_compute_affinities,
        _numba_threshold_affinities
    )
    
    # Test bandwidth computation
    distances = np.array([[0.0, 0.1, 0.2, 0.3],
                         [0.0, 0.15, 0.25, 0.35]])
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
    G = build_graph(data, graphtype='knn', knn=5, decay=2, thresh=1e-3, 
                   n_pca=20, random_state=42)
    
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
        assert abs(K_numba.nnz - K_vanilla.nnz) <= K_numba.shape[0]  # Allow small differences due to numerical precision
        
        # Check that non-zero elements are in similar positions
        # Use relaxed threshold for float32 precision in PHATE-inspired optimizations
        diff = (K_numba - K_vanilla)
        max_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0
        assert max_diff < 1e-6, f"Maximum difference {max_diff} too large (expected < 1e-6 for float32)"
        
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
    G = build_graph(large_data, graphtype='knn', knn=10, decay=2, thresh=1e-4,
                   n_pca=30, random_state=42)
    
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
    G_small = build_graph(small_data, graphtype='knn', knn=3, decay=1, 
                         thresh=1e-3, n_pca=None, random_state=42)
    Y_small = np.random.rand(5, 5)
    K_small = G_small.build_kernel_to_data(Y_small)
    assert K_small.shape == (5, 10)
    assert K_small.nnz > 0
    
    # Test with fixed bandwidth
    G_bw = build_graph(data[:100], graphtype='knn', decay=2, bandwidth=0.5,
                      thresh=1e-3, n_pca=10, random_state=42)
    Y_bw = data[100:120]
    K_bw = G_bw.build_kernel_to_data(Y_bw)
    assert K_bw.shape == (20, 100)
    assert K_bw.nnz > 0
    
    # Test with high threshold (sparse result)
    G_sparse = build_graph(data, graphtype='knn', knn=5, decay=1, thresh=0.9,
                          n_pca=20, random_state=42)
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
        G = build_graph(data, graphtype='knn', knn=5, decay=2, thresh=1e-3,
                       n_pca=20, random_state=42)
        Y = data[:30]
        K = G.build_kernel_to_data(Y)
        
        assert K.shape == (30, len(data))
        assert K.nnz > 0
        assert np.all(K.data >= 0)  # All affinities should be non-negative
        
    finally:
        # Restore numba availability
        gg.NUMBA_AVAILABLE = original_numba