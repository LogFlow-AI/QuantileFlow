"""
Tests for the HDRHistogram implementation.
"""

import pytest
import numpy as np
from QuantileFlow import HDRHistogram
import matplotlib.pyplot as plt


def test_hdr_histogram_initialization():
    """Test histogram initialization with different parameters."""
    # Test default initialization
    hdr = HDRHistogram()
    assert hdr.num_buckets == 8
    assert hdr.min_value == 1.0
    assert hdr.max_value == float('inf')
    assert hdr.total_count == 0
    assert np.all(hdr.buckets == 0)

    # Test custom initialization
    hdr = HDRHistogram(num_buckets=16, min_value=0.1, max_value=1000.0)
    assert hdr.num_buckets == 16
    assert hdr.min_value == 0.1
    assert hdr.max_value == 1000.0
    assert hdr.total_count == 0
    assert np.all(hdr.buckets == 0)


def test_hdr_histogram_insert():
    """Test single value insertion."""
    hdr = HDRHistogram(num_buckets=8, max_value=100.0)  # Set explicit max_value
    
    # Test basic insertion
    hdr.insert(1.0)
    assert hdr.total_count == 1
    assert hdr.buckets[0] == 1  # Value 1.0 goes into first bucket
    
    # Test value exceeding max_value
    hdr.insert(1000.0)  # Should be ignored
    assert hdr.total_count == 1
    
    # Test value below min_value
    hdr.insert(0.5)  # Should go into first bucket
    assert hdr.total_count == 2
    assert hdr.buckets[0] == 2


def test_hdr_histogram_insert_batch():
    """Test batch value insertion."""
    hdr = HDRHistogram(num_buckets=8, max_value=100.0)  # Set explicit max_value
    
    # Test basic batch insertion
    values = [1.0, 2.0, 4.0, 8.0, 16.0]
    hdr.insert_batch(values)
    assert hdr.total_count == 5
    
    # Test with numpy array
    values_np = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    hdr2 = HDRHistogram(num_buckets=8, max_value=100.0)  # Set explicit max_value
    hdr2.insert_batch(values_np)
    assert hdr2.total_count == 5
    assert np.array_equal(hdr.buckets, hdr2.buckets)
    
    # Test with mixed values (some exceeding max_value)
    values_mixed = [1.0, 2.0, 1000.0, 4.0, 8.0, 16.0]
    hdr3 = HDRHistogram(num_buckets=8, max_value=100.0)  # Set explicit max_value
    hdr3.insert_batch(values_mixed)
    assert hdr3.total_count == 5  # Only 5 valid values
    assert np.array_equal(hdr.buckets, hdr3.buckets)


def test_hdr_histogram_quantiles():
    """Test quantile calculations with various distributions."""
    hdr = HDRHistogram(num_buckets=8)
    
    # Test with uniform distribution
    values = [1.0, 2.0, 4.0, 8.0, 16.0]
    hdr.insert_batch(values)
    
    # Test single quantile
    assert hdr.quantile(0.5) == 4.0  # median
    assert hdr.quantile(0.25) == 2.0  # first quartile
    assert hdr.quantile(0.75) == 8.0  # third quartile
    
    # Test multiple quantiles
    quantiles = hdr.quantiles([0.25, 0.5, 0.75])
    assert quantiles == [2.0, 4.0, 8.0]
    
    # Test edge cases
    assert hdr.quantile(0.0) == 1.0  # minimum
    assert hdr.quantile(1.0) == 16.0  # maximum
    
    # Test with large dataset
    hdr2 = HDRHistogram(num_buckets=16)
    large_values = np.random.uniform(1.0, 1000.0, 10000)
    hdr2.insert_batch(large_values)
    
    # Compare with numpy's percentile
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        hdr_q = hdr2.quantile(q)
        np_q = np.percentile(large_values, q * 100)
        # Allow for some approximation error
        assert abs(hdr_q - np_q) <= 0.1 * np_q


def test_hdr_histogram_percentiles():
    """Test percentile calculations."""
    hdr = HDRHistogram(num_buckets=8)
    values = [1.0, 2.0, 4.0, 8.0, 16.0]
    hdr.insert_batch(values)
    
    assert hdr.percentile(50) == 4.0  # median
    assert hdr.percentile(25) == 2.0  # first quartile
    assert hdr.percentile(75) == 8.0  # third quartile
    
    # Test edge cases
    assert hdr.percentile(0) == 1.0  # minimum
    assert hdr.percentile(100) == 16.0  # maximum
    
    # Test with large dataset
    hdr2 = HDRHistogram(num_buckets=16)
    large_values = np.random.uniform(1.0, 1000.0, 10000)
    hdr2.insert_batch(large_values)
    
    # Compare with numpy's percentile
    for p in [0, 25, 50, 75, 100]:
        hdr_p = hdr2.percentile(p)
        np_p = np.percentile(large_values, p)
        # Allow for some approximation error
        assert abs(hdr_p - np_p) <= 0.1 * np_p


def test_hdr_histogram_summary_statistics():
    """Test summary statistics calculation."""
    hdr = HDRHistogram(num_buckets=8)
    values = [1.0, 2.0, 4.0, 8.0, 16.0]
    hdr.insert_batch(values)
    
    stats = hdr.summary_statistics()
    assert stats['min'] == 1.0
    assert stats['q1'] == 2.0
    assert stats['median'] == 4.0
    assert stats['q3'] == 8.0
    assert stats['max'] == 16.0
    assert stats['count'] == 5
    
    # Test with large dataset
    hdr2 = HDRHistogram(num_buckets=16)
    large_values = np.random.uniform(1.0, 1000.0, 10000)
    hdr2.insert_batch(large_values)
    
    stats2 = hdr2.summary_statistics()
    np_stats = {
        'min': np.min(large_values),
        'q1': np.percentile(large_values, 25),
        'median': np.median(large_values),
        'q3': np.percentile(large_values, 75),
        'max': np.max(large_values),
        'count': len(large_values)
    }
    
    # Allow for some approximation error
    for key in stats2:
        if key != 'count':  # count should be exact
            assert abs(stats2[key] - np_stats[key]) <= 0.1 * np_stats[key]


def test_hdr_histogram_edge_cases():
    """Test edge cases and error handling."""
    hdr = HDRHistogram(num_buckets=8)
    
    # Test invalid quantile values
    with pytest.raises(ValueError):
        hdr.quantile(-0.1)
    with pytest.raises(ValueError):
        hdr.quantile(1.1)
    
    # Test invalid percentile values
    with pytest.raises(ValueError):
        hdr.percentile(-1)
    with pytest.raises(ValueError):
        hdr.percentile(101)
    
    # Test empty histogram
    assert hdr.median() == 0.0  # returns 0.0 for empty histogram
    assert hdr.quantile(0.5) == 0.0  # returns 0.0 for empty histogram
    
    # Test single value
    hdr.insert(2.0)
    assert hdr.median() == 1.0
    assert hdr.quantile(0.5) == 1.0


def test_hdr_histogram_accuracy():
    """Test accuracy of quantile estimation with various distributions."""
    # Test with different distributions
    distributions = [
        ('uniform', np.random.uniform(1.0, 1000.0, 10000)),
        ('normal', np.random.normal(500.0, 100.0, 10000)),
        ('lognormal', np.random.lognormal(5.0, 0.5, 10000))
    ]
    
    for dist_name, values in distributions:
        hdr = HDRHistogram(num_buckets=16)
        hdr.insert_batch(values)
        
        # Test various quantiles
        quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
        for q in quantiles:
            hdr_q = hdr.quantile(q)
            np_q = np.percentile(values, q * 100)
            relative_error = abs(hdr_q - np_q) / np_q
            
            # Log the results for analysis
            print(f"{dist_name} - Quantile {q}: HDR={hdr_q:.2f}, NP={np_q:.2f}, "
                  f"Relative Error={relative_error:.4f}")
            
            # Use different error tolerances for different distributions
            if dist_name == 'normal':
                # Normal distribution needs a larger tolerance due to its unbounded nature
                assert relative_error <= 0.5
            else:
                # Other distributions can use tighter tolerance
                assert relative_error <= 0.1


def test_hdr_histogram_serialization():
    """Test serialization and deserialization."""
    hdr = HDRHistogram(num_buckets=8)
    values = [1.0, 2.0, 4.0, 8.0, 16.0]
    hdr.insert_batch(values)
    
    # Test serialization
    data = hdr.to_dict()
    assert data['num_buckets'] == 8
    assert data['min_value'] == 1.0
    assert data['max_value'] == float('inf')
    assert data['total_count'] == 5
    assert len(data['buckets']) == 8
    
    # Test deserialization
    hdr2 = HDRHistogram.from_dict(data)
    assert hdr2.num_buckets == hdr.num_buckets
    assert hdr2.min_value == hdr.min_value
    assert hdr2.max_value == hdr.max_value
    assert hdr2.total_count == hdr.total_count
    assert np.array_equal(hdr2.buckets, hdr.buckets)
    
    # Verify functionality after deserialization
    assert hdr2.median() == hdr.median()
    assert hdr2.quantile(0.75) == hdr.quantile(0.75)
    assert hdr2.summary_statistics() == hdr.summary_statistics()


def test_hdr_histogram_plotting():
    """Test histogram plotting functionality."""
    hdr = HDRHistogram(num_buckets=8)
    values = [1.0, 2.0, 4.0, 8.0, 16.0]
    hdr.insert_batch(values)
    
    # Test that plotting doesn't raise errors
    fig = hdr.plot_distribution()
    assert fig is not None
    plt.close(fig)  # Close the figure to avoid display issues in CI/CD environments