"""
Tests for the core MomentSketch class.

This module contains unit tests for the MomentSketch class, testing its
initialization, data insertion, merging, quantile calculation, serialization,
and other core functionality.
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from QuantileFlow import MomentSketch


class TestMomentSketch:
    """Test suite for the MomentSketch class."""

    def test_init(self):
        """Test initialization with default parameters."""
        sketch = MomentSketch()
        assert sketch is not None
        assert hasattr(sketch, 'sketch')
        
        # Test with custom parameters
        sketch = MomentSketch(num_moments=10, compress_values=True)
        assert sketch is not None
        assert sketch.sketch.get_k() == 10
        assert sketch.sketch.get_compressed() is True

    def test_insert_single(self):
        """Test inserting a single value."""
        sketch = MomentSketch(num_moments=5)
        sketch.insert(10.5)
        
        # The value should be in the range of values
        assert sketch.sketch.get_min() <= 10.5
        assert sketch.sketch.get_max() >= 10.5
        
        # Power sum[0] should be 1 (count)
        assert sketch.sketch.get_power_sums()[0] == 1
        
        # Power sum[1] should be the value (sum)
        assert sketch.sketch.get_power_sums()[1] == 10.5
        
    def test_insert_batch(self):
        """Test inserting multiple values."""
        sketch = MomentSketch(num_moments=5)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sketch.insert_batch(values)
        
        # Check min and max
        assert sketch.sketch.get_min() == 1.0
        assert sketch.sketch.get_max() == 5.0
        
        # Check count
        assert sketch.sketch.get_power_sums()[0] == 5.0
        
        # Check sum
        assert sketch.sketch.get_power_sums()[1] == 15.0  # sum of values
        
    def test_merge(self):
        """Test merging two sketches."""
        sketch1 = MomentSketch(num_moments=5)
        sketch2 = MomentSketch(num_moments=5)
        
        sketch1.insert_batch([1.0, 2.0, 3.0])
        sketch2.insert_batch([4.0, 5.0, 6.0])
        
        sketch1.merge(sketch2)
        
        # Check min and max after merge
        assert sketch1.sketch.get_min() == 1.0
        assert sketch1.sketch.get_max() == 6.0
        
        # Check count
        assert sketch1.sketch.get_power_sums()[0] == 6.0
        
        # Check sum
        assert sketch1.sketch.get_power_sums()[1] == 21.0
        
    def test_merge_incompatible(self):
        """Test merging with incompatible compression settings."""
        sketch1 = MomentSketch(compress_values=True)
        sketch2 = MomentSketch(compress_values=False)
        
        with pytest.raises(ValueError, match="Cannot merge sketches with different compression settings"):
            sketch1.merge(sketch2)
            
    def test_quantile(self):
        """Test quantile estimation."""
        sketch = MomentSketch(num_moments=30)  # More moments for better accuracy
        
        # Insert sorted values for easy testing
        values = np.arange(1, 101)
        sketch.insert_batch(values)
        
        # Test median
        median = sketch.quantile(0.5)
        
        # Allow wide error margin for the approximate algorithm
        assert median >= 1 and median <= 100
        
    def test_quantile_validation(self):
        """Test validation of quantile parameter."""
        sketch = MomentSketch()
        sketch.insert_batch([1, 2, 3])
        
        with pytest.raises(ValueError, match="Quantile must be between 0 and 1"):
            sketch.quantile(-0.1)
            
        with pytest.raises(ValueError, match="Quantile must be between 0 and 1"):
            sketch.quantile(1.1)
            
    def test_quantiles(self):
        """Test multiple quantile estimation."""
        sketch = MomentSketch(num_moments=30)  # More moments for better accuracy
        
        # Insert sorted values for easy testing
        values = np.arange(1, 101)
        sketch.insert_batch(values)
        
        # Test multiple quantiles
        qs = [0.1, 0.5, 0.9]
        results = sketch.quantiles(qs)
        
        assert len(results) == 3
        
        # All quantiles should be in the range of the data
        assert all(1 <= q <= 100 for q in results)
        
        # Quantiles should be in ascending order
        assert results[0] <= results[1] <= results[2]
        
    def test_quantiles_validation(self):
        """Test validation of quantiles parameters."""
        sketch = MomentSketch()
        sketch.insert_batch([1, 2, 3])
        
        with pytest.raises(ValueError, match="All quantiles must be between 0 and 1"):
            sketch.quantiles([0.5, 1.1])
            
        with pytest.raises(ValueError, match="All quantiles must be between 0 and 1"):
            sketch.quantiles([-0.1, 0.5])
            
    def test_median(self):
        """Test median computation."""
        sketch = MomentSketch(num_moments=30)  # More moments for better accuracy
        values = np.arange(1, 101)
        sketch.insert_batch(values)
        
        median = sketch.median()
        
        # Allow wide error margin for the approximate algorithm
        assert median >= 1 and median <= 100
        
    def test_percentile(self):
        """Test percentile computation."""
        sketch = MomentSketch(num_moments=30)  # More moments for better accuracy
        values = np.arange(1, 101)
        sketch.insert_batch(values)
        
        # Test 75th percentile
        p75 = sketch.percentile(75)
        
        # Allow wide error margin for the approximate algorithm
        assert p75 >= 1 and p75 <= 100
        
    def test_percentile_validation(self):
        """Test validation of percentile parameter."""
        sketch = MomentSketch()
        sketch.insert_batch([1, 2, 3])
        
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            sketch.percentile(-10)
            
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            sketch.percentile(110)
            
    def test_interquartile_range(self):
        """Test interquartile range computation."""
        sketch = MomentSketch(num_moments=30)  # More moments for better accuracy
        values = np.arange(1, 101)
        sketch.insert_batch(values)
        
        iqr = sketch.interquartile_range()
        
        # Expected IQR for uniform distribution 1-100 should be positive
        assert iqr > 0
        
        # IQR can't be greater than the range of data
        assert iqr <= 100
        
    def test_summary_statistics(self):
        """Test summary statistics."""
        sketch = MomentSketch(num_moments=30)  # More moments for better accuracy
        values = np.arange(1, 11)  # 1 to 10
        sketch.insert_batch(values)
        
        stats = sketch.summary_statistics()
        
        # Check that all expected keys are present
        expected_keys = ['min', 'q1', 'median', 'q3', 'max', 'count', 'mean']
        for key in expected_keys:
            assert key in stats
        
        # Check approximate values
        assert stats['min'] >= 1
        assert stats['max'] <= 10
        assert stats['count'] == 10
        assert 5 <= stats['mean'] <= 6  # Mean should be close to 5.5
        
        # Quartiles should be in order
        assert stats['min'] <= stats['q1'] <= stats['median'] <= stats['q3'] <= stats['max']
        
    @patch('matplotlib.pyplot.figure')
    def test_plot_distribution(self, mock_figure):
        """Test plotting distribution."""
        # Mock figure and axes
        mock_ax = MagicMock()
        mock_figure.return_value.axes = [mock_ax]
        
        sketch = MomentSketch()
        values = np.arange(1, 11)
        sketch.insert_batch(values)
        
        fig = sketch.plot_distribution()
        
        # Verify the figure was created and returned
        assert fig is not None
        mock_figure.assert_called_once()
        
    def test_to_dict(self):
        """Test serialization to dictionary."""
        sketch = MomentSketch(num_moments=5)
        sketch.insert_batch([1, 2, 3, 4, 5])
        
        data_dict = sketch.to_dict()
        
        # Check that all expected keys are present
        expected_keys = ['power_sums', 'min_val', 'max_val', 'use_arc_sinh']
        for key in expected_keys:
            assert key in data_dict
            
        # Check specific values
        assert data_dict['min_val'] == 1
        assert data_dict['max_val'] == 5
        assert len(data_dict['power_sums']) == 5
        
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        original = MomentSketch(num_moments=5)
        original.insert_batch([1, 2, 3, 4, 5])
        
        data_dict = original.to_dict()
        restored = MomentSketch.from_dict(data_dict)
        
        # Check that the restored sketch has the same properties
        assert restored.sketch.get_min() == original.sketch.get_min()
        assert restored.sketch.get_max() == original.sketch.get_max()
        assert restored.sketch.get_compressed() == original.sketch.get_compressed()
        
        # Check that quantiles are the same
        original_median = original.median()
        restored_median = restored.median()
        assert abs(original_median - restored_median) < 1e-10

    def test_empty_sketch(self):
        """Test behavior with empty sketch."""
        sketch = MomentSketch()
        
        # Min and max should be default values
        assert sketch.sketch.get_min() == float('inf')
        assert sketch.sketch.get_max() == float('-inf')
        
        # Power sums should be zeros
        assert all(x == 0 for x in sketch.sketch.get_power_sums())
        
    def test_compressed_values(self):
        """Test with compressed values."""
        # Create a sketch with compression
        sketch = MomentSketch(compress_values=True)
        
        # Insert some values including extreme ones
        sketch.insert_batch([0.001, 1, 1000, 1000000])
        
        # Check that the min and max are preserved
        stats = sketch.summary_statistics()
        assert stats['min'] >= 0.001
        assert stats['max'] <= 1000000
        
        # Check that we can compute quantiles without errors
        median = sketch.median()
        assert median is not None 