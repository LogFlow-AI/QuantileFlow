"""
Integration tests for the MomentSketch implementation.

This module contains end-to-end tests for the MomentSketch implementation with various
data distributions and usage scenarios. It verifies that:
- The sketch provides accurate quantile estimates for different distributions
- Serialization and deserialization work correctly
- Merging operations combine sketches properly
- The implementation handles edge cases like empty sketches and outliers
- Performance characteristics scale appropriately with sketch parameters
"""
import numpy as np
from unittest.mock import patch

from QuantileFlow import MomentSketch


class TestMomentSketchIntegration:
    """Integration tests for the MomentSketch implementation."""
    
    def test_uniform_distribution(self):
        """Test with uniform distribution."""
        # Generate uniform data
        np.random.seed(42)
        data = np.random.uniform(0, 100, 10000)
        
        # Create sketch
        sketch = MomentSketch(num_moments=15)
        sketch.insert_batch(data)
        
        # Test quantiles
        quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
        expected = [np.quantile(data, q) for q in quantiles]
        actual = sketch.quantiles(quantiles)
        
        # Check accuracy - allow 3% error for uniform distribution
        for exp, act in zip(expected, actual):
            assert abs(exp - act) < 0.03 * 100  # 3% of range (100)
            
    def test_normal_distribution(self):
        """Test with normal distribution."""
        # Generate normal data
        np.random.seed(42)
        data = np.random.normal(50, 10, 10000)
        
        # Create sketch
        sketch = MomentSketch(num_moments=15)
        sketch.insert_batch(data)
        
        # Test quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        expected = [np.quantile(data, q) for q in quantiles]
        actual = sketch.quantiles(quantiles)
        
        # Check accuracy - allow 5% relative error
        for exp, act in zip(expected, actual):
            assert abs(exp - act) / abs(exp) < 0.05
            
    def test_skewed_distribution(self):
        """Test with skewed distribution."""
        # Generate skewed data (lognormal)
        np.random.seed(42)
        data = np.random.lognormal(1, 1, 10000)
        
        # Create sketch
        sketch = MomentSketch(num_moments=15)
        sketch.insert_batch(data)
        
        # Test quantiles
        quantiles = [0.25, 0.5, 0.75]
        actual = sketch.quantiles(quantiles)
        
        # Verify that all quantiles are within the range of data
        assert all(min(data) <= q <= max(data) for q in actual)
        
        # Verify that quantiles are in ascending order
        assert actual[0] <= actual[1] <= actual[2]
            
    def test_bimodal_distribution(self):
        """Test with bimodal distribution."""
        # Generate bimodal data
        np.random.seed(42)
        data1 = np.random.normal(20, 5, 5000)
        data2 = np.random.normal(80, 5, 5000)
        data = np.concatenate([data1, data2])
        
        # Create sketch
        sketch = MomentSketch(num_moments=20)  # More moments for complex distribution
        sketch.insert_batch(data)
        
        # Test quantiles
        quantiles = [0.25, 0.5, 0.75]
        actual = sketch.quantiles(quantiles)
        
        # Bimodal distributions are challenging for moment-based methods
        # Verify that all quantiles are within the range of data
        assert all(min(data) <= q <= max(data) for q in actual)
        
        # Verify that quantiles are in ascending order
        assert actual[0] <= actual[1] <= actual[2]
            
    def test_large_data(self):
        """Test with large dataset."""
        # Generate a large dataset
        np.random.seed(42)
        data = np.random.normal(1000, 200, 100000)
        
        # Time to compute exact quantiles
        import time
        start_time = time.time()
        exact_median = np.median(data)
        exact_time = time.time() - start_time
        
        # Time to compute approximate quantiles
        start_time = time.time()
        sketch = MomentSketch(num_moments=15)
        sketch.insert_batch(data)
        approx_median = sketch.median()
        approx_time = time.time() - start_time
        
        # The approximate method should be faster for large datasets
        # However, this test is not deterministic and could fail on some systems
        # So we just print the times and check accuracy
        print(f"\nExact median: {exact_median}, time: {exact_time:.6f}s")
        print(f"Approx median: {approx_median}, time: {approx_time:.6f}s")
        print(f"Speedup: {exact_time / approx_time:.2f}x")
        
        # Check accuracy - allow 1% relative error
        rel_error = abs(exact_median - approx_median) / abs(exact_median)
        assert rel_error < 0.01, f"Relative error: {rel_error}"
            
    def test_extreme_values(self):
        """Test with extreme values using compression."""
        # Generate data with extreme values
        np.random.seed(42)
        data = np.random.lognormal(0, 5, 10000)  # Will generate some very large values
        
        # Create sketch with compression
        sketch = MomentSketch(num_moments=15, compress_values=True)
        sketch.insert_batch(data)
        
        # For extreme data, verify that we can at least get a result without errors
        approx_median = sketch.median()
        
        # Verify that the value is within the range of data
        assert min(data) <= approx_median <= max(data) or np.isnan(approx_median)
            
    def test_incremental_updates(self):
        """Test incremental updates to the sketch."""
        # Generate data
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 5000)
        data2 = np.random.normal(50, 10, 5000)
        data_combined = np.concatenate([data1, data2])
        
        # Create a sketch and update in batches
        sketch = MomentSketch(num_moments=15)
        sketch.insert_batch(data1)
        sketch.insert_batch(data2)
        
        # Create a sketch with all data at once
        sketch_all = MomentSketch(num_moments=15)
        sketch_all.insert_batch(data_combined)
        
        # The results should be identical
        median1 = sketch.median()
        median2 = sketch_all.median()
        
        assert abs(median1 - median2) < 1e-10
        
    def test_merge(self):
        """Test merging multiple sketches."""
        # Generate data
        np.random.seed(42)
        data1 = np.random.normal(40, 10, 5000)
        data2 = np.random.normal(60, 10, 5000)
        data_combined = np.concatenate([data1, data2])
        
        # Create separate sketches
        sketch1 = MomentSketch(num_moments=15)
        sketch1.insert_batch(data1)
        
        sketch2 = MomentSketch(num_moments=15)
        sketch2.insert_batch(data2)
        
        # Merge sketches
        merged_sketch = MomentSketch(num_moments=15)
        merged_sketch.merge(sketch1)
        merged_sketch.merge(sketch2)
        
        # Create a sketch with all data
        full_sketch = MomentSketch(num_moments=15)
        full_sketch.insert_batch(data_combined)
        
        # The merged sketch should approximate the full sketch
        merged_median = merged_sketch.median()
        full_median = full_sketch.median()
        exact_median = np.median(data_combined)
        
        # Both should be close to the exact median
        assert abs(merged_median - exact_median) / abs(exact_median) < 0.05
        assert abs(full_median - exact_median) / abs(exact_median) < 0.05
        
        # And close to each other
        assert abs(merged_median - full_median) / abs(full_median) < 0.01
        
    def test_serialization(self):
        """Test serialization and deserialization."""
        # Generate data
        np.random.seed(42)
        data = np.random.normal(50, 10, 10000)
        
        # Create and populate a sketch
        sketch = MomentSketch(num_moments=15)
        sketch.insert_batch(data)
        
        # Serialize
        data_dict = sketch.to_dict()
        
        # Deserialize
        restored_sketch = MomentSketch.from_dict(data_dict)
        
        # The results should be identical
        original_median = sketch.median()
        restored_median = restored_sketch.median()
        
        assert abs(original_median - restored_median) < 1e-10
        
        # Check other quantiles
        original_qs = sketch.quantiles([0.25, 0.75])
        restored_qs = restored_sketch.quantiles([0.25, 0.75])
        
        assert all(abs(o - r) < 1e-10 for o, r in zip(original_qs, restored_qs))
        
    @patch('matplotlib.pyplot.figure')
    def test_visualization(self, mock_figure):
        """Test visualization capabilities."""
        # Generate data
        np.random.seed(42)
        data = np.random.normal(50, 10, 1000)
        
        # Create and populate a sketch
        sketch = MomentSketch(num_moments=15)
        sketch.insert_batch(data)
        
        # Test plotting
        sketch.plot_distribution()
        
        # Should have called figure
        mock_figure.assert_called()
        
    def test_empty_sketch(self):
        """Test behavior with empty sketch."""
        sketch = MomentSketch()
        
        # Median of empty sketch should be NaN
        assert np.isnan(sketch.median())
        
        # Quantiles of empty sketch should be NaN
        quantiles = sketch.quantiles([0.5])
        assert len(quantiles) == 1
        assert np.isnan(quantiles[0])
        
    def test_stability_with_outliers(self):
        """Test stability in presence of outliers."""
        # Generate data with outliers
        np.random.seed(42)
        data = np.random.normal(50, 10, 10000)
        data = np.append(data, [1000, -1000])  # Add outliers
        
        # Create sketch with compression to handle outliers better
        sketch = MomentSketch(num_moments=15, compress_values=True)
        sketch.insert_batch(data)
        
        # For data with severe outliers, just verify we can compute a result
        approx_median = sketch.median()
        
        # Verify that the value is within the range of data
        assert min(data) <= approx_median <= max(data)
        
    def test_performance_vs_accuracy(self):
        """Test tradeoff between performance and accuracy."""
        # Generate data
        np.random.seed(42)
        data = np.random.normal(50, 10, 10000)
        
        # Test with different numbers of moments
        moments_to_test = [5, 10, 15, 20]
        results = []
        
        for moments in moments_to_test:
            import time
            
            # Time to build and query
            start_time = time.time()
            sketch = MomentSketch(num_moments=moments)
            sketch.insert_batch(data)
            median = sketch.median()
            elapsed = time.time() - start_time
            
            # Calculate error
            exact_median = np.median(data)
            rel_error = abs(exact_median - median) / abs(exact_median)
            
            results.append((moments, elapsed, rel_error))
            
        # Print results
        print("\nPerformance vs. Accuracy:")
        print(f"{'Moments':>10} | {'Time (s)':>10} | {'Rel. Error':>10}")
        print("-" * 34)
        for moments, elapsed, rel_error in results:
            print(f"{moments:10d} | {elapsed:10.6f} | {rel_error:10.6f}")
            
        # Higher moments should give better accuracy but take longer
        # We don't make assertions here since the exact performance depends on the system 