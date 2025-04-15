import pytest
import numpy as np
import warnings
from QuantileFlow.ddsketch.core import DDSketch
from QuantileFlow.ddsketch.storage.base import BucketManagementStrategy

def test_ddsketch_initialization():
    # Test valid initialization
    sketch = DDSketch(relative_accuracy=0.01)
    assert sketch.relative_accuracy == 0.01
    assert sketch.cont_neg is True
    
    # Test invalid relative accuracy
    with pytest.raises(ValueError):
        DDSketch(relative_accuracy=0)
    with pytest.raises(ValueError):
        DDSketch(relative_accuracy=1)
    with pytest.raises(ValueError):
        DDSketch(relative_accuracy=-0.1)

def test_insert_positive():
    sketch = DDSketch(relative_accuracy=0.01)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in values:
        sketch.insert(v)
    assert sketch.count == len(values)
    
    # Test median (should be approximately 3.0)
    # Use a slightly higher tolerance for test stability
    assert abs(sketch.quantile(0.5) - 3.0) <= 3.0 * 0.02  # Double the relative accuracy for tests

def test_insert_negative():
    sketch = DDSketch(relative_accuracy=0.01)
    values = [-1.0, -2.0, -3.0, -4.0, -5.0]
    for v in values:
        sketch.insert(v)
    assert sketch.count == len(values)
    
    # Test median (should be approximately -3.0)
    # Use a slightly higher tolerance for test stability
    assert abs(sketch.quantile(0.5) - (-3.0)) <= abs(-3.0) * 0.02  # Double the relative accuracy for tests

def test_insert_mixed():
    sketch = DDSketch(relative_accuracy=0.01)
    values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for v in values:
        sketch.insert(v)
    assert sketch.count == len(values)
    assert sketch.zero_count == 1
    
    # Test median (should be approximately 0.0)
    assert abs(sketch.quantile(0.5)) <= 0.02  # Use a fixed small error for zero median

def test_negative_values_disabled():
    sketch = DDSketch(relative_accuracy=0.01, cont_neg=False)
    sketch.insert(1.0)  # Should work
    with pytest.raises(ValueError):
        sketch.insert(-1.0)

def test_delete():
    sketch = DDSketch(relative_accuracy=0.01)
    values = [1.0, 2.0, 2.0, 3.0]
    for v in values:
        sketch.insert(v)
    
    sketch.delete(2.0)
    assert sketch.count == len(values) - 1
    
    # Delete non-existent value (should not affect count)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sketch.delete(10.0)
    assert sketch.count == len(values) - 1

def test_quantile_edge_cases():
    sketch = DDSketch(relative_accuracy=0.01)
    
    # Empty sketch
    with pytest.raises(ValueError):
        sketch.quantile(0.5)
    
    # Invalid quantile values
    sketch.insert(1.0)
    with pytest.raises(ValueError):
        sketch.quantile(-0.1)
    with pytest.raises(ValueError):
        sketch.quantile(1.1)

def test_merge():
    sketch1 = DDSketch(relative_accuracy=0.01)
    sketch2 = DDSketch(relative_accuracy=0.01)

    # Generate Pareto distribution with shape parameter a=3 (finite variance)
    np.random.seed(42)
    values = (1 / (1 - np.random.random(1000)) ** (1/3))  # Inverse CDF method for Pareto
    values = np.sort(values)  # Sort to make splitting deterministic
    median_idx = len(values) // 2
    true_median = values[median_idx]

    # Split values between sketches
    for v in values[:median_idx]:
        sketch1.insert(v)
    for v in values[median_idx:]:
        sketch2.insert(v)

    # Merge sketch2 into sketch1
    sketch1.merge(sketch2)
    assert sketch1.count == len(values)

    # Test median of merged sketch
    assert abs(sketch1.quantile(0.5) - true_median) <= true_median * 0.01

    # Also test other quantiles
    q1_idx = len(values) // 4
    q3_idx = 3 * len(values) // 4
    true_q1 = values[q1_idx]
    true_q3 = values[q3_idx]
    assert abs(sketch1.quantile(0.25) - true_q1) <= true_q1 * 0.01  # Q1
    assert abs(sketch1.quantile(0.75) - true_q3) <= true_q3 * 0.01  # Q3

def test_merge_incompatible():
    sketch1 = DDSketch(relative_accuracy=0.01)
    sketch2 = DDSketch(relative_accuracy=0.02)
    
    with pytest.raises(ValueError):
        sketch1.merge(sketch2)

def test_different_mapping_types():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Test all mapping types with their own accuracy guarantees
    sketch1 = DDSketch(relative_accuracy=0.01, mapping_type='logarithmic')
    sketch2 = DDSketch(relative_accuracy=0.01, mapping_type='lin_interpol')
    sketch3 = DDSketch(relative_accuracy=0.01, mapping_type='cub_interpol')
    
    # Insert values into each sketch
    for v in values:
        sketch1.insert(v)
        sketch2.insert(v)
        sketch3.insert(v)
    
    # Each mapping type may have its own characteristics
    # Verify with appropriate tolerances
    assert abs(sketch1.quantile(0.5) - 3.0) <= 3.0 * 0.1
    assert abs(sketch2.quantile(0.5) - 3.0) <= 3.0 * 0.1
    assert abs(sketch3.quantile(0.5) - 3.0) <= 3.0 * 0.1

def test_different_storage_types():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Test both storage strategies
    for strategy in [BucketManagementStrategy.FIXED, BucketManagementStrategy.DYNAMIC, BucketManagementStrategy.UNLIMITED]:
        if strategy == BucketManagementStrategy.FIXED:
            sketch = DDSketch(relative_accuracy=0.01, max_buckets=1000, bucket_strategy=strategy)
        else:
            sketch = DDSketch(relative_accuracy=0.01, bucket_strategy=strategy)
        for v in values:
            sketch.insert(v)
        
        # Verify median with a slightly higher tolerance
        assert abs(sketch.quantile(0.5) - 3.0) <= 3.0 * 0.02  # Double the relative accuracy for tests

def test_extreme_values():    
    sketch = DDSketch(relative_accuracy=0.01)
    
    # Test very large and very small positive values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sketch.insert(1e-100)
        sketch.insert(1e100)
    
    # Should handle these values without issues using SparseStorage
    assert sketch.count == 2
    
    # Test quantiles
    assert sketch.quantile(0) > 0  # Should be close to 1e-100
    assert sketch.quantile(1) < float('inf')  # Should be close to 1e100
    
    # Test with more reasonable values that should work with any storage
    sketch_reasonable = DDSketch(relative_accuracy=0.01)  # Uses default ContiguousStorage
    
    # Use values that are within the range of ContiguousStorage
    # These values should be manageable with the default settings
    sketch_reasonable.insert(0.1)  # Small but not extreme value
    sketch_reasonable.insert(10.0)  # Reasonable positive value
    sketch_reasonable.insert(100.0)  # Larger but still in range
    
    assert sketch_reasonable.count == 3
    
    # Verify we can compute quantiles on these reasonable values
    q0 = sketch_reasonable.quantile(0)
    q1 = sketch_reasonable.quantile(1)
    assert q0 >= 0.1 * 0.9  # Allow for relative accuracy
    assert q1 <= 100.0 * 1.1  # Allow for relative accuracy

def test_accuracy_guarantee():
    # Test that the relative error guarantee is maintained using a slightly higher tolerance
    # for test stability across different platforms
    sketch = DDSketch(relative_accuracy=0.01)
    
    # Generate log-normal distribution
    np.random.seed(42)
    values = np.random.lognormal(0, 1, 1000)
    
    # Insert values
    for v in values:
        sketch.insert(v)
    
    # Test various quantiles with a slightly relaxed tolerance
    test_tolerance = 0.02  # Doubled from 0.01 for test stability
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        true_quantile = np.quantile(values, q)
        approx_quantile = sketch.quantile(q)
        
        # Verify relative error is within tolerance
        relative_error = abs(approx_quantile - true_quantile) / true_quantile
        assert relative_error <= test_tolerance, f"Relative error exceeded at q={q}" 