import pytest
from QuantileFlow.ddsketch.storage.base import BucketManagementStrategy
from QuantileFlow.ddsketch.storage.contiguous import ContiguousStorage
from QuantileFlow.ddsketch.storage.sparse import SparseStorage
import warnings
import numpy as np

@pytest.fixture(params=[ContiguousStorage, SparseStorage])
def storage_class(request):
    return request.param

@pytest.fixture(params=[32, 64, 128])
def max_buckets(request):
    return request.param

@pytest.fixture(params=[
    BucketManagementStrategy.FIXED,
    BucketManagementStrategy.DYNAMIC,
    BucketManagementStrategy.UNLIMITED
])
def bucket_strategy(request):
    return request.param

def test_storage_initialization(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage:
        if bucket_strategy != BucketManagementStrategy.FIXED:
            pytest.skip("ContiguousStorage only supports FIXED strategy")
        storage = storage_class(max_buckets)
    else:
        if bucket_strategy == BucketManagementStrategy.FIXED:
            storage = storage_class(max_buckets, bucket_strategy)
        else:
            storage = storage_class(strategy=bucket_strategy)
    
    # Check max_buckets based on strategy
    if bucket_strategy == BucketManagementStrategy.FIXED:
        expected_max_buckets = max_buckets
    elif bucket_strategy == BucketManagementStrategy.UNLIMITED:
        expected_max_buckets = -1
    else:  # DYNAMIC
        expected_max_buckets = 32  # Initial value for dynamic strategy
        
    assert storage.max_buckets == expected_max_buckets
    if hasattr(storage, 'strategy'):
        assert storage.strategy == bucket_strategy

def test_add_and_get_count(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage:
        if bucket_strategy != BucketManagementStrategy.FIXED:
            pytest.skip("ContiguousStorage only supports FIXED strategy")
        storage = storage_class(max_buckets)
    else:
        if bucket_strategy == BucketManagementStrategy.FIXED:
            storage = storage_class(max_buckets, bucket_strategy)
        else:
            storage = storage_class(strategy=bucket_strategy)
    
    # Add counts to some buckets
    test_buckets = {0: 1, 5: 3, 10: 2}
    for bucket, count in test_buckets.items():
        for _ in range(count):
            storage.add(bucket)
    
    # Verify counts
    for bucket, expected_count in test_buckets.items():
        count = storage.get_count(bucket)
        assert count == expected_count
    
    # Test non-existent bucket
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert storage.get_count(999) == 0

def test_remove(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage:
        if bucket_strategy != BucketManagementStrategy.FIXED:
            pytest.skip("ContiguousStorage only supports FIXED strategy")
        storage = storage_class(max_buckets)
    else:
        if bucket_strategy == BucketManagementStrategy.FIXED:
            storage = storage_class(max_buckets, bucket_strategy)
        else:
            storage = storage_class(strategy=bucket_strategy)
    
    # Add and remove counts
    bucket_idx = 5
    storage.add(bucket_idx)
    storage.add(bucket_idx)
    storage.remove(bucket_idx)
    
    assert storage.get_count(bucket_idx) == 1
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Remove from empty bucket
        storage.remove(999)  # Should not raise error
        assert storage.get_count(999) == 0

def test_merge(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage:
        if bucket_strategy != BucketManagementStrategy.FIXED:
            pytest.skip("ContiguousStorage only supports FIXED strategy")
        storage1 = storage_class(max_buckets)
        storage2 = storage_class(max_buckets)
    else:
        if bucket_strategy == BucketManagementStrategy.FIXED:
            storage1 = storage_class(max_buckets, bucket_strategy)
            storage2 = storage_class(max_buckets, bucket_strategy)
        else:
            storage1 = storage_class(strategy=bucket_strategy)
            storage2 = storage_class(strategy=bucket_strategy)
    
    # Add counts to both storages
    storage1.add(0)
    storage1.add(5)
    storage2.add(5)
    storage2.add(10)
    
    # Merge storage2 into storage1
    storage1.merge(storage2)
    
    assert storage1.get_count(0) == 1
    assert storage1.get_count(5) == 2
    assert storage1.get_count(10) == 1

def test_bucket_limit(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage:
        if bucket_strategy != BucketManagementStrategy.FIXED:
            pytest.skip("ContiguousStorage only supports FIXED strategy")
        storage = storage_class(max_buckets)
    else:
        if bucket_strategy == BucketManagementStrategy.FIXED:
            storage = storage_class(max_buckets, bucket_strategy)
        else:
            storage = storage_class(strategy=bucket_strategy)
    
    # Add more buckets than max_buckets
    for i in range(max_buckets + 10):
        storage.add(i)
    
    # Count total non-zero buckets
    if storage_class == ContiguousStorage:
        # For ContiguousStorage, count physical storage buckets
        non_zero_buckets = sum(1 for count in storage.counts if count > 0)
    else:
        # For other storage types, count logical buckets
        non_zero_buckets = sum(1 for i in range(max_buckets + 10) if storage.get_count(i) > 0)
    
    if bucket_strategy == BucketManagementStrategy.FIXED:
        # FIXED strategy should respect max_buckets exactly
        assert non_zero_buckets <= max_buckets
    elif bucket_strategy == BucketManagementStrategy.DYNAMIC:
        # DYNAMIC strategy should use logarithmic growth
        # For n values, expect roughly 100*log10(n+1) buckets
        expected_buckets = int(100 * np.log10(storage.total_count + 1))
        assert non_zero_buckets <= expected_buckets
    else:
        # UNLIMITED strategy should keep all buckets
        assert non_zero_buckets == max_buckets + 10

def test_contiguous_storage_specific():
    """Test ContiguousStorage-specific features"""
    storage = ContiguousStorage(max_buckets=32)
    
    # Initialize storage by adding a value first
    storage.add(0)  # This will set min_index and max_index
    
    # Test bucket range limits
    # Since we can only use non-negative values, we'll test from 0 to max_buckets-1
    max_bucket = storage.max_buckets - 1
    
    # Add some values to ensure we have enough buckets for testing
    for i in range(3):  # Add a few values to have multiple non-zero buckets
        storage.add(i)
    
    # Add value at maximum allowed index
    storage.add(max_bucket)
    
    # Verify counts
    assert storage.get_count(0) == 2  # One from initial add, one from the loop
    assert storage.get_count(max_bucket) == 1
    
    # Test out of range buckets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert storage.get_count(max_bucket + 1) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert storage.get_count(-1) == 0

def test_sparse_storage_specific():
    """Test SparseStorage-specific features"""
    storage = SparseStorage(strategy=BucketManagementStrategy.DYNAMIC)
    
    # Test dynamic bucket management behavior
    # Add many buckets to force dynamic resizing
    for i in range(100):
        storage.add(i)
    
    # Verify that buckets are managed according to dynamic strategy
    # For n values, expect roughly 100*log10(n+1) buckets
    expected_buckets = int(100 * np.log10(storage.total_count + 1))
    assert len(storage.counts) <= expected_buckets
    
    # Test that we can still add to existing buckets
    last_bucket = max(storage.counts.keys())
    initial_count = storage.get_count(last_bucket)
    storage.add(last_bucket)
    assert storage.get_count(last_bucket) == initial_count + 1

def test_storage_clear(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage:
        if bucket_strategy != BucketManagementStrategy.FIXED:
            pytest.skip("ContiguousStorage only supports FIXED strategy")
        storage = storage_class(max_buckets)
    else:
        if bucket_strategy == BucketManagementStrategy.FIXED:
            storage = storage_class(max_buckets, bucket_strategy)
        else:
            storage = storage_class(strategy=bucket_strategy)
    
    # Add some counts
    storage.add(0)
    storage.add(5)
    storage.add(10)
    
    # Clear the storage
    if hasattr(storage, 'clear'):
        storage.clear()
        
        # Verify all counts are zero
        assert storage.get_count(0) == 0
        assert storage.get_count(5) == 0
        assert storage.get_count(10) == 0

def test_negative_buckets(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage:
        if bucket_strategy != BucketManagementStrategy.FIXED:
            pytest.skip("ContiguousStorage only supports FIXED strategy")
        storage = storage_class(max_buckets)
    else:
        if bucket_strategy == BucketManagementStrategy.FIXED:
            storage = storage_class(max_buckets, bucket_strategy)
        else:
            storage = storage_class(strategy=bucket_strategy)
    
    # Test adding negative bucket indices
    if storage_class == SparseStorage:
        storage.add(-1)
        storage.add(-5)
        assert storage.get_count(-1) == 1
        assert storage.get_count(-5) == 1
    else:
        # ContiguousStorage should handle negative indices within its range
        min_bucket = -(max_buckets // 2)
        storage.add(min_bucket)
        assert storage.get_count(min_bucket) == 1

def test_unlimited_strategy_warning():
    """Test that a warning is raised when max_buckets is provided with UNLIMITED strategy."""
    with pytest.warns(UserWarning, match="max_buckets=100 was provided but will be ignored"):
        SparseStorage(max_buckets=100, strategy=BucketManagementStrategy.UNLIMITED)
    
    # No warning should be raised when using default max_buckets
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        SparseStorage(strategy=BucketManagementStrategy.UNLIMITED)  # Just create without assigning 