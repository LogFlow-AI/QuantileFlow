"""Contiguous array storage implementation for DDSketch using circular buffer."""

import numpy as np
import warnings
from .base import Storage, BucketManagementStrategy

class ContiguousStorage(Storage):
    """
    Contiguous array storage for DDSketch using a circular buffer.
    
    Uses a bucket mapping scheme where:
    bucket_array_index = (bucket_index - min_bucket_index + arr_index_of_min_bucket) % num_buckets
    
    Implements collapsing strategy where:
    - If inserting below min: collapse if range too large, otherwise adjust min
    - If inserting above max: collapse lowest buckets to make room
    """

    __slots__ = ('total_count', 'counts', 'min_index', 'max_index', 
                 'num_buckets', 'arr_index_of_min_bucket', 'collapse_count',
                 'max_buckets', 'bucket_mask', 'strategy')
    
    def __init__(self, max_buckets: int = 2048):
        """
        Initialize contiguous storage.
        
        Args:
            max_buckets: Maximum number of buckets (default 2048).
        """
        if max_buckets <= 0:
            raise ValueError("max_buckets must be positive for ContiguousStorage")
        super().__init__(max_buckets, BucketManagementStrategy.FIXED)
        self.total_count = 0
        self.counts = np.zeros(max_buckets, dtype=np.int64)
        self.max_buckets = max_buckets
        self.min_index = None  # Minimum bucket index seen
        self.max_index = None  # Maximum bucket index seen
        self.num_buckets = 0   # Number of non-zero buckets
        self.arr_index_of_min_bucket = 0  # Array index where min bucket is stored
        self.collapse_count = 0  # Number of times buckets have been collapsed
    
    def add(self, bucket_index: int, count: int = 1):
        """
        Add count to bucket_index using new collapsing strategy.
        
        Args:
            bucket_index: The bucket index to add to.
            count: The count to add (default 1).
        """
            
        if self.min_index is None:
            # First insertion
            self.min_index = bucket_index
            self.max_index = bucket_index
            self.counts[0] = count
            self.num_buckets = 1
            self.arr_index_of_min_bucket = 0
        else:
            if bucket_index < self.min_index:
                new_range = self.max_index - bucket_index + 1
                # Handle insertion below current minimum
                if new_range > self.max_buckets:
                    # Range too large, collapse into min bucket
                    pos = (self.arr_index_of_min_bucket) % self.max_buckets
                    self.counts[pos] += count
                    self.collapse_count += 1
                else:
                    # Update min and place value
                    shift = self.min_index - bucket_index
                    self.min_index = bucket_index
                    self.arr_index_of_min_bucket = self.arr_index_of_min_bucket - shift
                    pos = (bucket_index - self.min_index + self.arr_index_of_min_bucket) % self.max_buckets
                    self.counts[pos] = count
                    self.num_buckets += 1
                    
            elif bucket_index > self.max_index:
                new_range = bucket_index - self.min_index + 1
                if new_range > self.max_buckets:
                    # Handle insertion above current maximum
                    buckets_to_collapse = bucket_index - self.max_index
                    # Collapse lowest buckets
                    collapse_sum = 0
                    for i in range(buckets_to_collapse):
                        if i >= self.max_index - self.min_index + 1:
                            warnings.warn("Collapsing all buckets in the sketch. "
                                          "Range is too large to be contained by the buckets allocated, "
                                          "and you should increase max_buckets.", UserWarning)
                            break
                        pos = i + self.arr_index_of_min_bucket
                        collapse_sum += self.counts[pos]
                        self.counts[pos] = 0
                        
                    # Add collapsed values to new min bucket
                    new_min = self.min_index + buckets_to_collapse
                    new_min_pos = (buckets_to_collapse + self.arr_index_of_min_bucket) % self.max_buckets
                    self.counts[new_min_pos] += collapse_sum
                    
                    # Update tracking variables
                    self.min_index = new_min
                    self.arr_index_of_min_bucket = new_min_pos
                    self.collapse_count += buckets_to_collapse
                
                # Place new value
                self.max_index = bucket_index
                pos = (bucket_index - self.min_index + self.arr_index_of_min_bucket) % self.max_buckets
                was_zero = self.counts[pos] == 0
                self.counts[pos] += count
                if was_zero:
                    self.num_buckets += 1
            else:
                # Normal insertion within current range
                pos = (bucket_index - self.min_index + self.arr_index_of_min_bucket) % self.max_buckets
                was_zero = self.counts[pos] == 0
                self.counts[pos] += count
                if was_zero:
                    self.num_buckets += 1
                    
        self.total_count += count
    
    def remove(self, bucket_index: int, count: int = 1) -> bool:
        """
        Remove count from bucket_index.
        
        Args:
            bucket_index: The bucket index to remove from.
            count: The count to remove (default 1).
            
        Returns:
            bool: True if any value was actually removed, False otherwise.
        """
        if count <= 0 or self.min_index is None:
            return False
            
        if self.min_index <= bucket_index <= self.max_index:
            pos = (bucket_index - self.min_index + self.arr_index_of_min_bucket) % self.max_buckets
            old_count = self.counts[pos]
            
            if old_count == 0:
                return False
                
            self.counts[pos] = max(0, old_count - count)
            self.total_count = max(0, self.total_count - count)
            
            if old_count > 0 and self.counts[pos] == 0:
                self.num_buckets -= 1
                if self.num_buckets == 0:
                    self.min_index = None
                    self.max_index = None
                elif bucket_index == self.min_index:
                    # Find new minimum index
                    for i in range(self.max_index - self.min_index + 1):
                        pos = (self.arr_index_of_min_bucket + i) % self.max_buckets
                        if self.counts[pos] > 0:
                            self.min_index += i
                            self.arr_index_of_min_bucket = pos
                            break
                elif bucket_index == self.max_index:
                    # Find new maximum index
                    for i in range(self.max_index - self.min_index + 1):
                        pos = (self.arr_index_of_min_bucket + (self.max_index - self.min_index - i)) % self.max_buckets
                        if self.counts[pos] > 0:
                            self.max_index -= i
                            break
            return True
        else:
            warnings.warn("Removing count from non-existent bucket. "
                              "Bucket index is out of range.", UserWarning)
            return False
    
    def get_count(self, bucket_index: int) -> int:
        """
        Get count for bucket_index.
        
        Args:
            bucket_index: The bucket index to get count for.
            
        Returns:
            The count at the specified bucket index.
        """
        if self.min_index is None or bucket_index < self.min_index or bucket_index > self.max_index:
            return 0
        pos = (bucket_index - self.min_index + self.arr_index_of_min_bucket) % self.max_buckets
        return int(self.counts[pos])
    
    def merge(self, other: 'ContiguousStorage'):
        """
        Merge another storage into this one.
        
        Args:
            other: Another ContiguousStorage instance to merge with this one.
        """
        if other.min_index is None:
            return
            
        # Add each non-zero bucket
        for i in range(other.max_index - other.min_index + 1):
            pos = (other.arr_index_of_min_bucket + i) % other.max_buckets
            if other.counts[pos] > 0:
                bucket_index = other.min_index + i
                self.add(bucket_index, int(other.counts[pos]))
    