"""Sparse storage implementation for DDSketch using dictionary."""

from typing import Dict
from .base import Storage, BucketManagementStrategy

class SparseStorage(Storage):
    """
    Sparse storage for DDSketch using dictionary.
    
    This implementation is memory-efficient for sparse data where bucket indices
    are widely spread. It only stores non-zero counts and has no constraints
    on the range of bucket indices.
    """
    
    def __init__(self, max_buckets: int = 2048,
                 strategy: BucketManagementStrategy = BucketManagementStrategy.FIXED):
        """
        Initialize sparse storage.
        
        Args:
            max_buckets: Maximum number of buckets (default 2048).
                        Ignored if strategy is UNLIMITED.
            strategy: Bucket management strategy (default FIXED).
        """
        super().__init__(max_buckets, strategy)
        self.counts: Dict[int, int] = {}
        self.min_index = None  # Minimum bucket index seen
        self.max_index = None  # Maximum bucket index seen
    
    def add(self, bucket_index: int, count: int = 1):
        """
        Add count to bucket_index.
        
        Args:
            bucket_index: The bucket index to add to.
            count: The count to add (default 1).
        """
        if count <= 0:
            return
            
        self.counts[bucket_index] = self.counts.get(bucket_index, 0) + count
        self.total_count += count
        
        # Update min and max indices
        if self.min_index is None or bucket_index < self.min_index:
            self.min_index = bucket_index
        if self.max_index is None or bucket_index > self.max_index:
            self.max_index = bucket_index
        
        if self.strategy == BucketManagementStrategy.DYNAMIC:
            self._update_dynamic_limit()
            
        if (self.strategy != BucketManagementStrategy.UNLIMITED and 
            len(self.counts) > self.max_buckets):
            self.collapse_smallest_buckets()
    
    def remove(self, bucket_index: int, count: int = 1) -> bool:
        """
        Remove count from bucket_index.
        
        Args:
            bucket_index: The bucket index to remove from.
            count: The count to remove (default 1).
            
        Returns:
            bool: True if any value was actually removed, False otherwise.
        """
        if count <= 0 or bucket_index not in self.counts:
            return False
            
        self.counts[bucket_index] = max(0, self.counts[bucket_index] - count)
        self.total_count = max(0, self.total_count - count)
        
        if self.counts[bucket_index] == 0:
            del self.counts[bucket_index]
            # Update min and max indices if needed
            if bucket_index == self.min_index or bucket_index == self.max_index:
                if not self.counts:  # If no buckets left
                    self.min_index = None
                    self.max_index = None
                else:
                    # Recalculate min and max
                    self.min_index = min(self.counts.keys())
                    self.max_index = max(self.counts.keys())
            
        return True
    
    def get_count(self, bucket_index: int) -> int:
        """
        Get count for bucket_index.
        
        Args:
            bucket_index: The bucket index to get count for.
            
        Returns:
            The count at the specified bucket index.
        """
        return self.counts.get(bucket_index, 0)
    
    def merge(self, other: 'SparseStorage'):
        """
        Merge another storage into this one.
        
        Args:
            other: Another SparseStorage instance to merge with this one.
        """
        for idx, count in other.counts.items():
            self.add(idx, count)
            
    def collapse_smallest_buckets(self):
        """Collapse the two buckets with smallest counts to maintain max bucket limit."""
        if len(self.counts) < 2:
            return
            
        # Find two buckets with smallest counts
        sorted_buckets = sorted(self.counts.items(), key=lambda x: (x[1], x[0]))
        i0, count0 = sorted_buckets[0]
        i1, count1 = sorted_buckets[1]
        
        # Ensure i0 is the smaller index for consistency
        if i1 < i0:
            i0, i1 = i1, i0
        
        # Merge buckets
        self.counts[i1] += self.counts[i0]
        del self.counts[i0] 