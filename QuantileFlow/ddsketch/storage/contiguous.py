"""Contiguous array storage implementation for DDSketch using offset-based indexing.

Optimized for high throughput by using Python lists instead of numpy arrays
and adopting Datadog's chunk-based dynamic growth pattern.
"""

import math
import warnings
from .base import Storage, BucketManagementStrategy


# Chunk size for dynamic growth (matches Datadog's default)
CHUNK_SIZE = 128


class ContiguousStorage(Storage):
    """
    Contiguous storage for DDSketch using offset-based direct indexing.
    
    Uses a simple offset scheme where:
    array_index = bucket_index - offset
    
    This implementation is optimized for high throughput by:
    - Using Python lists instead of numpy arrays for faster scalar operations
    - Growing dynamically in chunks rather than pre-allocating
    - Minimizing attribute access in the hot path
    
    Implements collapsing strategy where:
    - If inserting below min: shift array or collapse if range too large
    - If inserting above max: collapse lowest buckets to make room
    """

    __slots__ = ('count', 'bins', 'min_key', 'max_key', 
                 'offset', 'collapse_count', 'bin_limit', 
                 'chunk_size', 'is_collapsed')
    
    def __init__(self, bin_limit: int = 2048, chunk_size: int = CHUNK_SIZE, max_buckets: int = None):
        """
        Initialize contiguous storage.
        
        Args:
            bin_limit: Maximum number of bins (default 2048).
            chunk_size: Size of chunks for dynamic growth (default 128).
            max_buckets: Alias for bin_limit for API compatibility.
        """
        # Support max_buckets as alias for bin_limit
        if max_buckets is not None:
            bin_limit = max_buckets
        
        if bin_limit <= 0:
            raise ValueError("bin_limit must be positive for ContiguousStorage")
        
        # Don't call super().__init__ to avoid overhead - inline what we need
        self.count = 0.0  # Use float like Datadog for weighted values
        self.bins = []  # Start empty, grow dynamically
        self.bin_limit = bin_limit
        self.chunk_size = chunk_size
        self.min_key = None  # Will use special infinity values
        self.max_key = None
        self.offset = 0
        self.collapse_count = 0
        self.is_collapsed = False
    
    @property
    def total_count(self):
        """Alias for count to maintain API compatibility."""
        return self.count
    
    @total_count.setter
    def total_count(self, value):
        self.count = value
    
    @property
    def min_index(self):
        """Alias for min_key to maintain API compatibility."""
        return self.min_key
    
    @property
    def max_index(self):
        """Alias for max_key to maintain API compatibility."""
        return self.max_key
    
    @property
    def max_buckets(self):
        """Alias for bin_limit to maintain API compatibility."""
        return self.bin_limit
    
    @property
    def num_buckets(self):
        """Lazily compute the number of non-zero buckets."""
        if not self.bins:
            return 0
        return sum(1 for b in self.bins if b > 0)
    
    @property
    def counts(self):
        """Return bins for API compatibility (used by some code paths)."""
        return self.bins
    
    def length(self):
        """Return the number of bins."""
        return len(self.bins)
    
    def _get_new_length(self, new_min_key, new_max_key):
        """Calculate new length needed, respecting bin_limit."""
        desired_length = new_max_key - new_min_key + 1
        chunk_length = self.chunk_size * int(math.ceil(desired_length / self.chunk_size))
        return min(chunk_length, self.bin_limit)
    
    def add(self, key, weight=1.0):
        """
        Add weight to the bin at key.
        
        Args:
            key: The bucket index to add to.
            weight: The weight to add (default 1.0).
        """
        idx = self._get_index(key)
        self.bins[idx] += weight
        self.count += weight
    
    def _get_index(self, key):
        """Calculate the bin index for the key, extending the range if necessary."""
        if self.min_key is None:
            # First insertion
            self._extend_range(key)
        elif key < self.min_key:
            if self.is_collapsed:
                return 0
            self._extend_range(key)
            if self.is_collapsed:
                return 0
        elif key > self.max_key:
            self._extend_range(key)
        
        return key - self.offset
    
    def _extend_range(self, key, second_key=None):
        """Grow the bins as necessary and adjust."""
        if second_key is None:
            second_key = key
        
        if self.min_key is None:
            new_min_key = min(key, second_key)
            new_max_key = max(key, second_key)
        else:
            new_min_key = min(key, second_key, self.min_key)
            new_max_key = max(key, second_key, self.max_key)
        
        if self.length() == 0:
            # Initialize bins
            new_length = self._get_new_length(new_min_key, new_max_key)
            self.bins = [0.0] * new_length
            self.offset = new_min_key
            self._adjust(new_min_key, new_max_key)
        elif new_min_key >= self.min_key and new_max_key < self.offset + self.length():
            # No need to change the range; just update min/max keys
            self.min_key = new_min_key
            self.max_key = new_max_key
        else:
            # Grow the bins
            new_length = self._get_new_length(new_min_key, new_max_key)
            if new_length > self.length():
                self.bins.extend([0.0] * (new_length - self.length()))
            self._adjust(new_min_key, new_max_key)
    
    def _adjust(self, new_min_key, new_max_key):
        """
        Adjust the bins, the offset, the min_key, and max_key.
        Collapse to the left if necessary (lowest bins collapsed).
        """
        if new_max_key - new_min_key + 1 > self.length():
            # The range of keys is too wide, the lowest bins need to be collapsed
            new_min_key = new_max_key - self.length() + 1
            
            if self.min_key is None or new_min_key >= self.max_key:
                # Put everything in the first bin
                self.offset = new_min_key
                self.min_key = new_min_key
                self.bins[:] = [0.0] * self.length()
                self.bins[0] = self.count
            else:
                shift = self.offset - new_min_key
                if shift < 0:
                    collapse_start_index = self.min_key - self.offset
                    collapse_end_index = new_min_key - self.offset
                    collapsed_count = sum(self.bins[collapse_start_index:collapse_end_index])
                    self.bins[collapse_start_index:collapse_end_index] = [0.0] * (new_min_key - self.min_key)
                    self.bins[collapse_end_index] += collapsed_count
                    self.min_key = new_min_key
                    self._shift_bins(shift)
                else:
                    self.min_key = new_min_key
                    self._shift_bins(shift)
            
            self.max_key = new_max_key
            self.is_collapsed = True
            self.collapse_count += 1
        else:
            self._center_bins(new_min_key, new_max_key)
            self.min_key = new_min_key
            self.max_key = new_max_key
    
    def _shift_bins(self, shift):
        """Shift the bins; this changes the offset."""
        if shift > 0:
            self.bins = self.bins[:-shift] if shift < len(self.bins) else []
            self.bins[:0] = [0.0] * shift
        elif shift < 0:
            self.bins = self.bins[abs(shift):]
            self.bins.extend([0.0] * abs(shift))
        self.offset -= shift
    
    def _center_bins(self, new_min_key, new_max_key):
        """Center the bins; this changes the offset."""
        middle_key = new_min_key + (new_max_key - new_min_key + 1) // 2
        self._shift_bins(self.offset + self.length() // 2 - middle_key)
    
    def remove(self, bucket_index: int, count: int = 1) -> bool:
        """
        Remove count from bucket_index.
        
        Args:
            bucket_index: The bucket index to remove from.
            count: The count to remove (default 1).
            
        Returns:
            bool: True if any value was actually removed, False otherwise.
        """
        if count <= 0 or self.min_key is None:
            return False
        
        if self.min_key <= bucket_index <= self.max_key:
            pos = bucket_index - self.offset
            if pos < 0 or pos >= len(self.bins):
                return False
            
            old_count = self.bins[pos]
            if old_count == 0:
                return False
            
            self.bins[pos] = max(0, old_count - count)
            self.count = max(0, self.count - count)
            
            # Update min/max keys if we emptied a boundary bucket
            if old_count > 0 and self.bins[pos] == 0:
                if bucket_index == self.min_key:
                    # Find new minimum
                    for i in range(len(self.bins)):
                        if self.bins[i] > 0:
                            self.min_key = self.offset + i
                            break
                    else:
                        self.min_key = None
                        self.max_key = None
                elif bucket_index == self.max_key:
                    # Find new maximum
                    for i in range(len(self.bins) - 1, -1, -1):
                        if self.bins[i] > 0:
                            self.max_key = self.offset + i
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
        if self.min_key is None or bucket_index < self.min_key or bucket_index > self.max_key:
            return 0
        pos = bucket_index - self.offset
        if pos < 0 or pos >= len(self.bins):
            return 0
        return int(self.bins[pos])
    
    def key_at_rank(self, rank, lower=True):
        """
        Return the key for the value at given rank.
        
        This method is compatible with Datadog's interface.
        
        Args:
            rank: The rank to find.
            lower: If True, return key where running_count > rank.
                   If False, return key where running_count >= rank + 1.
        
        Returns:
            The key at the specified rank.
        """
        running_ct = 0.0
        for i, bin_ct in enumerate(self.bins):
            running_ct += bin_ct
            if (lower and running_ct > rank) or (not lower and running_ct >= rank + 1):
                return i + self.offset
        
        return self.max_key if self.max_key is not None else 0
    
    def merge(self, other: 'ContiguousStorage'):
        """
        Merge another storage into this one.
        
        Args:
            other: Another ContiguousStorage instance to merge with this one.
        """
        if other.count == 0:
            return
        
        if self.count == 0:
            self.copy(other)
            return
        
        if other.min_key < self.min_key or other.max_key > self.max_key:
            self._extend_range(other.min_key, other.max_key)
        
        for key in range(other.min_key, other.max_key + 1):
            other_idx = key - other.offset
            if 0 <= other_idx < len(other.bins):
                self_idx = key - self.offset
                if 0 <= self_idx < len(self.bins):
                    self.bins[self_idx] += other.bins[other_idx]
        
        self.count += other.count
    
    def copy(self, store: 'ContiguousStorage'):
        """Copy another storage into this one."""
        self.bins = store.bins[:]
        self.count = store.count
        self.min_key = store.min_key
        self.max_key = store.max_key
        self.offset = store.offset
        self.is_collapsed = store.is_collapsed
        self.collapse_count = store.collapse_count
