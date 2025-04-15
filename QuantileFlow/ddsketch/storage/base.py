"""Base classes for DDSketch storage implementations."""

from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np
import warnings

class BucketManagementStrategy(Enum):
    """Strategy for managing the number of buckets in the sketch."""
    UNLIMITED = auto()  # No limit on number of buckets
    FIXED = auto()     # Fixed maximum number of buckets
    DYNAMIC = auto()   # Dynamic limit based on log(n), ignores max_buckets parameter

class Storage(ABC):
    """Abstract base class for different storage types."""
    
    def __init__(self, max_buckets: int = 2048, 
                 strategy: BucketManagementStrategy = BucketManagementStrategy.FIXED):
        """
        Initialize storage with bucket management strategy.
        
        Args:
            max_buckets: Maximum number of buckets (default 2048). 
                        Only used if strategy is FIXED. Ignored for UNLIMITED and DYNAMIC.
            strategy: Bucket management strategy (default FIXED).
        """
        self.strategy = strategy
        if (strategy in [BucketManagementStrategy.UNLIMITED, BucketManagementStrategy.DYNAMIC] 
            and max_buckets != 2048):
            warnings.warn(
                f"max_buckets={max_buckets} was provided but will be ignored because "
                f"strategy={strategy} was selected. The storage will use strategy-specific "
                "bucket management.",
                UserWarning
            )
        
        if strategy == BucketManagementStrategy.FIXED:
            self.max_buckets = max_buckets
        elif strategy == BucketManagementStrategy.UNLIMITED:
            self.max_buckets = -1
        else:  # DYNAMIC
            # Initialize with reasonable minimum for small counts
            self.max_buckets = 32
            
        self.total_count = 0  # Used for dynamic strategy
        self.last_order_of_magnitude = 0  # Track last order of magnitude for dynamic updates
        
    @abstractmethod
    def add(self, bucket_index: int, count: int = 1):
        """Add count to bucket_index."""
        pass
    
    @abstractmethod
    def remove(self, bucket_index: int, count: int = 1) -> bool:
        """
        Remove count from bucket_index.
        
        Args:
            bucket_index: The bucket index to remove from.
            count: The count to remove (default 1).
            
        Returns:
            bool: True if any value was actually removed, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_count(self, bucket_index: int) -> int:
        """Get count for bucket_index."""
        pass
    
    @abstractmethod
    def merge(self, other: 'Storage'):
        """Merge another storage into this one."""
        pass
    
    def _should_update_dynamic_limit(self) -> bool:
        """Check if we should update the dynamic limit based on order of magnitude change."""
        if self.strategy != BucketManagementStrategy.DYNAMIC:
            return False
            
        if self.total_count <= 0:
            return False
            
        current_order = int(np.floor(np.log10(self.total_count)))
        if current_order != self.last_order_of_magnitude:
            self.last_order_of_magnitude = current_order
            return True
        return False
    
    def _update_dynamic_limit(self):
        """Update max_buckets for dynamic strategy based on total count."""
        if self._should_update_dynamic_limit():
            # Set m = max(32, c*log(n)) where c is a constant
            # This ensures we have at least 32 buckets for small counts
            # while still maintaining logarithmic growth for larger counts
            self.max_buckets = max(32, int(100 * np.log10(self.total_count + 1))) 