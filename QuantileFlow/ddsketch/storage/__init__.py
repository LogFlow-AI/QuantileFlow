"""
Storage implementations for DDSketch algorithm.

This module provides different storage strategies for DDSketch:

- ContiguousStorage: Memory-efficient fixed-size array for a limited bucket range
- SparseStorage: Hash-based structure for handling wider bucket ranges
- BucketManagementStrategy: Enumeration of strategies for handling bucket limitations

All storage classes derive from the Storage base class and provide methods to add,
remove, and merge bucket counts, as well as retrieve counts for specific buckets.
"""

from QuantileFlow.ddsketch.storage.base import Storage, BucketManagementStrategy
from QuantileFlow.ddsketch.storage.contiguous import ContiguousStorage
from QuantileFlow.ddsketch.storage.sparse import SparseStorage

__all__ = [
    "Storage",
    "BucketManagementStrategy",
    "ContiguousStorage",
    "SparseStorage"
] 