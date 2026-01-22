"""Core DDSketch implementation.

Optimized for high throughput to match or exceed Datadog's implementation.
"""

from typing import Literal, Union
from .mapping.logarithmic import LogarithmicMapping
from .mapping.linear_interpolation import LinearInterpolationMapping
from .mapping.cubic_interpolation import CubicInterpolationMapping
from .storage.base import BucketManagementStrategy
from .storage.contiguous import ContiguousStorage
from .storage.sparse import SparseStorage


class DDSketch:
    """
    DDSketch implementation for quantile approximation with relative-error guarantees.
    
    This implementation supports different mapping schemes and storage types for
    optimal performance in different scenarios. It can handle both positive and
    negative values, and provides configurable bucket management strategies.
    
    Reference:
        "DDSketch: A Fast and Fully-Mergeable Quantile Sketch with Relative-Error Guarantees"
        by Charles Masson, Jee E. Rim and Homin K. Lee
    """
    
    __slots__ = ('relative_accuracy', 'cont_neg', 'mapping', 'positive_store',
                 'negative_store', 'count', 'zero_count', '_min', '_max', '_sum')
    
    def __init__(
        self,
        relative_accuracy: float,
        mapping_type: Literal['logarithmic', 'lin_interpol', 'cub_interpol'] = 'logarithmic',
        max_buckets: int = 2048,
        bucket_strategy: BucketManagementStrategy = BucketManagementStrategy.FIXED,
        cont_neg: bool = True
    ):
        """
        Initialize DDSketch.
        
        Args:
            relative_accuracy: The relative accuracy guarantee (alpha).
                             Must be between 0 and 1.
            mapping_type: The type of mapping scheme to use:
                        - 'logarithmic': Basic logarithmic mapping
                        - 'lin_interpol': Linear interpolation mapping
                        - 'cub_interpol': Cubic interpolation mapping
            max_buckets: Maximum number of buckets per store (default 2048).
                        If cont_neg is True, each store will have max_buckets buckets.
            bucket_strategy: Strategy for managing bucket count.
                           If FIXED, uses ContiguousStorage, otherwise uses SparseStorage.
            cont_neg: Whether to handle negative values (default True).
        
        Raises:
            ValueError: If relative_accuracy is not between 0 and 1.
        """
        if not 0 < relative_accuracy < 1:
            raise ValueError("relative_accuracy must be between 0 and 1")
            
        self.relative_accuracy = relative_accuracy
        self.cont_neg = cont_neg
        
        # Initialize mapping scheme
        if mapping_type == 'logarithmic':
            self.mapping = LogarithmicMapping(relative_accuracy)
        elif mapping_type == 'lin_interpol':
            self.mapping = LinearInterpolationMapping(relative_accuracy)
        elif mapping_type == 'cub_interpol':
            self.mapping = CubicInterpolationMapping(relative_accuracy)
            
        # Choose storage type based on strategy
        if bucket_strategy == BucketManagementStrategy.FIXED:
            self.positive_store = ContiguousStorage(max_buckets)
            self.negative_store = ContiguousStorage(max_buckets) if cont_neg else None
        else:
            self.positive_store = SparseStorage(strategy=bucket_strategy)
            self.negative_store = SparseStorage(strategy=bucket_strategy) if cont_neg else None
            
        self.count = 0.0
        self.zero_count = 0.0
        
        # Summary stats (like Datadog)
        self._min = float('+inf')
        self._max = float('-inf')
        self._sum = 0.0
    
    def insert(self, value: Union[int, float], weight: float = 1.0) -> None:
        """
        Insert a value into the sketch.
        
        Args:
            value: The value to insert.
            weight: The weight of the value (default 1.0).
            
        Raises:
            ValueError: If value is negative and cont_neg is False.
        """
        if value > 0:
            self.positive_store.add(self.mapping.compute_bucket_index(value), weight)
        elif value < 0:
            if self.cont_neg:
                self.negative_store.add(self.mapping.compute_bucket_index(-value), weight)
            else:
                raise ValueError("Negative values not supported when cont_neg is False")
        else:
            self.zero_count += weight
        
        # Track summary stats
        self.count += weight
        self._sum += value * weight
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value
    
    # Alias for compatibility with Datadog's API
    def add(self, value: Union[int, float], weight: float = 1.0) -> None:
        """Alias for insert() to match Datadog's API."""
        self.insert(value, weight)
    
    def delete(self, value: Union[int, float]) -> None:
        """
        Delete a value from the sketch.
        
        Args:
            value: The value to delete.
            
        Raises:
            ValueError: If value is negative and cont_neg is False.
        """
        if self.count == 0:
            return
            
        deleted = False
        if value == 0 and self.zero_count > 0:
            self.zero_count -= 1
            deleted = True
        elif value > 0:
            bucket_idx = self.mapping.compute_bucket_index(value)
            deleted = self.positive_store.remove(bucket_idx)
        elif value < 0 and self.cont_neg:
            bucket_idx = self.mapping.compute_bucket_index(-value)
            deleted = self.negative_store.remove(bucket_idx)
        elif value < 0:
            raise ValueError("Negative values not supported when cont_neg is False")
            
        if deleted:
            self.count -= 1
            self._sum -= value
    
    def quantile(self, q: float) -> float:
        """
        Compute the approximate quantile.
        
        Args:
            q: The desired quantile (between 0 and 1).
            
        Returns:
            The approximate value at the specified quantile.
            
        Raises:
            ValueError: If q is not between 0 and 1 or if the sketch is empty.
        """
        if not 0 <= q <= 1:
            raise ValueError("Quantile must be between 0 and 1")
        if self.count == 0:
            raise ValueError("Cannot compute quantile of empty sketch")
            
        rank = q * (self.count - 1)
        
        if self.cont_neg and self.negative_store is not None:
            neg_count = self.negative_store.count
            if rank < neg_count:
                # Handle negative values - use reversed rank
                reversed_rank = neg_count - rank - 1
                key = self.negative_store.key_at_rank(reversed_rank, lower=False)
                return -self.mapping.compute_value_from_index(key)
            rank -= neg_count
            
        if rank < self.zero_count:
            return 0.0
        rank -= self.zero_count
        
        # Use key_at_rank for consistency with storage implementation
        key = self.positive_store.key_at_rank(rank)
        return self.mapping.compute_value_from_index(key)
    
    # Alias for Datadog compatibility
    def get_quantile_value(self, quantile: float) -> float:
        """Alias for quantile() to match Datadog's API."""
        try:
            return self.quantile(quantile)
        except ValueError:
            return None
    
    @property
    def avg(self) -> float:
        """Return the exact average of values added to the sketch."""
        if self.count == 0:
            return 0.0
        return self._sum / self.count
    
    @property
    def sum(self) -> float:
        """Return the exact sum of values added to the sketch."""
        return self._sum
    
    @property
    def min(self) -> float:
        """Return the minimum value added to the sketch."""
        return self._min
    
    @property
    def max(self) -> float:
        """Return the maximum value added to the sketch."""
        return self._max
    
    def merge(self, other: 'DDSketch') -> None:
        """
        Merge another DDSketch into this one.
        
        Args:
            other: Another DDSketch instance to merge with this one.
            
        Raises:
            ValueError: If the sketches are incompatible.
        """
        if self.relative_accuracy != other.relative_accuracy:
            raise ValueError("Cannot merge sketches with different relative accuracies")
        
        if other.count == 0:
            return
            
        self.positive_store.merge(other.positive_store)
        if self.cont_neg and other.cont_neg and other.negative_store is not None:
            self.negative_store.merge(other.negative_store)
        elif other.cont_neg and other.negative_store is not None and other.negative_store.count > 0:
            raise ValueError("Cannot merge sketch containing negative values when cont_neg is False")
            
        self.zero_count += other.zero_count
        self.count += other.count
        self._sum += other._sum
        if other._min < self._min:
            self._min = other._min
        if other._max > self._max:
            self._max = other._max