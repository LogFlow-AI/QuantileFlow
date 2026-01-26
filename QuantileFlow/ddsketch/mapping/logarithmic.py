"""Logarithmic mapping scheme for DDSketch."""

import math
from .base import MappingScheme


class LogarithmicMapping(MappingScheme):
    """
    A memory-optimal KeyMapping that uses logarithmic mapping.
    
    Given a targeted relative accuracy, it requires the least number of keys 
    to cover a given range of values.
    """
    __slots__ = ('relative_accuracy', 'gamma', 'multiplier')
    
    def __init__(self, relative_accuracy: float):        
        self.relative_accuracy = relative_accuracy
        self.gamma = (1 + relative_accuracy) / (1 - relative_accuracy)
        self.multiplier = 1 / math.log(self.gamma)
    
    def key(self, value: float) -> int:
        """Alias for compute_bucket_index for API compatibility."""
        return self.compute_bucket_index(value)
    
    def value(self, key: int) -> float:
        """Alias for compute_value_from_index for API compatibility."""
        return self.compute_value_from_index(key)
        
    def compute_bucket_index(self, value: float) -> int:
        """Compute the bucket index for a given value.
        
        ceil(log_gamma(value)) = ceil(log(value) / log(gamma))
        """
        return math.ceil(math.log(value) * self.multiplier)
    
    def compute_value_from_index(self, index: int) -> float:
        """Compute the representative value for a given bucket index.
        
        Returns the geometric mean of bucket boundaries to ensure
        the relative error is bounded by relative_accuracy.
        """
        return math.pow(self.gamma, index) * (2.0 / (1.0 + self.gamma))