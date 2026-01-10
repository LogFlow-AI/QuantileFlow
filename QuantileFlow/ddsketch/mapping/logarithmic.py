"""Logarithmic mapping scheme for DDSketch."""

import math
from .base import MappingScheme

class LogarithmicMapping(MappingScheme):
    __slots__ = ('relative_accuracy', 'gamma', 'multiplier')
    
    def __init__(self, relative_accuracy: float):        
        self.relative_accuracy = relative_accuracy
        self.gamma = (1 + relative_accuracy) / (1 - relative_accuracy)
        self.multiplier = 1 / math.log(self.gamma)
        
    def compute_bucket_index(self, value: float) -> int:
        # ceil(log_gamma(value) = ceil(log(value) / log(gamma))
        return math.ceil(math.log(value) * self.multiplier)
    
    def compute_value_from_index(self, index: int) -> float:
        # Return geometric mean of bucket boundaries
        # This ensures the relative error is bounded by relative_accuracy
        return math.pow(self.gamma, index) * (2.0 / (1.0 + self.gamma))