"""Logarithmic mapping scheme for DDSketch."""

import numpy as np
from .base import MappingScheme

class LogarithmicMapping(MappingScheme):
    def __init__(self, relative_accuracy: float):
        self.relative_accuracy = relative_accuracy
        self.gamma = (1 + relative_accuracy) / (1 - relative_accuracy)
        self.multiplier = 1 / np.log(self.gamma)
        
    def compute_bucket_index(self, value: float) -> int:
        if value <= 0:
            raise ValueError(f"Value must be positive, got {value}")
        # ceil(log_gamma(value) = ceil(log(value) / log(gamma))
        return int(np.ceil(np.log(value) * self.multiplier))
    
    def compute_value_from_index(self, index: int) -> float:
        # Return geometric mean of bucket boundaries
        # This ensures the relative error is bounded by relative_accuracy
        return np.power(self.gamma, index) * (2.0 / (1.0 + self.gamma))