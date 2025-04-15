"""
Linear interpolation mapping scheme for DDSketch.

This implementation approximates the memory-optimal logarithmic mapping by:
1. Extracting the floor value of log2 from binary representation
2. Linearly interpolating the logarithm between consecutive powers of 2
"""

import numpy as np
from .base import MappingScheme

class LinearInterpolationMapping(MappingScheme):
    def __init__(self, relative_accuracy: float):
        self.relative_accuracy = relative_accuracy
        self.gamma = (1 + relative_accuracy) / (1 - relative_accuracy)
        self.log_gamma = np.log(self.gamma)
        
    def _extract_exponent(self, value: float) -> tuple[int, float]:
        """
        Extract the binary exponent and normalized fraction from an IEEE 754 float.
        
        Returns:
            tuple: (exponent, normalized_fraction)
            where normalized_fraction is in [1, 2)
        """
        # Use numpy's frexp for better numerical precision
        mantissa, exponent = np.frexp(value) # mantissa * 2^exponent = value
        exponent -= 1  # Convert to floor(log2)
        normalized_fraction = mantissa * 2  # Scale to [1, 2)
        return exponent, normalized_fraction
        
    def compute_bucket_index(self, value: float) -> int:
        if value <= 0:
            raise ValueError("Value must be positive")
            
        # Get binary exponent and normalized fraction
        exponent, normalized_fraction = self._extract_exponent(value)
        
        # Linear interpolation between powers of 2
        # normalized_fraction is in [1, 2), so we interpolate log_gamma
        log2_fraction = normalized_fraction - 1  # Map [1, 2) to [0, 1)
        
        # Compute final index
        log2_value = exponent + log2_fraction
        return int(np.ceil(log2_value / self.log_gamma))
        
    def compute_value_from_index(self, index: int) -> float:
        """
        Compute the value corresponding to a bucket index using the inverse mapping.
        
        Args:
            index: The bucket index
            
        Returns:
            The value at the center of the bucket
        """
        # Follow the same approach as LinearlyInterpolatedMapping.value
        log2_value = index * self.log_gamma
        
        # Extract the integer and fractional parts of log2_value
        exponent = int(np.floor(log2_value) + 1)
        mantissa = (log2_value - exponent + 2) / 2.0
        
        # Use ldexp to efficiently compute 2^exponent * mantissa
        result = np.ldexp(mantissa, exponent)
        
        # Apply the centering factor
        return result * (2.0 / (1 + self.gamma))
        