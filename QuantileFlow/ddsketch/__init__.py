"""
DDSketch: Distributed and Mergeable Quantile Sketch with Relative Error Guarantees

This module provides an implementation of the DDSketch algorithm for computing approximate
quantiles with a user-defined relative error bound. Key features include:

- Configurable accuracy: Set the desired relative error guarantee for quantile estimation
- Mergeable: Sketches can be combined for distributed applications
- Space efficient: Uses compact bucket structures to minimize memory usage
- Fast updates: Insert operations are O(1) time complexity
- Robust: Maintains error guarantees across the entire value range

The implementation includes different mapping schemes:
- Logarithmic: The canonical implementation with provable relative error guarantees
- Linear interpolation: Approximation using linear interpolation for improved performance
- Cubic interpolation: Approximation using cubic interpolation for better memory efficiency
"""

from QuantileFlow.ddsketch.core import DDSketch
from QuantileFlow.ddsketch.mapping.logarithmic import LogarithmicMapping
from QuantileFlow.ddsketch.mapping.linear_interpolation import LinearInterpolationMapping
from QuantileFlow.ddsketch.mapping.cubic_interpolation import CubicInterpolationMapping
from QuantileFlow.ddsketch.storage.contiguous import ContiguousStorage
from QuantileFlow.ddsketch.storage.sparse import SparseStorage
from QuantileFlow.ddsketch.storage.base import BucketManagementStrategy, Storage

__all__ = [
    "DDSketch",
    "LogarithmicMapping",
    "LinearInterpolationMapping",
    "CubicInterpolationMapping",
    "ContiguousStorage", 
    "SparseStorage",
    "BucketManagementStrategy",
    "Storage"
]
