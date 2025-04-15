"""
Mapping schemes for DDSketch algorithm.

This module provides various mapping schemes for the DDSketch algorithm:

- LogarithmicMapping: The canonical implementation with provable relative error guarantees
- LinearInterpolationMapping: Faster approximation using linear interpolation
- CubicInterpolationMapping: Memory-efficient approximation using cubic interpolation

All mapping schemes derive from the MappingScheme base class and provide methods to
compute the bucket index for a given value and to recover a value from a bucket index.
"""

from QuantileFlow.ddsketch.mapping.base import MappingScheme
from QuantileFlow.ddsketch.mapping.logarithmic import LogarithmicMapping
from QuantileFlow.ddsketch.mapping.linear_interpolation import LinearInterpolationMapping
from QuantileFlow.ddsketch.mapping.cubic_interpolation import CubicInterpolationMapping

__all__ = [
    "MappingScheme",
    "LogarithmicMapping",
    "LinearInterpolationMapping",
    "CubicInterpolationMapping"
] 