"""
MomentSketch: Quantile Estimation Using Moment-Based Sketching

This module provides an efficient implementation of the moment-based sketching algorithm
for quantile estimation and summary statistics. Key features include:

- Memory efficiency: Uses a fixed number of moments regardless of data size
- Mergeable: Supports distributed computation through sketch merging
- Accurate: Employs maximum entropy optimization for accurate distribution estimation
- Flexible: Supports optional data compression for handling widely distributed values
- Comprehensive: Provides various summary statistics beyond just quantiles

The implementation is based on power sums and maximum entropy optimization,
making it suitable for streaming data applications where memory efficiency
and accuracy are important.
"""

from .core import MomentSketch

__all__ = [
    "MomentSketch"
] 