"""
Core MomentSketch implementation.

This module provides the main MomentSketch class which serves as the public API
for the moment-based quantile estimation algorithm. The MomentSketch implementation:

- Maintains a compact representation of a dataset using power sums
- Provides accurate quantile estimates through maximum entropy optimization
- Supports merging sketches for distributed computation
- Offers comprehensive summary statistics and visualization capabilities
- Handles data compression for widely distributed values

The implementation is designed to be memory-efficient and accurate, making it suitable
for streaming data applications and monitoring systems where traditional approaches
would require excessive memory.
"""

from typing import List, Union, Dict
import numpy as np

from .simple_moment_sketch import SimpleMS

class MomentSketch:
    """
    MomentSketch implementation for quantile approximation using the moment-based approach.
    
    This implementation uses power sums, Chebyshev moment conversion, and maximum entropy
    optimization to estimate the probability distribution of data and compute quantiles.
    It supports merging sketches from distributed sources and provides accurate quantile
    estimates with a compact representation.
    
    Reference:
        "Space- and Computationally-Efficient Set Similarity via Locality Sensitive Sketching"
        by Anshumali Shrivastava
    """
    
    def __init__(
        self,
        num_moments: int = 20,
        compress_values: bool = False
    ):
        """
        Initialize MomentSketch.
        
        Args:
            num_moments: Number of moments to track (default 20).
                         Higher values increase accuracy at the cost of computation.
            compress_values: Whether to compress values using arcsinh transformation (default False).
                            Useful for handling widely distributed data with extreme values.
        """
        self.sketch = SimpleMS(num_moments)
        self.sketch.set_compressed(compress_values)
    
    def insert(self, value: Union[int, float]) -> None:
        """
        Insert a single value into the sketch.
        
        Args:
            value: The value to insert.
        """
        self.sketch.add(value)
    
    def insert_batch(self, values: Union[List[float], np.ndarray]) -> None:
        """
        Insert multiple values into the sketch.
        
        Args:
            values: Array or list of values to insert.
        """
        self.sketch.add_many(values)
    
    def merge(self, other: 'MomentSketch') -> None:
        """
        Merge another MomentSketch into this one.
        
        Args:
            other: Another MomentSketch instance to merge.
        
        Raises:
            ValueError: If the sketches are incompatible (different compression settings).
        """
        self.sketch.merge(other.sketch)
    
    def quantile(self, fraction: float) -> float:
        """
        Get the value at a given quantile.
        
        Args:
            fraction: Quantile fraction between 0 and 1 (e.g., 0.5 for median).
        
        Returns:
            Estimated value at the requested quantile.
            
        Raises:
            ValueError: If fraction is not between 0 and 1.
        """
        if not 0 <= fraction <= 1:
            raise ValueError("Quantile must be between 0 and 1")
            
        return self.sketch.get_quantile(fraction)
    
    def quantiles(self, fractions: List[float]) -> List[float]:
        """
        Get values at multiple quantiles.
        
        Args:
            fractions: List of quantile fractions between 0 and 1.
        
        Returns:
            List of estimated values at the requested quantiles.
            
        Raises:
            ValueError: If any fraction is not between 0 and 1.
        """
        for q in fractions:
            if not 0 <= q <= 1:
                raise ValueError("All quantiles must be between 0 and 1")
                
        return self.sketch.get_quantiles(fractions)
    
    def median(self) -> float:
        """
        Get the median value (50th percentile).
        
        Returns:
            Estimated median value.
        """
        return self.sketch.get_median()
    
    def percentile(self, p: float) -> float:
        """
        Get the p-th percentile value.
        
        Args:
            p: Percentile between 0 and 100 (e.g., 75 for 75th percentile).
        
        Returns:
            Estimated value at the requested percentile.
            
        Raises:
            ValueError: If p is not between 0 and 100.
        """
        if not 0 <= p <= 100:
            raise ValueError("Percentile must be between 0 and 100")
            
        return self.sketch.get_percentile(p)
    
    def interquartile_range(self) -> float:
        """
        Get the interquartile range (IQR).
        
        Returns:
            Estimated IQR (difference between 75th and 25th percentiles).
        """
        return self.sketch.get_iqr()
    
    def summary_statistics(self) -> Dict[str, float]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary containing min, q1, median, q3, max, count, and mean.
        """
        return self.sketch.get_stats()
    
    def plot_distribution(self, figsize=(10, 6)):
        """
        Plot the estimated probability distribution.
        
        Args:
            figsize: Figure size (width, height) in inches.
        
        Returns:
            Matplotlib figure object.
        """
        return self.sketch.plot_dist(figsize=figsize)
    
    def to_dict(self) -> Dict:
        """
        Convert sketch to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the sketch.
        """
        return self.sketch.to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MomentSketch':
        """
        Create a sketch from a dictionary.
        
        Args:
            data: Dictionary representation of a sketch.
        
        Returns:
            New MomentSketch instance.
        """
        sketch = cls()
        sketch.sketch = SimpleMS.from_dict(data)
        return sketch 