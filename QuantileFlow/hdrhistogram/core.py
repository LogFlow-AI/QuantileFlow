"""
High Dynamic Range Histogram implementation for quantile approximation.

This module provides an efficient implementation of HDR Histogram for tracking values
across a wide range using logarithmic bucketing. It supports streaming data and
provides accurate quantile estimates with configurable precision.
"""

import numpy as np
from typing import List, Union, Dict
import matplotlib.pyplot as plt


class HDRHistogram:
    """
    High Dynamic Range Histogram implementation for quantile approximation.

    Provides efficient tracking of values across a wide range using logarithmic bucketing,
    with methods for inserting values, computing quantiles, and generating summary statistics.
    """

    def __init__(
            self,
            num_buckets: int = 8,
            min_value: float = 1.0,
            max_value: float = float('inf')
    ):
        """
        Initialize HDR Histogram.

        Args:
            num_buckets: Number of logarithmic buckets (default 8)
            min_value: Minimum trackable value (default 1.0)
            max_value: Maximum trackable value (default infinity)
        """
        self.num_buckets = num_buckets
        self.min_value = min_value
        self.max_value = max_value

        # Use NumPy array for buckets
        self.buckets = np.zeros(num_buckets, dtype=np.int64)
        self.total_count = 0
        
        # Track actual min and max values seen
        self.actual_min = float('inf')
        self.actual_max = float('-inf')

    def insert(self, value: Union[int, float]) -> None:
        """
        Insert a single value into the histogram.

        Args:
            value: The value to insert.
        """
        if value > self.max_value:
            return

        if value < self.min_value:
            value = self.min_value

        bucket_index = self._calculate_bucket_index(value)
        self.buckets[bucket_index] += 1
        self.total_count += 1
        
        # Update actual min and max
        self.actual_min = min(self.actual_min, value)
        self.actual_max = max(self.actual_max, value)

    def insert_batch(self, values: Union[List[float], np.ndarray]) -> None:
        """
        Insert multiple values into the histogram.

        Args:
            values: Array or list of values to insert.
        """
        # Vectorized filtering of values within max range
        values_array = np.array(values)
        valid_mask = values_array <= self.max_value
        valid_values = values_array[valid_mask]

        if len(valid_values) == 0:
            return

        # Clip values below min_value
        valid_values = np.clip(valid_values, self.min_value, self.max_value)

        # Update actual min and max
        self.actual_min = min(self.actual_min, np.min(valid_values))
        self.actual_max = max(self.actual_max, np.max(valid_values))

        # Vectorized bucket calculation
        bucket_indices = np.clip(
            np.floor(np.log2(valid_values)).astype(int),
            0,
            self.num_buckets - 1
        )

        # Use NumPy's bincount to efficiently count occurrences
        increments = np.bincount(bucket_indices, minlength=self.num_buckets)

        # Update buckets and total count
        self.buckets += increments
        self.total_count += len(valid_values)

    def _calculate_bucket_index(self, value):
        """
        Calculate the appropriate bucket index for a given value.
        """
        if value < self.min_value:
            return 0

        try:
            bucket = int(np.log2(value))
            return min(max(0, bucket), self.num_buckets - 1)
        except (ValueError, TypeError):
            return self.num_buckets - 1

    def _reconstruct_bucket_bounds(self):
        """
        Reconstruct the lower and upper bounds for each bucket.
        """
        bounds = []
        for i in range(self.num_buckets):
            lower = 2 ** i if i > 0 else self.min_value
            upper = min(2 ** (i + 1), self.max_value) if i < self.num_buckets - 1 else self.max_value
            bounds.append((lower, upper))
        return bounds

    def quantile(self, fraction: float) -> float:
        """
        Get the value at a given quantile.

        Args:
            fraction: Quantile fraction between 0 and 1 (e.g., 0.5 for median).

        Returns:
            Estimated value at the requested quantile.
        """
        if not 0 <= fraction <= 1:
            raise ValueError("Quantile must be between 0 and 1")

        if self.total_count == 0:
            return 0.0

        if fraction == 0:
            return self.actual_min
        if fraction == 1:
            return self.actual_max

        bucket_counts = self.buckets
        bucket_bounds = self._reconstruct_bucket_bounds()

        # Calculate target count
        target_count = int(self.total_count * fraction)
        cumulative_count = 0

        for i, (count, (lower, upper)) in enumerate(zip(bucket_counts, bucket_bounds)):
            cumulative_count += count

            if cumulative_count >= target_count:
                # If no values in this bucket, return lower bound
                if count == 0:
                    return lower

                # Linear interpolation within the bucket
                bucket_proportion = (target_count - (cumulative_count - count)) / count
                return lower + (upper - lower) * bucket_proportion

        # If we've exhausted all buckets, return the maximum value
        return self.actual_max

    def quantiles(self, fractions: List[float]) -> List[float]:
        """
        Get values at multiple quantiles.

        Args:
            fractions: List of quantile fractions between 0 and 1.

        Returns:
            List of estimated values at the requested quantiles.
        """
        for q in fractions:
            if not 0 <= q <= 1:
                raise ValueError("All quantiles must be between 0 and 1")

        return [self.quantile(q) for q in fractions]

    def median(self) -> float:
        """
        Get the median value (50th percentile).

        Returns:
            Estimated median value.
        """
        return self.quantile(0.5)

    def percentile(self, p: float) -> float:
        """
        Get the p-th percentile value.

        Args:
            p: Percentile between 0 and 100 (e.g., 75 for 75th percentile).

        Returns:
            Estimated value at the requested percentile.
        """
        if not 0 <= p <= 100:
            raise ValueError("Percentile must be between 0 and 100")

        return self.quantile(p / 100)

    def interquartile_range(self) -> float:
        """
        Get the interquartile range (IQR).

        Returns:
            Estimated IQR (difference between 75th and 25th percentiles).
        """
        return self.percentile(75) - self.percentile(25)

    def summary_statistics(self) -> Dict[str, float]:
        """
        Get summary statistics.

        Returns:
            Dictionary containing min, q1, median, q3, max, and count.
        """
        if self.total_count == 0:
            return {
                'min': 0.0,
                'q1': 0.0,
                'median': 0.0,
                'q3': 0.0,
                'max': 0.0,
                'count': 0
            }

        return {
            'min': self.actual_min,
            'q1': self.percentile(25),
            'median': self.median(),
            'q3': self.percentile(75),
            'max': self.actual_max,
            'count': self.total_count
        }

    def plot_distribution(self, figsize=(10, 6)):
        """
        Plot the estimated probability distribution.

        Args:
            figsize: Figure size (width, height) in inches.

        Returns:
            Matplotlib figure object.
        """
        bounds = self._reconstruct_bucket_bounds()
        midpoints = [(lower + upper) / 2 for lower, upper in bounds]

        fig = plt.figure(figsize=figsize)
        plt.bar(midpoints, self.buckets, width=[upper - lower for lower, upper in bounds], alpha=0.7)
        plt.title('HDR Histogram Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.xscale('log')
        return fig

    def to_dict(self) -> Dict:
        """
        Convert histogram to a dictionary for serialization.

        Returns:
            Dictionary representation of the histogram.
        """
        return {
            'buckets': self.buckets.tolist(),
            'total_count': self.total_count,
            'num_buckets': self.num_buckets,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'actual_min': self.actual_min,
            'actual_max': self.actual_max
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'HDRHistogram':
        """
        Create a histogram from a dictionary.

        Args:
            data: Dictionary representation of a histogram.

        Returns:
            New HDRHistogram instance.
        """
        histogram = cls(
            num_buckets=data['num_buckets'],
            min_value=data['min_value'],
            max_value=data['max_value']
        )
        histogram.buckets = np.array(data['buckets'])
        histogram.total_count = data['total_count']
        histogram.actual_min = data['actual_min']
        histogram.actual_max = data['actual_max']
        return histogram 