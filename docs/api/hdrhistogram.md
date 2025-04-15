# HDRHistogram

HDRHistogram (High Dynamic Range Histogram) is a logarithmic-bucketed histogram implementation designed for efficient tracking of values across multiple orders of magnitude.

## Overview

HDRHistogram offers the following features:

* **Wide Value Range**: Efficiently tracks values across multiple orders of magnitude
* **Configurable Precision**: Adjustable number of buckets for different accuracy needs
* **Memory Efficient**: Uses logarithmic bucketing to minimize memory usage
* **Summary Statistics**: Provides comprehensive statistics including min, max, quartiles, and count
* **Visualization**: Built-in support for plotting distributions with logarithmic scales
* **Serialization**: Supports converting histograms to/from dictionaries for storage and transmission
* **Value Range Control**: Configurable minimum and maximum trackable values

## Main Class

```{eval-rst}
.. autoclass:: QuantileFlow.HDRHistogram
   :members:
   :undoc-members:
   :special-members: __init__
   :exclude-members: __dict__, __weakref__
   :noindex:
   :show-inheritance:
```

## Mathematical Background

HDRHistogram works by maintaining a fixed number of logarithmic buckets to efficiently track values across a wide range. The key aspects of the implementation are:

1. **Logarithmic Bucketing**: Values are assigned to buckets based on their logarithm, allowing efficient tracking of values across multiple orders of magnitude
2. **Value Range Control**: Configurable minimum and maximum values ensure memory efficiency
3. **Linear Interpolation**: Within each bucket, linear interpolation is used to estimate quantiles

### Bucket Calculation

The bucket index for a value $x$ is calculated as:

$$
\text{bucket_index} = \lfloor \log_2(x) \rfloor
$$

This ensures that:
- Values in the same order of magnitude are grouped together
- The number of buckets grows logarithmically with the value range
- Memory usage remains constant regardless of the number of values inserted

### Quantile Estimation

Quantiles are estimated by:
1. Finding the bucket containing the target quantile
2. Using linear interpolation within the bucket to estimate the exact value

For a target quantile $q$ and total count $N$, the estimated value is:

$$
\text{value} = \text{lower_bound} + (\text{upper_bound} - \text{lower_bound}) \times \frac{qN - \text{cumulative_count}}{\text{bucket_count}}
$$

Where:
- $\text{lower_bound}$ and $\text{upper_bound}$ are the bucket boundaries
- $\text{cumulative_count}$ is the count of values in previous buckets
- $\text{bucket_count}$ is the count of values in the current bucket

## Performance Characteristics

- **Memory Usage**: O(num_buckets)
- **Insertion Time**: O(1) per value
- **Query Time**: O(num_buckets) for quantile queries
- **Merge Time**: O(num_buckets) for merging histograms

## Use Cases

HDRHistogram is particularly useful for:
- Tracking metrics with wide value ranges (e.g., response times, request sizes)
- Monitoring systems with multiple orders of magnitude in measurements
- Applications requiring configurable precision and value range control
- Distributed systems where histograms need to be merged
- Real-time monitoring and alerting systems

## Usage Examples

For basic usage, see the [Getting Started](../getting-started.md) guide.

For more advanced examples, see the [Examples](../examples.md) page. 