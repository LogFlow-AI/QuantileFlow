# MomentSketch

## Overview

MomentSketch is a technique for approximating quantiles of a dataset using moment-based sketching and maximum entropy optimization. This implementation is integrated into the QuantileFlow package and provides a memory-efficient way to estimate quantiles with good accuracy.

## Key Features

- **Compact representation**: Stores only a fixed number of moments (power sums) regardless of input data size
- **Mergeable**: Sketches can be combined to represent the union of their datasets
- **Accurate estimates**: Uses maximum entropy optimization to estimate the underlying distribution
- **Flexible**: Supports optional data compression for handling widely distributed values
- **Serializable**: Sketches can be serialized to dictionaries for storage or transmission

## Usage

```python
from QuantileFlow import MomentSketch
import numpy as np

# Create a sketch with 20 moments
sketch = MomentSketch(num_moments=20)

# Add data
data = np.random.normal(10, 2, 10000)
sketch.insert_batch(data)

# Get quantiles
median = sketch.median()
p90 = sketch.percentile(90)  # 90th percentile
iqr = sketch.interquartile_range()

# Get multiple quantiles at once
quantiles = sketch.quantiles([0.25, 0.5, 0.75])

# Get summary statistics
stats = sketch.summary_statistics()

# Visualize the distribution
fig = sketch.plot_distribution()
```

## Implementation Details

The MomentSketch implementation consists of four main components:

1. **Core Interface (`MomentSketch` class)**: Provides a clean API for users to interact with the sketch.

2. **Moment Computation (`Moment` class)**: 
   - Accumulates power sums of the input data
   - Tracks minimum and maximum values
   - Supports merging moments from different data sources

3. **Maximum Entropy Optimization**:
   - Converts power sums to Chebyshev moments
   - Uses Newton's method to solve the maximum entropy problem
   - Computes weights that form the estimated probability distribution

4. **Quantile Estimation**:
   - Uses the estimated distribution to compute quantiles
   - Supports various summarization methods (median, percentiles, IQR)

## Performance Considerations

- **Time Complexity**:
  - Insertion: O(k) per value, where k is the number of moments
  - Quantile estimation: O(k³ + m), where m is the grid size for the optimization
  - Merging: O(k) for combining two sketches

- **Space Complexity**:
  - O(k) regardless of input data size

- **Accuracy vs. Memory Tradeoff**:
  - Higher number of moments increases accuracy at the cost of computation
  - 10-20 moments typically provide good accuracy for most distributions

## References

- "Space- and Computationally-Efficient Set Similarity via Locality Sensitive Sketching" by Anshumali Shrivastava
- "Maximum Entropy Modeling with Moment Constraints" by E.T. Jaynes