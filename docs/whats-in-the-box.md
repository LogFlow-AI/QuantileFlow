# What's in The Box?

QuantileFlow provides two main algorithms for quantile estimation:

## DDSketch

DDSketch (Distributed and Deterministic Sketch) is a quantile approximation algorithm with the following properties:

- **Relative Error Guarantee**: Configurable error bounds on quantile estimations
- **Mergeable**: Sketches can be combined for distributed processing
- **Storage Options**:
  - `ContiguousStorage`: Efficient array-based storage for limited bucket ranges
  - `SparseStorage`: Hash-based storage for wider bucket ranges
- **Mapping Schemes**:
  - `LogarithmicMapping`: The canonical implementation with provable error guarantees
  - `LinearInterpolationMapping`: Faster approximation using linear interpolation
  - `CubicInterpolationMapping`: Memory-efficient approximation using cubic interpolation

## MomentSketch

MomentSketch is a moment-based approach to quantile estimation:

- **Compact Representation**: Stores only a fixed number of moments regardless of data size
- **Maximum Entropy Optimization**: Estimates the underlying distribution with high accuracy
- **Mergeable**: Supports combining sketches from different data sources
- **Compression Support**: Optional data compression for handling widely distributed values
- **Summary Statistics**: Provides comprehensive statistics beyond just quantiles

Both algorithms are designed to be memory-efficient and accurate, making them suitable for streaming data applications where traditional approaches would require excessive memory.