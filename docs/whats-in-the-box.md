# What's in The Box?

QuantileFlow provides three main algorithms for quantile estimation:

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
- **Bucket Management**:
  - `FIXED`: Fixed maximum number of buckets
  - `UNLIMITED`: No limit on number of buckets
  - `DYNAMIC`: Dynamic limit based on log(n)

## MomentSketch

MomentSketch is a moment-based approach to quantile estimation:

- **Compact Representation**: Stores only a fixed number of moments regardless of data size
- **Maximum Entropy Optimization**: Estimates the underlying distribution with high accuracy
- **Mergeable**: Supports combining sketches from different data sources
- **Compression Support**: Optional arcsinh transformation for handling widely distributed values
- **Summary Statistics**: Provides comprehensive statistics including min, max, quartiles, mean, and count
- **Visualization**: Built-in support for plotting estimated distributions
- **Serialization**: Supports converting sketches to/from dictionaries for storage and transmission

## HDRHistogram

HDRHistogram (High Dynamic Range Histogram) is a logarithmic-bucketed histogram implementation:

- **Wide Value Range**: Efficiently tracks values across multiple orders of magnitude
- **Configurable Precision**: Adjustable number of buckets for different accuracy needs
- **Memory Efficient**: Uses logarithmic bucketing to minimize memory usage
- **Summary Statistics**: Provides comprehensive statistics including min, max, quartiles, and count
- **Visualization**: Built-in support for plotting distributions with logarithmic scales
- **Serialization**: Supports converting histograms to/from dictionaries for storage and transmission
- **Value Range Control**: Configurable minimum and maximum trackable values

All algorithms are designed to be memory-efficient and accurate, making them suitable for streaming data applications where traditional approaches would require excessive memory.