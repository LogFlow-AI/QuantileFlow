# Getting Started

## Installation

You can install QuantileFlow with pip:

```bash
pip install QuantileFlow
```

## Basic Usage

### DDSketch

```python
from QuantileFlow import DDSketch

# Create a DDSketch with 1% relative accuracy
sketch = DDSketch(relative_accuracy=0.01)

# Add data points
for value in [1.0, 2.5, 3.0, 4.2, 5.0, 6.8, 7.5, 8.1, 9.2, 10.0]:
    sketch.insert(value)

# Query quantiles
median = sketch.quantile(0.5)  # Get the median
p99 = sketch.quantile(0.99)    # Get the 99th percentile

print(f"Median: {median}")
print(f"99th percentile: {p99}")
```

### Different Mapping Types

```python
# Using logarithmic mapping (default)
log_sketch = DDSketch(relative_accuracy=0.01, mapping_type='logarithmic')

# Using linear interpolation mapping
lin_sketch = DDSketch(relative_accuracy=0.01, mapping_type='lin_interpol')

# Using cubic interpolation mapping
cub_sketch = DDSketch(relative_accuracy=0.01, mapping_type='cub_interpol')
```

### MomentSketch

```python
from QuantileFlow import MomentSketch

# Create a MomentSketch with 20 moments
sketch = MomentSketch(num_moments=20)

# Add data points
for value in range(1000):
    sketch.insert(value)

# Query quantiles
median = sketch.quantile(0.5)
p95 = sketch.quantile(0.95)

# Get summary statistics
stats = sketch.summary_statistics()

print(f"Median: {median}")
print(f"95th percentile: {p95}")
print(f"Summary statistics: {stats}")
```

### HDRHistogram

```python
from QuantileFlow import HDRHistogram

# Create an HDRHistogram with 8 buckets
histogram = HDRHistogram(num_buckets=8)

# Add data points
for value in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]:
    histogram.insert(value)

# Query quantiles
median = histogram.quantile(0.5)
p95 = histogram.quantile(0.95)

# Get summary statistics
stats = histogram.summary_statistics()

print(f"Median: {median}")
print(f"95th percentile: {p95}")
print(f"Summary statistics: {stats}")

# Visualize the distribution
fig = histogram.plot_distribution()
```

## Merging Sketches

All three algorithms support merging for distributed processing:

```python
from QuantileFlow import DDSketch

# Create two histograms
hist1 = DDSketch(0.01)
hist2 = DDSketch(0.01)

# Add different data to each
for i in range(100):
    hist1.insert(i)
    
for i in range(100, 200):
    hist2.insert(i)
    
# Merge hist2 into hist1
hist1.merge(hist2)

# Now hist1 contains the combined data
print(f"Median of combined data: {hist1.quantile(0.5)}")
```

## Advanced Configuration

### DDSketch Storage Options

```python
# Using contiguous storage (default)
contiguous_sketch = DDSketch(
    relative_accuracy=0.01,
    bucket_strategy=BucketManagementStrategy.FIXED
)

# Using sparse storage for widely distributed data
sparse_sketch = DDSketch(
    relative_accuracy=0.01,
    bucket_strategy=BucketManagementStrategy.UNLIMITED
)

# Using dynamic bucket management
dynamic_sketch = DDSketch(
    relative_accuracy=0.01,
    bucket_strategy=BucketManagementStrategy.DYNAMIC
)
```

### MomentSketch with Compression

```python
from QuantileFlow import MomentSketch

# Create a MomentSketch with compression
sketch = MomentSketch(num_moments=20, compress_values=True)

# Add data
for value in range(1000):
    sketch.insert(value)

# Query results as normal
median = sketch.quantile(0.5)
print(f"Compressed sketch median: {median}")

# Visualize the distribution
fig = sketch.plot_distribution()
```

### HDRHistogram with Value Range Control

```python
from QuantileFlow import HDRHistogram

# Create an HDRHistogram with value range control
histogram = HDRHistogram(
    num_buckets=8,
    min_value=1.0,    # Minimum trackable value
    max_value=1000.0  # Maximum trackable value
)

# Add data
for value in range(1, 1001):
    histogram.insert(value)

# Query results as normal
median = histogram.quantile(0.5)
print(f"Histogram median: {median}")

# Visualize the distribution
fig = histogram.plot_distribution()
```