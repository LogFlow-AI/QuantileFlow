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
    sketch.add(value)

# Query quantiles
median = sketch.get_quantile_value(0.5)  # Get the median
p99 = sketch.get_quantile_value(0.99)    # Get the 99th percentile

# Get value rank
rank = sketch.get_rank(5.0)  # Gets approximate rank of value 5.0

print(f"Median: {median}")
print(f"99th percentile: {p99}")
print(f"Rank of 5.0: {rank}")
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

# Create a MomentSketch with 10 moments
sketch = MomentSketch(10)

# Add data points
for value in range(1000):
    sketch.add(value)

# Query quantiles
median = sketch.quantile(0.5)
p95 = sketch.quantile(0.95)

print(f"Median: {median}")
print(f"95th percentile: {p95}")
```

## Merging Sketches

Both DDSketch and MomentSketch support merging for distributed processing:

```python
# Create two sketches
sketch1 = DDSketch(relative_accuracy=0.01)
sketch2 = DDSketch(relative_accuracy=0.01)

# Add different data to each
for i in range(100):
    sketch1.add(i)
    
for i in range(100, 200):
    sketch2.add(i)
    
# Merge sketch2 into sketch1
sketch1.merge(sketch2)

# Now sketch1 contains the combined data
print(f"Median of combined data: {sketch1.get_quantile_value(0.5)}")
```

## Advanced Configuration

### DDSketch Storage Options

```python
# Using contiguous storage (default)
contiguous_sketch = DDSketch(
    relative_accuracy=0.01,
    storage_type='contiguous'
)

# Using sparse storage for widely distributed data
sparse_sketch = DDSketch(
    relative_accuracy=0.01,
    storage_type='sparse'
)
```

### MomentSketch with Compression

```python
from QuantileFlow import MomentSketch

# Create a MomentSketch with compression
sketch = MomentSketch(10, compress=True, max_value=1000)

# Add data
for value in range(1000):
    sketch.add(value)

# Query results as normal
median = sketch.quantile(0.5)
print(f"Compressed sketch median: {median}")
```