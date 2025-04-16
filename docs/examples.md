# Examples

## DDSketch Performance Comparison

Compare different mapping types for DDSketch:

```python
from QuantileFlow import DDSketch
import time
import numpy as np

# Generate test data
np.random.seed(42)
data = np.random.normal(0, 1, 100000)

# Test different mapping types
mappings = ['logarithmic', 'lin_interpol', 'cub_interpol']
results = {}

for mapping in mappings:
    # Create sketch with this mapping
    sketch = DDSketch(relative_accuracy=0.01, mapping_type=mapping)
    
    # Measure insertion time
    start = time.time()
    for value in data:
        sketch.insert(value)
    insert_time = time.time() - start
    
    # Measure query time
    start = time.time()
    for q in [0.1, 0.5, 0.9, 0.95, 0.99]:
        sketch.quantile(q)
    query_time = time.time() - start
    
    # Store results
    results[mapping] = {
        'insert_time': insert_time,
        'query_time': query_time,
        'p50': sketch.quantile(0.5),
        'p99': sketch.quantile(0.99)
    }

# Print comparison
print("Mapping Type Comparison:")
for mapping, metrics in results.items():
    print(f"\n{mapping}:")
    print(f"  Insert time: {metrics['insert_time']:.4f}s")
    print(f"  Query time: {metrics['query_time']:.4f}s")
    print(f"  p50: {metrics['p50']:.4f}")
    print(f"  p99: {metrics['p99']:.4f}")
```

## Handling Outliers with Sparse Storage

For datasets with outliers that span a wide range, sparse storage can be more appropriate:

```python
from QuantileFlow import DDSketch
from QuantileFlow.ddsketch import BucketManagementStrategy
import numpy as np

# Generate data with outliers
normal_data = np.random.normal(0, 1, 10000)
outliers = np.random.uniform(1000, 10000, 50)  # Few extreme values
mixed_data = np.concatenate([normal_data, outliers])

# Create sketches with different storage types
contiguous_sketch = DDSketch(
    relative_accuracy=0.01,
    bucket_strategy=BucketManagementStrategy.FIXED
)
sparse_sketch = DDSketch(
    relative_accuracy=0.01,
    bucket_strategy=BucketManagementStrategy.UNLIMITED
)

# Add data to both
for value in mixed_data:
    contiguous_sketch.insert(value)
    sparse_sketch.insert(value)

# Compare results
quantiles = [0.5, 0.9, 0.99, 0.999]
print("Contiguous vs Sparse Storage with Outliers:")
for q in quantiles:
    c_val = contiguous_sketch.quantile(q)
    s_val = sparse_sketch.quantile(q)
    print(f"q={q}: Contiguous={c_val:.4f}, Sparse={s_val:.4f}")
```

## MomentSketch for Summary Statistics

MomentSketch can provide additional statistics beyond quantiles:

```python
from QuantileFlow import MomentSketch
import numpy as np

# Generate data
data = np.random.normal(5, 2, 10000)

# Create MomentSketch
sketch = MomentSketch(num_moments=20)  # Using 20 moments for better accuracy

# Add data
for value in data:
    sketch.insert(value)

# Get summary statistics
stats = sketch.summary_statistics()

print("Summary Statistics:")
print(f"Min: {stats['min']:.4f}")
print(f"Q1: {stats['q1']:.4f}")
print(f"Median: {stats['median']:.4f}")
print(f"Q3: {stats['q3']:.4f}")
print(f"Max: {stats['max']:.4f}")
print(f"Count: {stats['count']}")
print(f"Mean: {stats['mean']:.4f}")

# Compare with actual values
actual_min = np.min(data)
actual_max = np.max(data)
actual_mean = np.mean(data)
print(f"\nActual Min: {actual_min:.4f}")
print(f"Actual Max: {actual_max:.4f}")
print(f"Actual Mean: {actual_mean:.4f}")

# Visualize the distribution
fig = sketch.plot_distribution()
```

## HDRHistogram for Wide-Range Data

HDRHistogram is particularly useful for data spanning multiple orders of magnitude:

```python
from QuantileFlow import HDRHistogram
import numpy as np

# Generate data spanning multiple orders of magnitude
data = np.concatenate([
    np.random.normal(1, 0.1, 1000),      # Values around 1
    np.random.normal(10, 1, 1000),       # Values around 10
    np.random.normal(100, 10, 1000),     # Values around 100
    np.random.normal(1000, 100, 1000)    # Values around 1000
])

# Create HDRHistogram with appropriate configuration
histogram = HDRHistogram(
    num_buckets=12,      # More buckets for better precision
    min_value=0.1,       # Minimum trackable value
    max_value=10000.0    # Maximum trackable value
)

# Add data
for value in data:
    histogram.insert(value)

# Get summary statistics
stats = histogram.summary_statistics()

print("Summary Statistics:")
print(f"Min: {stats['min']:.4f}")
print(f"Q1: {stats['q1']:.4f}")
print(f"Median: {stats['median']:.4f}")
print(f"Q3: {stats['q3']:.4f}")
print(f"Max: {stats['max']:.4f}")
print(f"Count: {stats['count']}")

# Compare with actual values
actual_min = np.min(data)
actual_max = np.max(data)
actual_median = np.median(data)
print(f"\nActual Min: {actual_min:.4f}")
print(f"Actual Max: {actual_max:.4f}")
print(f"Actual Median: {actual_median:.4f}")

# Visualize the distribution
fig = histogram.plot_distribution()
```

## Comparing All Three Algorithms

In the below code block, we show how to compare the performance and accuracy of all three algorithms. Note that HDRHistogram gives wildly inaccurate estimates because we're sampling from a log normal distribution (it's works perfectly if you just change the distribution to something with a smaller tail, such as log); we hypothesize this occurs because of two reasons: 

(1) Moments (especially higher ones) are extremely sensitive to the long tail in lognormal distributions. Each moment becomes increasingly dominated by the extreme values.

(2) The maximum entropy optimization used to reconstruct the distribution from moments struggles to accurately capture both the bulk and tail of highly skewed distributions.

```python
from QuantileFlow import DDSketch, MomentSketch, HDRHistogram
import numpy as np
import time

# Generate test data
np.random.seed(42)
data = np.random.lognormal(0, 2, 10000)  # Log-normal distribution

# Initialize all three algorithms
dd_sketch = DDSketch(relative_accuracy=0.01)
moment_sketch = MomentSketch(num_moments=20)
hdr_histogram = HDRHistogram(num_buckets=12)

# Time insertion
start = time.time()
for value in data:
    dd_sketch.insert(value)
dd_time = time.time() - start

start = time.time()
for value in data:
    moment_sketch.insert(value)
moment_time = time.time() - start

start = time.time()
for value in data:
    hdr_histogram.insert(value)
hdr_time = time.time() - start

# Get quantiles
quantiles = [0.5, 0.9, 0.99]
print("Performance Comparison:")
print(f"DDSketch insertion time: {dd_time:.4f}s")
print(f"MomentSketch insertion time: {moment_time:.4f}s")
print(f"HDRHistogram insertion time: {hdr_time:.4f}s")

print("\nQuantile Comparison:")
print(f"{'Quantile':>10} | {'DDSketch':>10} | {'MomentSketch':>12} | {'HDRHistogram':>12} | {'Actual':>10}")
print("-" * 70)

for q in quantiles:
    dd_val = dd_sketch.quantile(q)
    moment_val = moment_sketch.quantile(q)
    hdr_val = hdr_histogram.quantile(q)
    actual_val = np.quantile(data, q)
    
    print(f"{q:10.2f} | {dd_val:10.4f} | {moment_val:12.4f} | {hdr_val:12.4f} | {actual_val:10.4f}")
```