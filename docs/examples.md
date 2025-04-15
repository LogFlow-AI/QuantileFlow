# Examples

This page contains various examples of how to use the QuantileFlow library for different scenarios.

## Streaming Data Analysis

When processing a continuous stream of data, sketches can be updated incrementally:

```python
from QuantileFlow import DDSketch
import random

# Initialize sketch
sketch = DDSketch(relative_accuracy=0.01)

# Simulate a data stream
for _ in range(10000):
    # Process each new data point as it arrives
    value = random.expovariate(0.5)  # Example: exponential distribution
    sketch.add(value)
    
    # Periodically check quantiles (e.g., every 1000 points)
    if _ % 1000 == 0 and _ > 0:
        p50 = sketch.get_quantile_value(0.5)
        p95 = sketch.get_quantile_value(0.95)
        p99 = sketch.get_quantile_value(0.99)
        print(f"After {_} points - median: {p50:.2f}, p95: {p95:.2f}, p99: {p99:.2f}")
```

## Distributed Processing

For distributed systems, you can process data in parallel and merge the results:

```python
from QuantileFlow import DDSketch
import numpy as np
from multiprocessing import Pool

def process_chunk(chunk):
    # Process one chunk of data
    sketch = DDSketch(relative_accuracy=0.01)
    for value in chunk:
        sketch.add(value)
    return sketch

# Generate sample data
data = np.random.lognormal(0, 2, size=1000000)

# Split data into chunks
chunks = np.array_split(data, 10)

# Process chunks in parallel
with Pool(4) as pool:
    sketches = pool.map(process_chunk, chunks)
    
# Merge results
final_sketch = sketches[0]
for sketch in sketches[1:]:
    final_sketch.merge(sketch)
    
# Query final results
quantiles = [0.1, 0.5, 0.9, 0.95, 0.99]
for q in quantiles:
    print(f"Quantile {q:.2f}: {final_sketch.get_quantile_value(q):.4f}")
```

## Comparing Mapping Types

Different mapping types offer trade-offs between speed, accuracy, and memory usage:

```python
from QuantileFlow import DDSketch
import numpy as np
import time

# Generate test data
data = np.random.lognormal(0, 2, size=100000)

# Test different mapping types
mappings = ['logarithmic', 'lin_interpol', 'cub_interpol']
results = {}

for mapping in mappings:
    # Create sketch with this mapping
    sketch = DDSketch(relative_accuracy=0.01, mapping_type=mapping)
    
    # Measure insertion time
    start = time.time()
    for value in data:
        sketch.add(value)
    insert_time = time.time() - start
    
    # Measure query time
    start = time.time()
    for q in [0.1, 0.5, 0.9, 0.95, 0.99]:
        sketch.get_quantile_value(q)
    query_time = time.time() - start
    
    # Calculate memory usage (approximation)
    memory = sketch.get_storage_size()
    
    # Store results
    results[mapping] = {
        'insert_time': insert_time,
        'query_time': query_time,
        'memory': memory,
        'p50': sketch.get_quantile_value(0.5),
        'p99': sketch.get_quantile_value(0.99)
    }

# Print comparison
print("Mapping Type Comparison:")
for mapping, metrics in results.items():
    print(f"\n{mapping}:")
    print(f"  Insert time: {metrics['insert_time']:.4f}s")
    print(f"  Query time: {metrics['query_time']:.4f}s")
    print(f"  Memory usage: {metrics['memory']} buckets")
    print(f"  p50: {metrics['p50']:.4f}")
    print(f"  p99: {metrics['p99']:.4f}")
```

## Handling Outliers with Sparse Storage

For datasets with outliers that span a wide range, sparse storage can be more appropriate:

```python
from QuantileFlow import DDSketch
import numpy as np

# Generate data with outliers
normal_data = np.random.normal(0, 1, 10000)
outliers = np.random.uniform(1000, 10000, 50)  # Few extreme values
mixed_data = np.concatenate([normal_data, outliers])

# Create sketches with different storage types
contiguous_sketch = DDSketch(relative_accuracy=0.01, storage_type='contiguous')
sparse_sketch = DDSketch(relative_accuracy=0.01, storage_type='sparse')

# Add data to both
for value in mixed_data:
    contiguous_sketch.add(value)
    sparse_sketch.add(value)

# Compare results
quantiles = [0.5, 0.9, 0.99, 0.999]
print("Contiguous vs Sparse Storage with Outliers:")
for q in quantiles:
    c_val = contiguous_sketch.get_quantile_value(q)
    s_val = sparse_sketch.get_quantile_value(q)
    print(f"q={q}: Contiguous={c_val:.4f}, Sparse={s_val:.4f}")

# Compare memory usage
c_size = contiguous_sketch.get_storage_size()
s_size = sparse_sketch.get_storage_size()
print(f"Storage size - Contiguous: {c_size} buckets, Sparse: {s_size} buckets")
```

## MomentSketch for Summary Statistics

MomentSketch can provide additional statistics beyond quantiles:

```python
from QuantileFlow import MomentSketch
import numpy as np

# Generate data
data = np.random.normal(5, 2, 10000)

# Create MomentSketch
sketch = MomentSketch(20)  # Using 20 moments for better accuracy

# Add data
for value in data:
    sketch.add(value)

# Get summary statistics
mean = sketch.mean()
variance = sketch.variance()
std_dev = sketch.std_dev()
skewness = sketch.skewness()
kurtosis = sketch.kurtosis()

print("Summary Statistics:")
print(f"Mean: {mean:.4f}")
print(f"Variance: {variance:.4f}")
print(f"Standard Deviation: {std_dev:.4f}")
print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurtosis:.4f}")

# Compare with actual values
actual_mean = np.mean(data)
actual_std = np.std(data)
print(f"\nActual Mean: {actual_mean:.4f}")
print(f"Actual Std Dev: {actual_std:.4f}")
``` 