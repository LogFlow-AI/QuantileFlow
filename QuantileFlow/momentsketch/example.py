"""
Example script demonstrating the usage of MomentSketch for quantile estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from QuantileFlow import MomentSketch

def basic_usage_demo():
    """Demonstrate basic usage of the MomentSketch"""
    print("\nBasic MomentSketch Usage Example")
    print("-" * 50)

    # Generate sample data - skewed distribution
    np.random.seed(8990)
    data = np.concatenate([
        np.random.normal(10, 2, 10000),
        np.random.exponential(5, 5000)
    ])

    # Create a MomentSketch with 20 moments
    sketch = MomentSketch(num_moments=20)

    # Add data to the sketch
    start_time = time.time()
    sketch.insert_batch(data)
    sketch_time = time.time() - start_time

    # Get quantiles
    start_time = time.time()
    percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    quantiles = sketch.quantiles(percentiles)
    query_time = time.time() - start_time

    # Calculate true quantiles for comparison
    true_quantiles = np.quantile(data, percentiles)

    # Print results
    print(f"Data size: {len(data):,d} points")
    print(f"Time to build sketch: {sketch_time * 1000:.2f} ms")
    print(f"Time to query quantiles: {query_time * 1000:.2f} ms")
    print("\nQuantile Comparison:")
    print(f"{'Percentile':>10} | {'True':>10} | {'Estimated':>10} | {'Error %':>10}")
    print("-" * 50)

    for p, tq, eq in zip(percentiles, true_quantiles, quantiles):
        error_pct = 100 * abs(tq - eq) / abs(tq) if tq != 0 else 0
        print(f"{p * 100:10.1f}% | {tq:10.4f} | {eq:10.4f} | {error_pct:10.2f}%")

    # Get summary statistics
    stats = sketch.summary_statistics()
    print("\nSummary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return sketch, data


def distribution_demo(sketch, data):
    """Demonstrate plotting the estimated distribution"""
    print("\nDistribution Visualization")
    print("-" * 50)
    
    # Plot the distribution
    fig = sketch.plot_distribution(figsize=(10, 6))
    
    # Add a histogram of the original data for comparison
    ax = fig.axes[0]
    ax.hist(data, bins=50, density=True, alpha=0.5, color='blue', label='Original Data')
    ax.legend()
    
    return fig


def merge_demo():
    """Demonstrate merging multiple sketches"""
    print("\nMerging Sketches Example")
    print("-" * 50)

    # Create two different data streams
    np.random.seed(8990)
    data1 = np.random.normal(1, 2, 10000)
    data2 = np.random.normal(2, 3, 15000)

    # Create sketches for each stream
    sketch1 = MomentSketch(num_moments=20)
    sketch2 = MomentSketch(num_moments=20)

    # Add data to sketches
    sketch1.insert_batch(data1)
    sketch2.insert_batch(data2)

    # Create a combined sketch by merging
    combined_sketch = MomentSketch(num_moments=20)
    combined_sketch.merge(sketch1)
    combined_sketch.merge(sketch2)

    # Create a reference by combining the data
    combined_data = np.concatenate([data1, data2])

    # Compare median estimates
    median1 = sketch1.median()
    median2 = sketch2.median()
    combined_median = combined_sketch.median()
    true_combined_median = np.median(combined_data)

    print(f"Data 1 size: {len(data1):,d}, Median: {median1:.4f}")
    print(f"Data 2 size: {len(data2):,d}, Median: {median2:.4f}")
    print(f"Combined data size: {len(combined_data):,d}")
    print(f"True combined median: {true_combined_median:.4f}")
    print(f"Estimated combined median: {combined_median:.4f}")
    print(f"Error: {100 * abs(combined_median - true_combined_median) / true_combined_median:.2f}%")

    return combined_sketch, combined_data


def serialization_demo():
    """Demonstrate sketch serialization"""
    print("\nSerialization Example")
    print("-" * 50)

    # Create and populate a sketch
    np.random.seed(8990)
    data = np.random.lognormal(0, 1, 20000)

    sketch = MomentSketch(num_moments=20)
    sketch.insert_batch(data)

    # Serialize to dictionary
    sketch_dict = sketch.to_dict()
    print(f"Serialized sketch keys: {list(sketch_dict.keys())}")

    # Deserialize
    restored_sketch = MomentSketch.from_dict(sketch_dict)

    # Compare results
    original_quantiles = sketch.quantiles([0.25, 0.5, 0.75])
    restored_quantiles = restored_sketch.quantiles([0.25, 0.5, 0.75])

    print("\nQuantile comparison:")
    print(f"{'Percentile':>10} | {'Original':>10} | {'Restored':>10}")
    print("-" * 50)

    for p, oq, rq in zip([0.25, 0.5, 0.75], original_quantiles, restored_quantiles):
        print(f"{p * 100:10.1f}% | {oq:10.4f} | {rq:10.4f}")

    return sketch_dict


if __name__ == "__main__":
    # Run all demonstrations
    sketch, data = basic_usage_demo()
    fig = distribution_demo(sketch, data)
    
    combined_sketch, combined_data = merge_demo()
    serialized_sketch = serialization_demo()
    
    plt.tight_layout()
    plt.show() 