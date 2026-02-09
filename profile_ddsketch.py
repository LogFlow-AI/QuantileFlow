import cProfile
import pstats
import io
import numpy as np  # Using numpy for faster random data generation

from QuantileFlow.ddsketch.core import DDSketch

def run_sketch_operations():
    """Runs typical DDSketch operations for profiling."""
    print("Initializing DDSketch...")
    sketch = DDSketch(relative_accuracy=0.01)
    
    num_values = 1_000_000
    print(f"Inserting {num_values} random values...")
    # Generate random data more efficiently with numpy
    data = np.random.rand(num_values) * 1000
    for value in data:
        sketch.insert(value)
        
    quantiles_to_compute = [0.5, 0.9, 0.99, 0.999]
    print(f"Computing quantiles: {quantiles_to_compute}...")
    for q in quantiles_to_compute:
        try:
            quantile_value = sketch.quantile(q)
            print(f"Quantile({q}): {quantile_value}")
        except ValueError as e:
            print(f"Error computing quantile {q}: {e}")

    print("Profiling complete.")

def profile():
    """Profiles the run_sketch_operations function."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_sketch_operations()
    
    profiler.disable()
    
    s = io.StringIO()
    # Sort stats by cumulative time spent in the function and its callees
    sortby = pstats.SortKey.CUMULATIVE 
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats()
    
    print("\n--- cProfile Results (Sorted by Cumulative Time) ---")
    print(s.getvalue())

if __name__ == "__main__":
    profile() 