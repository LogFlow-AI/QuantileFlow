from QuantileFlow.ddsketch import DDSketch as QuantileFlowDDSketch
from QuantileFlow.momentsketch import MomentSketch as QuantileFlowMomentSketch
from ddsketch import DDSketch as DDsketch
import numpy as np

# Generate random data
data = np.random.normal(0, 1, 1000)

# Initialize sketches
qf_dd_sketch = QuantileFlowDDSketch(0.01)  # 1% relative accuracy
qf_moment_sketch = QuantileFlowMomentSketch(10)  # 10 moments
dd_sketch = DDsketch(0.01)  # 1% relative accuracy

# Insert data into sketches
for val in data:
    dd_sketch.add(val)
    qf_moment_sketch.insert(val)
    qf_dd_sketch.insert(val)

# Compute percentiles
print('True P50:', np.percentile(data, 50))
print('True P95:', np.percentile(data, 95))
print('True P99:', np.percentile(data, 99))
print()
print('DDSketch P50:', dd_sketch.get_quantile_value(0.5))
print('DDSketch P95:', dd_sketch.get_quantile_value(0.95))
print('DDSketch P99:', dd_sketch.get_quantile_value(0.99))
print()
print('QuantileFlow MomentSketch P50:', qf_moment_sketch.quantile(0.5))
print('QuantileFlow MomentSketch P95:', qf_moment_sketch.quantile(0.95))
print('QuantileFlow MomentSketch P99:', qf_moment_sketch.quantile(0.99)) 
print()
print('QuantileFlow DDSketch P50:', qf_dd_sketch.quantile(0.5))
print('QuantileFlow DDSketch P95:', qf_dd_sketch.quantile(0.95))
print('QuantileFlow DDSketch P99:', qf_dd_sketch.quantile(0.99)) 