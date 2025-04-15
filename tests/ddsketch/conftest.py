import pytest
import numpy as np
from QuantileFlow.ddsketch.core import DDSketch
from QuantileFlow.ddsketch.storage.base import BucketManagementStrategy

@pytest.fixture(params=[0.01, 0.001, 0.1])
def relative_accuracy(request):
    return request.param

@pytest.fixture(params=[32, 64, 128])
def max_buckets(request):
    return request.param

@pytest.fixture(params=['logarithmic', 'lin_interpol', 'cub_interpol'])
def mapping_type(request):
    return request.param

@pytest.fixture(params=[True, False])
def cont_neg(request):
    return request.param

@pytest.fixture(params=[
    BucketManagementStrategy.FIXED,
    BucketManagementStrategy.DYNAMIC,
    BucketManagementStrategy.UNLIMITED
])
def bucket_strategy(request):
    return request.param

@pytest.fixture
def basic_sketch():
    """Returns a basic DDSketch instance with default parameters"""
    return DDSketch(relative_accuracy=0.01)

@pytest.fixture
def populated_sketch():
    """Returns a DDSketch instance populated with some test data"""
    sketch = DDSketch(relative_accuracy=0.01)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in values:
        sketch.insert(v)
    return sketch

@pytest.fixture
def random_data():
    """Returns random test data from different distributions"""
    np.random.seed(42)
    return {
        'uniform': np.random.uniform(1, 100, 1000),
        'normal': np.random.normal(50, 10, 1000),
        'lognormal': np.random.lognormal(0, 1, 1000),
        'exponential': np.random.exponential(10, 1000)
    }

@pytest.fixture
def mixed_sign_sketch():
    """Returns a DDSketch instance with both positive and negative values"""
    sketch = DDSketch(relative_accuracy=0.01, cont_neg=True)
    values = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]
    for v in values:
        sketch.insert(v)
    return sketch 