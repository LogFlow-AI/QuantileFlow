# DDSketch

The DDSketch (Distributed and Deterministic Sketch) is a quantile sketch algorithm that provides accuracy guarantees for quantile estimation.

## Overview

DDSketch offers the following features:

* **Relative Error Guarantee**: Estimates quantiles with a relative error that is at most the configured accuracy parameter
* **Mergeable**: Multiple sketches can be combined without loss of accuracy
* **Memory Efficient**: Uses buckets to approximate the distribution, requiring less memory than storing all data points
* **Fast Operations**: Provides O(1) time complexity for adding values and querying quantiles

## Main Class

```{eval-rst}
.. autoclass:: QuantileFlow.DDSketch
   :members:
   :undoc-members:
   :special-members: __init__
   :exclude-members: __dict__, __weakref__
   :noindex:
   :show-inheritance:
```

## Mapping Types

DDSketch supports different mapping strategies that translate input values to bucket indices:

### Logarithmic Mapping

The canonical implementation with provable error guarantees:

```{eval-rst}
.. autoclass:: QuantileFlow.LogarithmicMapping
   :members:
   :undoc-members:
   :special-members: __init__
   :exclude-members: __dict__, __weakref__
   :noindex:
   :show-inheritance:
```

### Linear Interpolation Mapping

An approximation using linear interpolation for faster computation:

```{eval-rst}
.. autoclass:: QuantileFlow.LinearInterpolationMapping
   :members:
   :undoc-members:
   :special-members: __init__
   :exclude-members: __dict__, __weakref__
   :noindex:
   :show-inheritance:
```

### Cubic Interpolation Mapping

An approximation using cubic interpolation for better memory efficiency:

```{eval-rst}
.. autoclass:: QuantileFlow.CubicInterpolationMapping
   :members:
   :undoc-members:
   :special-members: __init__
   :exclude-members: __dict__, __weakref__
   :noindex:
   :show-inheritance:
```

## Storage Types

DDSketch provides two storage implementations for different use cases:

### Contiguous Storage

An array-based storage optimized for limited bucket ranges:

```{eval-rst}
.. autoclass:: QuantileFlow.ContiguousStorage
   :members:
   :undoc-members:
   :special-members: __init__
   :exclude-members: __dict__, __weakref__
   :noindex:
   :show-inheritance:
```

### Sparse Storage

A hash-based storage for handling wider bucket ranges:

```{eval-rst}
.. autoclass:: QuantileFlow.SparseStorage
   :members:
   :undoc-members:
   :special-members: __init__
   :exclude-members: __dict__, __weakref__
   :noindex:
   :show-inheritance:
```

## Bucket Management

```{eval-rst}
.. autoclass:: QuantileFlow.BucketManagementStrategy
   :members:
   :undoc-members:
   :special-members: __init__
   :exclude-members: __dict__, __weakref__
   :noindex:
   :show-inheritance:
```

## Usage Examples

For basic usage, see the [Getting Started](../getting-started.md) guide.

For more advanced examples, see the [Examples](../examples.md) page. 