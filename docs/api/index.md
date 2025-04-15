# API Reference

This section provides detailed documentation for all the public classes and methods available in the QuantileFlow package.

## Core Components

### DDSketch

```{eval-rst}
.. autoclass:: QuantileFlow.DDSketch
   :members:
   :undoc-members:
   :show-inheritance:
```

### MomentSketch

```{eval-rst}
.. autoclass:: QuantileFlow.MomentSketch
   :members:
   :undoc-members:
   :show-inheritance:
```

## Mapping Classes

These classes implement different mapping strategies for DDSketch:

```{eval-rst}
.. autoclass:: QuantileFlow.LogarithmicMapping
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: QuantileFlow.LinearInterpolationMapping
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: QuantileFlow.CubicInterpolationMapping
   :members:
   :undoc-members:
   :show-inheritance:
```

## Storage Classes

These classes implement different storage strategies for DDSketch:

```{eval-rst}
.. autoclass:: QuantileFlow.ContiguousStorage
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: QuantileFlow.SparseStorage
   :members:
   :undoc-members:
   :show-inheritance:
```

## Utility Classes

```{eval-rst}
.. autoclass:: QuantileFlow.BucketManagementStrategy
   :members:
   :undoc-members:
   :show-inheritance:
``` 