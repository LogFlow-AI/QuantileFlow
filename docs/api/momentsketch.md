# MomentSketch

MomentSketch is a moment-based quantile approximation algorithm that provides accurate quantile estimates using only a small number of statistical moments.

## Overview

MomentSketch offers the following features:

* **Compact Representation**: Stores only a fixed number of statistical moments regardless of data size
* **Maximum Entropy Optimization**: Uses maximum entropy optimization to estimate the underlying distribution
* **Mergeable**: Multiple sketches can be combined without loss of accuracy
* **Summary Statistics**: Provides comprehensive statistics including mean, variance, skewness, and kurtosis
* **Compression Support**: Optional data compression for handling widely distributed values

## Main Class

```{eval-rst}
.. autoclass:: QuantileFlow.MomentSketch
   :members:
   :undoc-members:
   :special-members: __init__
   :exclude-members: __dict__, __weakref__
   :noindex:
   :show-inheritance:
```

## Optimizer

The optimizer module handles the maximum entropy optimization required for quantile estimation:

```{eval-rst}
.. automodule:: QuantileFlow.momentsketch.optimizer
   :members:
   :undoc-members:
   :noindex:
   :show-inheritance:
```

## Utilities

```{eval-rst}
.. automodule:: QuantileFlow.momentsketch.utils
   :members:
   :undoc-members:
   :noindex:
   :show-inheritance:
```

## Mathematical Background

MomentSketch works by maintaining a fixed number of statistical moments (typically 10-20) of the data distribution. These moments are used to reconstruct an approximation of the original distribution using maximum entropy optimization.

The key steps in the algorithm are:

1. **Moment Collection**: Compute and store statistical moments (mean, variance, etc.) from the data stream
2. **Distribution Estimation**: Solve a maximum entropy optimization problem to find a distribution matching the stored moments
3. **Quantile Estimation**: Query the estimated distribution to find approximate quantiles

### Maximum Entropy Optimization

The maximum entropy principle seeks the probability distribution that maximizes entropy while satisfying constraints (in this case, matching the observed moments). This approach provides the least biased estimate possible based on the limited information available.

Mathematically, we solve:

$$\max_{p} -\int p(x) \log p(x) dx$$

Subject to:

$$\int x^k p(x) dx = \mu_k \quad \text{for} \quad k = 1, 2, \ldots, n$$

Where $\mu_k$ are the observed moments and $p(x)$ is the probability density function.

## Usage Examples

For basic usage, see the [Getting Started](../getting-started.md) guide.

For more advanced examples, see the [Examples](../examples.md) page. 