"""
Core implementation of the moment-based sketching algorithm for quantile estimation.

This module provides the core classes that implement the moment-based sketching algorithm:

- Moment: Stores and manages power sums, the fundamental data structure of the sketch
- MomentSolver: Base class for solving the maximum entropy problem using moments
- NewtonMS: Newton's method implementation for solving the maximum entropy problem
- SimpleMS: Main implementation class that provides the complete moment sketch algorithm

The moment-based approach uses power sums to track the distribution of data
and maximum entropy optimization to estimate quantiles. This provides a memory-efficient
way to estimate quantiles with controllable accuracy.
"""

import math
import numpy as np
from scipy.optimize import minimize
from .utils import MaxEntropy, Util
from .optimizer import NewtonOptimizer

class Moment:
    def __init__(self, k=None, power_sums=None, min_val=None, max_val=None):
        if power_sums is not None:
            self.power_sums = np.array(power_sums)
            self.min_val = min_val
            self.max_val = max_val
        else:
            self.power_sums = np.zeros(k)
            self.min_val = float('inf')
            self.max_val = float('-inf')

    def add_value(self, x) -> None:
        """Add a single value to the sketch"""
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)
        powers = np.power(x, np.arange(len(self.power_sums)))
        self.power_sums += powers

    def add_values(self, values) -> None:
        """Add multiple values to the sketch"""
        if len(values) == 0:
            return
        self.min_val = min(self.min_val, np.min(values))
        self.max_val = max(self.max_val, np.max(values))

        # Compute powers for all values at once
        for i, x in enumerate(values):
            powers = np.power(x, np.arange(len(self.power_sums)))
            self.power_sums += powers

    def merge(self, other: "Moment") -> None:
        """Merge another Moment into this one"""
        self.min_val = min(self.min_val, other.min_val)
        self.max_val = max(self.max_val, other.max_val)
        self.power_sums += other.power_sums

    def __str__(self) -> str:
        """String representation of the moments"""
        return f"{self.min_val:g}:{self.max_val:g}:{self.power_sums}"


class MomentSolver:
    def __init__(self, moments):
        self.x_min = moments.min_val
        self.x_max = moments.max_val
        self.x_center = (self.x_max + self.x_min) / 2
        self.x_scale = (self.x_max - self.x_min) / 2

        self.c_moments = Util.power_sums_to_cheby_moments(
            self.x_min, self.x_max, moments.power_sums
        )

        self.grid_size = 1024
        self.max_iter = 15
        self.verbose = False

        self.xs = None
        self.lambd = None
        self.weights = None

    def set_grid_size(self, size: int) -> None:
        """Set the grid size"""
        self.grid_size = size

    def set_max_iter(self, max_iter: int) -> None:
        """Set the maximum number of iterations"""
        self.max_iter = max_iter

    def set_verbose(self, flag):
        """Set verbose flag"""
        self.verbose = flag

    def solve(self) -> None:
        """Solve for quantiles"""
        n = self.grid_size
        P = MaxEntropy(self.c_moments, self.grid_size)

        # Initialize lambda
        l0 = np.zeros(len(self.c_moments))
        l0[0] = np.log(1.0 / n)

        # Define cost function for scipy.optimize
        def cost_function(x):
            val = P.compute_all(x)
            return val

        def gradient(x):
            P.compute_all(x)
            return P.get_gradient()

        def hessian(x):
            P.compute_all(x)
            return P.get_hessian()

        # Use scipy's minimize with Newton-CG method
        result = minimize(
            cost_function,
            l0,
            method='Newton-CG',
            jac=gradient,
            hess=hessian,
            options={'maxiter': self.max_iter, 'disp': self.verbose}
        )

        self.lambd = result.x
        P.compute_all(self.lambd)
        self.weights = P.weights

        # Create grid points
        self.xs = np.linspace(-1, 1, self.grid_size)
        self.xs = self.xs * self.x_scale + self.x_center

    def get_quantiles(self, ps):
        """Get multiple quantiles"""
        if self.weights is None:
            raise ValueError("You must call solve() before getting quantiles")

        # Compute CDF
        cdf = np.cumsum(self.weights)
        cdf /= cdf[-1]  # Normalize

        # Find quantiles using binary search
        qs = np.zeros(len(ps))
        for i, p in enumerate(ps):
            if p <= cdf[0]:
                qs[i] = self.xs[0]
            elif p >= cdf[-1]:
                qs[i] = self.xs[-1]
            else:
                # Binary search
                idx = np.searchsorted(cdf, p)
                qs[i] = self.xs[idx]

        return qs

    def get_quantile(self, p):
        """Get a single quantile"""
        return self.get_quantiles([p])[0]

    def get_xs(self):
        """Get grid points"""
        return self.xs

    def get_weights(self):
        """Get weights"""
        return self.weights


class NewtonMS(Moment):
    def __init__(self, moments):
        # Initialize the Moment part
        super().__init__(power_sums=moments.power_sums,
                         min_val=moments.min_val,
                         max_val=moments.max_val)
        # Use inherited attributes
        self.x_min = self.min_val
        self.x_max = self.max_val
        self.x_center = (self.x_max + self.x_min) / 2.0
        self.x_scale = (self.x_max - self.x_min) / 2.0

        # Compute Chebyshev moments
        self.c_moments = Util.power_sums_to_cheby_moments(
            self.x_min, self.x_max, self.power_sums
        )

        self.grid_size = 1024
        self.max_iter = 15
        self.verbose = False

        self.xs = None
        self.lambd = None
        self.weights = None

    def set_grid_size(self, size: int) -> None:
        self.grid_size = size

    def set_max_iter(self, max_iter: int) -> None:
        self.max_iter = max_iter

    def set_verbose(self, flag: bool) -> None:
        self.verbose = flag

    def solve(self) -> None:
        """Solve for optimal lambda parameters"""
        n = self.grid_size
        # Create max entropy engine
        P = MaxEntropy(self.c_moments, self.grid_size)

        # Initialize lambda
        l0 = np.zeros(len(self.c_moments))
        l0[0] = np.log(1.0 / n)

        # Use NewtonOptimizer
        opt = NewtonOptimizer(P)
        opt.set_verbose(self.verbose)
        opt.set_max_iterations(self.max_iter)
        self.lambd = opt.solve(l0, 1e-6)

        # Compute final values
        P.compute_all(self.lambd)
        self.weights = P.weights

        # Create grid points
        self.xs = np.linspace(-1, 1, self.grid_size)
        self.xs = self.xs * self.x_scale + self.x_center

    def get_quantiles(self, ps):
        """Get multiple quantiles"""
        if self.weights is None:
            raise ValueError("You must call solve() before getting quantiles")

        cdf = np.cumsum(self.weights)
        cdf /= cdf[-1]  # Normalize CDF

        qs = np.zeros(len(ps))
        for i, p in enumerate(ps):
            if p <= cdf[0]:
                qs[i] = self.xs[0]
            elif p >= cdf[-1]:
                qs[i] = self.xs[-1]
            else:
                idx = np.searchsorted(cdf, p)
                qs[i] = self.xs[idx]
        return qs

    def get_quantile(self, p):
        """Get a single quantile"""
        return self.get_quantiles([p])[0]

    def get_xs(self):
        """Return grid points"""
        return self.xs

    def get_weights(self):
        """Return weights"""
        return self.weights


class SimpleMS:
    def __init__(self, param):
        """
        Initialize a Sketch

        Args:
            param (int or Moment): Number of moments or existing Moment instance
        """
        if isinstance(param, int):
            self.data = Moment(param)
        elif isinstance(param, Moment):
            self.data = param
        else:
            raise ValueError("Parameter must be an int or Moment")
        self.use_arcsinh: bool = False  # Default to no compression

    def set_compressed(self, flag: bool) -> None:
        """Set compression flag"""
        self.use_arcsinh = flag

    def get_compressed(self) -> bool:
        """Get compression setting"""
        return self.use_arcsinh

    def get_k(self) -> int:
        """Get number of moments"""
        return len(self.data.power_sums)

    def get_power_sums(self):
        """Get power sums"""
        return self.data.power_sums

    def get_min(self) -> float:
        """Get minimum value"""
        return self.data.min_val

    def get_max(self) -> float:
        """Get maximum value"""
        return self.data.max_val

    def add(self, raw_x: float) -> None:
        """Add a data point to the sketch"""
        # Track min/max of raw values
        if self.data.power_sums[0] == 0:  # If this is the first value
            self.data.min_val = raw_x
            self.data.max_val = raw_x
        else:
            self.data.min_val = min(self.data.min_val, raw_x)
            self.data.max_val = max(self.data.max_val, raw_x)
            
        # Apply transform for internal representation
        x = math.asinh(raw_x) if self.use_arcsinh else raw_x
        
        # Update power sums with transformed value
        powers = np.power(x, np.arange(len(self.data.power_sums)))
        self.data.power_sums += powers

    def add_many(self, raw_x_vals):
        """Add multiple values to the sketch"""
        if len(raw_x_vals) == 0:
            return
            
        # Track min/max of raw values
        if self.data.power_sums[0] == 0:  # If these are the first values
            self.data.min_val = np.min(raw_x_vals)
            self.data.max_val = np.max(raw_x_vals)
        else:
            self.data.min_val = min(self.data.min_val, np.min(raw_x_vals))
            self.data.max_val = max(self.data.max_val, np.max(raw_x_vals))
        
        # Apply transform for internal representation
        if self.use_arcsinh:
            x_vals = np.arcsinh(np.array(raw_x_vals))
        else:
            x_vals = np.array(raw_x_vals)
        
        # Update power sums with transformed values
        for x in x_vals:
            powers = np.power(x, np.arange(len(self.data.power_sums)))
            self.data.power_sums += powers

    def merge(self, other: "SimpleMS") -> None:
        """Merge another Sketch"""
        # Check for incompatible compression settings
        if self.use_arcsinh != other.use_arcsinh:
            raise ValueError("Cannot merge sketches with different compression settings")
        self.data.merge(other.data)

    def get_solver(self) -> NewtonMS:
        """Get a solver"""
        return NewtonMS(self.data)

    def get_quantiles(self, fractions):
        """Get quantiles for fractions"""
        # Check if sketch is empty
        if self.data.power_sums[0] == 0:
            return [float('nan')] * len(fractions)
            
        # Handle edge case with only one value
        if self.data.min_val == self.data.max_val:
            return [self.data.min_val] * len(fractions)
            
        # If min and max are infinity values, sketch is effectively empty
        if not np.isfinite(self.data.min_val) or not np.isfinite(self.data.max_val):
            return [float('nan')] * len(fractions)
            
        try:
            # Create a solver that will compute the distribution
            solver = self.get_solver()
            solver.set_grid_size(1024)
            solver.set_max_iter(15)
            solver.solve()

            quantiles = []
            for frac in fractions:
                # Get the quantile in the transformed space
                transformed_q = solver.get_quantile(frac)
                
                # Apply inverse transform if compression was used
                if self.use_arcsinh:
                    q = math.sinh(transformed_q)
                else:
                    q = transformed_q
                    
                quantiles.append(q)
                
            # Ensure quantiles are within min and max bounds and handle non-finite values
            for i, q in enumerate(quantiles):
                if not np.isfinite(q):
                    # Use linear interpolation between min and max as fallback for this point
                    quantiles[i] = self.data.min_val + fractions[i] * (self.data.max_val - self.data.min_val)
                else:
                    # Clamp to min/max range
                    quantiles[i] = max(min(q, self.data.max_val), self.data.min_val)
                    
            return quantiles
        except Exception as e:
            # Fallback to linear interpolation between min and max for all points
            print(f"Warning: Quantile computation failed, using fallback. Error: {e}")
            return [self.data.min_val + frac * (self.data.max_val - self.data.min_val) for frac in fractions]

    def get_quantile(self, fraction):
        """Get a single quantile"""
        return self.get_quantiles([fraction])[0]

    def get_median(self):
        """Get median value"""
        return self.get_quantile(0.5)

    def get_percentile(self, p):
        """Get p-th percentile"""
        return self.get_quantile(p / 100.0)

    def get_iqr(self):
        """Get interquartile range"""
        qs = self.get_quantiles([0.25, 0.75])
        return qs[1] - qs[0]

    def get_stats(self):
        """Get summary statistics"""
        qs = self.get_quantiles([0, 0.25, 0.5, 0.75, 1])
        return {
            'min': qs[0],
            'q1': qs[1],
            'median': qs[2],
            'q3': qs[3],
            'max': qs[4],
            'count': self.data.power_sums[0],
            'mean': self.data.power_sums[1] / self.data.power_sums[0] if self.data.power_sums[0] > 0 else 0
        }

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'power_sums': self.data.power_sums.tolist(),
            'min_val': self.data.min_val,
            'max_val': self.data.max_val,
            'use_arc_sinh': self.use_arcsinh
        }

    @classmethod
    def from_dict(cls, data_dict):
        """Create from dictionary"""
        k = len(data_dict['power_sums'])
        sketch = cls(k)
        sketch.data = Moment(
            power_sums=data_dict['power_sums'],
            min_val=data_dict['min_val'],
            max_val=data_dict['max_val']
        )
        sketch.use_arcsinh = data_dict.get('use_arc_sinh', False)
        return sketch

    def __str__(self) -> str:
        """String representation"""
        return str(self.data)

    def plot_dist(self, bins=50, figsize=(10, 6)):
        """Plot distribution"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting. Install with 'pip install matplotlib'")

        # Create the figure
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

        try:
            solver = self.get_solver()
            solver.set_grid_size(1024)
            solver.solve()

            xs = solver.get_xs()
            weights = solver.get_weights()

            # Transform back if using compression
            if self.use_arcsinh:
                xs = np.sinh(xs)

            # Normalize weights
            weights = weights / np.sum(weights)

            # Plot
            ax.plot(xs, weights, 'b-', linewidth=2)
            ax.fill_between(xs, weights, alpha=0.3)
            ax.set_title('Estimated Probability Distribution')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            # If we fail to compute distribution, create an empty plot with error message
            ax.text(0.5, 0.5, f"Failed to compute distribution: {str(e)}",
                   horizontalalignment='center', verticalalignment='center')
            
        return fig