"""
Tests for the Utils module.

This module contains unit tests for the utility classes used in the MomentSketch implementation:
- Util: Statistical utilities and mathematical helper functions
- MaxEntropy: Implementation of the maximum entropy optimization problem
- QuadraticFunction: Test function for the optimization algorithms
"""
import math
import numpy as np
import pytest

from QuantileFlow.momentsketch.utils import Util, MaxEntropy, QuadraticFunction


class TestUtil:
    """Test suite for the Util class."""

    def test_get_binomial_coefficients(self):
        """Test computation of binomial coefficients."""
        coeffs = Util.get_binomial_coefficients(3)
        
        # Check dimensions
        assert len(coeffs) == 4
        assert all(len(row) == 4 for row in coeffs)
        
        # Check specific values
        # Pascal's triangle:
        # 1
        # 1 1
        # 1 2 1
        # 1 3 3 1
        assert coeffs[0][0] == 1
        assert coeffs[1][0] == 1
        assert coeffs[1][1] == 1
        assert coeffs[2][0] == 1
        assert coeffs[2][1] == 2
        assert coeffs[2][2] == 1
        assert coeffs[3][0] == 1
        assert coeffs[3][1] == 3
        assert coeffs[3][2] == 3
        assert coeffs[3][3] == 1
        
    def test_get_cheby_coefficients(self):
        """Test computation of Chebyshev polynomial coefficients."""
        coeffs = Util.get_cheby_coefficients(3)
        
        # Check dimensions
        assert len(coeffs) == 4
        assert all(len(row) == 4 for row in coeffs)
        
        # Check specific values for first few Chebyshev polynomials
        # T_0(x) = 1
        # T_1(x) = x
        # T_2(x) = 2x^2 - 1
        # T_3(x) = 4x^3 - 3x
        assert coeffs[0][0] == 1
        assert coeffs[1][1] == 1
        assert coeffs[2][0] == -1
        assert coeffs[2][2] == 2
        assert coeffs[3][1] == -3
        assert coeffs[3][3] == 4
        
    def test_shift_power_sums(self):
        """Test shifting and scaling power sums."""
        # Simple example: [1, 3, 9, 27] - representing sum of 1, x, x^2, x^3 for x=3
        power_sums = [1, 3, 9, 27]
        
        # Shift to center at 0 and scale to interval [-1, 1]
        shifted = Util.shift_power_sums(power_sums, 3, 3)
        
        # After shifting and scaling, the power sums should represent x=0
        # So the first value should still be 1 (count), but other powers should be close to 0
        assert shifted[0] == 1.0
        assert abs(shifted[1]) < 1e-10
        assert abs(shifted[2]) < 1e-10
        assert abs(shifted[3]) < 1e-10
        
    def test_power_sums_to_normalized_moments(self):
        """Test conversion of power sums to normalized moments."""
        # Power sums for 2 values: 2 and 4
        power_sums = [2, 6, 20, 68]  # count, sum, sum of squares, sum of cubes
        
        # Min and max
        min_val = 2.0
        max_val = 4.0
        
        # Compute normalized moments
        normalized = Util.power_sums_to_normalized_moments(power_sums, min_val, max_val)
        
        # The normalized moments should have mean 0 and be in range [-1, 1]
        assert abs(normalized[1]) < 1e-10  # Mean should be close to 0
        assert -1.0 <= normalized[2] <= 1.0  # Second moment should be in [-1, 1]
        
    def test_power_sums_to_cheby_moments(self):
        """Test conversion of power sums to Chebyshev moments."""
        # Power sums for values 0, 1, 2, 3, 4 (count=5, sum=10, etc.)
        power_sums = [5, 10, 30, 100, 354]
        
        # Min and max
        min_val = 0.0
        max_val = 4.0
        
        # Compute Chebyshev moments
        cheby_moments = Util.power_sums_to_cheby_moments(min_val, max_val, power_sums)
        
        # T_0(x) = 1, so first Chebyshev moment should be 1
        assert abs(cheby_moments[0] - 1.0) < 1e-10
        
        # T_1(x) = x, this should be related to the mean, which is 2.0 after normalization to [-1, 1]
        # But with normalization, the mean in [-1, 1] should be 0
        assert abs(cheby_moments[1]) < 1e-10
        
    def test_calculate_mean(self):
        """Test calculating the mean."""
        values = [1, 2, 3, 4, 5]
        mean = Util.calculate_mean(values)
        assert mean == 3.0
        
    def test_calculate_powers(self):
        """Test calculating powers of a base value."""
        powers = Util.calculate_powers(2.0, 4)
        
        # Should be [1, 2, 4, 8]
        assert powers[0] == 1.0
        assert powers[1] == 2.0
        assert powers[2] == 4.0
        assert powers[3] == 8.0
        
    def test_calculate_entropy(self):
        """Test calculating entropy of a probability distribution."""
        # Uniform distribution has maximum entropy
        probs_uniform = [0.5, 0.5]
        entropy_uniform = Util.calculate_entropy(probs_uniform)
        assert entropy_uniform == pytest.approx(math.log(2))
        
        # Distribution with all probability on one outcome has minimum entropy
        probs_certain = [1.0, 0.0]
        entropy_certain = Util.calculate_entropy(probs_certain)
        assert entropy_certain == 0.0
        
    def test_get_mse(self):
        """Test computing the mean squared error."""
        errors = [1.0, 2.0, 3.0]
        mse = Util.get_mse(errors)
        expected = (1.0*1.0 + 2.0*2.0 + 3.0*3.0) / 3.0
        assert mse == expected


class TestMaxEntropy:
    """Test suite for the MaxEntropy class."""

    def setup_method(self):
        """Set up a test instance."""
        # Target moments for a uniform distribution on [-1, 1]
        # For uniform distribution on [-1, 1]:
        # M_0 = 1 (normalization)
        # M_1 = 0 (symmetric around 0)
        # M_2 = 1/3 (second moment)
        # M_3 = 0 (symmetric around 0)
        # M_4 = 1/5 (fourth moment)
        self.target_moments = [1.0, 0.0, 1.0/3.0, 0.0, 1.0/5.0]
        self.grid_size = 128
        
        self.max_entropy = MaxEntropy(self.target_moments, self.grid_size)
        
    def test_init(self):
        """Test initialization."""
        assert self.max_entropy.dim == len(self.target_moments)
        assert self.max_entropy.grid_size == self.grid_size
        assert np.array_equal(self.max_entropy.target_moments, self.target_moments)
        
        # Check grid points
        assert len(self.max_entropy.grid_points) == self.grid_size
        assert self.max_entropy.grid_points[0] == -1.0
        assert self.max_entropy.grid_points[-1] == 1.0
        
        # Check Chebyshev values
        assert self.max_entropy.chebyshev_values.shape == (2 * len(self.target_moments), self.grid_size)
        
    def test_set_lambda(self):
        """Test setting lambda values."""
        lambda_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.max_entropy.set_lambda(lambda_vals)
        
        assert np.array_equal(self.max_entropy.lambda_values, lambda_vals)
        
    def test_compute_all(self):
        """Test computing all values."""
        # Start with simple initialization
        lambda_vals = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Compute value, gradient, Hessian
        self.max_entropy.compute_all(lambda_vals)
        
        # Check weights
        assert len(self.max_entropy.weights) == self.grid_size
        assert all(w > 0 for w in self.max_entropy.weights)
        
        # Check moments
        assert len(self.max_entropy.moments) == 2 * len(self.target_moments)
        
        # Check gradient
        assert len(self.max_entropy.gradient) == len(self.target_moments)
        
        # Check Hessian
        assert self.max_entropy.hessian.shape == (len(self.target_moments), len(self.target_moments))
        
    def test_get_value(self):
        """Test getting the function value."""
        lambda_vals = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.max_entropy.compute_all(lambda_vals)
        
        value = self.max_entropy.get_value()
        assert isinstance(value, float)
        
    def test_get_gradient(self):
        """Test getting the gradient."""
        lambda_vals = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.max_entropy.compute_all(lambda_vals)
        
        gradient = self.max_entropy.get_gradient()
        assert len(gradient) == len(self.target_moments)
        
    def test_get_hessian(self):
        """Test getting the Hessian matrix."""
        lambda_vals = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.max_entropy.compute_all(lambda_vals)
        
        hessian = self.max_entropy.get_hessian()
        assert hessian.shape == (len(self.target_moments), len(self.target_moments))
        
        # Hessian should be symmetric
        assert np.allclose(hessian, hessian.T)
        
        # Hessian should be positive definite (for convex function)
        eigenvalues = np.linalg.eigvals(hessian)
        assert all(eigval > 0 for eigval in eigenvalues)


class TestQuadraticFunction:
    """Test suite for the QuadraticFunction class."""

    def test_init(self):
        """Test initialization."""
        func = QuadraticFunction(3)
        assert func.dimension == 3
        
    def test_compute_only_value(self):
        """Test computing only the function value."""
        func = QuadraticFunction(3)
        point = [1.0, 2.0, 3.0]
        
        func.compute_only_value(point, 1e-10)
        
        # Value should be sum of squares
        expected = 1.0*1.0 + 2.0*2.0 + 3.0*3.0
        assert func.function_value == expected
        
    def test_compute_all(self):
        """Test computing value, gradient, and Hessian."""
        func = QuadraticFunction(3)
        point = [1.0, 2.0, 3.0]
        
        func.compute_all(point, 1e-10)
        
        # Value should be sum of squares
        expected = 1.0*1.0 + 2.0*2.0 + 3.0*3.0
        assert func.function_value == expected
        
        # Gradient should be 2 * point
        assert np.array_equal(func.gradient, [2.0, 4.0, 6.0])
        
        # Hessian should be 2 * identity matrix
        expected_hessian = 2.0 * np.eye(3)
        assert np.array_equal(func.hessian, expected_hessian)
        
    def test_get_value(self):
        """Test getting the function value."""
        func = QuadraticFunction(3)
        point = [1.0, 2.0, 3.0]
        
        func.compute_all(point, 1e-10)
        
        assert func.get_value() == func.function_value
        
    def test_get_gradient(self):
        """Test getting the gradient."""
        func = QuadraticFunction(3)
        point = [1.0, 2.0, 3.0]
        
        func.compute_all(point, 1e-10)
        
        assert np.array_equal(func.get_gradient(), func.gradient)
        
    def test_get_hessian(self):
        """Test getting the Hessian matrix."""
        func = QuadraticFunction(3)
        point = [1.0, 2.0, 3.0]
        
        func.compute_all(point, 1e-10)
        
        assert np.array_equal(func.get_hessian(), func.hessian) 