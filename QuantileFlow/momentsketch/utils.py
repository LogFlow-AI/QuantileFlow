import math
import numpy as np
from typing import List

class Util:
    @staticmethod
    def get_binomial_coefficients(max_degree: int) -> List[List[int]]:
        """
        Compute a binomial coefficient table up to max_degree

        Args:
            max_degree (int): max value for which to compute binomials

        Returns:
            List[List[int]]: 2D list of binomial coeffs
        """
        coefficients = [[0] * (max_degree + 1) for _ in range(max_degree + 1)]
        for i in range(max_degree + 1):
            coefficients[i][0] = 1
            for j in range(1, i + 1):
                coefficients[i][j] = coefficients[i - 1][j - 1] + coefficients[i - 1][j]
        return coefficients

    @staticmethod
    def get_cheby_coefficients(max_degree: int) -> List[List[int]]:
        """
        Compute Chebyshev polynomial coefficients up to degree max_degree

        Args:
            max_degree (int): max degree

        Returns:
            List[List[int]]: 2D list of coeffs
        """
        chebyshev_coeffs = [[0] * (max_degree + 1) for _ in range(max_degree + 1)]
        chebyshev_coeffs[0][0] = 1
        if max_degree == 0:
            return chebyshev_coeffs
        chebyshev_coeffs[1][1] = 1
        for i in range(2, max_degree + 1):
            prev_coeffs = chebyshev_coeffs[i - 1]
            prev_prev_coeffs = chebyshev_coeffs[i - 2]
            chebyshev_coeffs[i][0] = -prev_prev_coeffs[0]
            for j in range(1, i + 1):
                chebyshev_coeffs[i][j] = 2 * prev_coeffs[j - 1] - prev_prev_coeffs[j]
        return chebyshev_coeffs

    @staticmethod
    def shift_power_sums(power_sums: List[float], scaling_factor: float, center: float) -> List[float]:
        """
        Calculate shifted and scaled power sums for vals

        Args:
            power_sums (List[float]): Original powers
            scaling_factor (float): Scaling fact
            center (float): Translation val

        Returns:
            List[float]: Shifted and scaled powers
        """
        degree = len(power_sums) - 1
        scaled_power_sums = [0.0] * (degree + 1)
        # Compute (-center)**i and scaling_factor**(-i)
        negative_center_powers = [1.0] + [0.0] * degree
        inverse_scaling_powers = [1.0] + [0.0] * degree
        for i in range(1, degree + 1):
            negative_center_powers[i] = negative_center_powers[i - 1] * (-center)
            inverse_scaling_powers[i] = inverse_scaling_powers[i - 1] / scaling_factor

        binomial_coeffs = Util.get_binomial_coefficients(degree)
        for m in range(degree + 1):
            sum_val = 0.0
            for j in range(m + 1):
                sum_val += binomial_coeffs[m][j] * negative_center_powers[m - j] * power_sums[j]
            scaled_power_sums[m] = inverse_scaling_powers[m] * sum_val
        return scaled_power_sums

    @staticmethod
    def power_sums_to_normalized_moments(power_sums: List[float], min_val: float, max_val: float) -> List[float]:
        """
        Convert power sums to normalized moments

        Args:
            power_sums (List[float]): Original powers
            min_val (float): Min observed val
            max_val (float): Max observed val

        Returns:
            List[float]: Norm moments
        """
        radius = (max_val - min_val) / 2.0
        center = (max_val + min_val) / 2.0
        scaled_power_sums = Util.shift_power_sums(power_sums, radius, center)
        count = scaled_power_sums[0]
        # Normalize each moment by the count.
        for i in range(len(scaled_power_sums)):
            scaled_power_sums[i] /= count
        return scaled_power_sums

    @staticmethod
    def power_sums_to_cheby_moments(min_val: float, max_val: float, power_sums: List[float]) -> List[float]:
        """
        Convert power sums to Chebyshev moments.

        Args:
            min_val (float): Minimum observed value.
            max_val (float): Maximum observed value.
            power_sums (List[float]): Original power sums.

        Returns:
            List[float]: Chebyshev moments.
        """
        degree = len(power_sums) - 1
        radius = (max_val - min_val) / 2.0
        center = (max_val + min_val) / 2.0
        # Rescale the variables so they lie in [-1, 1]
        scaled_power_sums = Util.shift_power_sums(power_sums, radius, center)
        count = power_sums[0]
        chebyshev_coeffs = Util.get_cheby_coefficients(degree)
        chebyshev_moments = [0.0] * (degree + 1)
        for i in range(degree + 1):
            sum_val = 0.0
            for j in range(i + 1):
                sum_val += chebyshev_coeffs[i][j] * scaled_power_sums[j]
            chebyshev_moments[i] = sum_val / count
        return chebyshev_moments

    @staticmethod
    def calculate_mean(values: List[float]) -> float:
        """
        Calculate the mean of an array

        """
        return sum(values) / len(values)

    @staticmethod
    def calculate_powers(base: float, num_powers: int) -> List[float]:
        """
        Calculate the powers of base up to num_powers - 1

        Args:
            base (float): Base value.
            num_powers (int): Number of power terms to compute

        Returns:
            List[float]: A list of powers
        """
        powers = [0.0] * num_powers
        current_power = 1.0
        for i in range(num_powers):
            powers[i] = current_power
            current_power *= base
        return powers

    @staticmethod
    def calculate_entropy(probabilities: List[float]) -> float:
        """
        Compute the entropy of a probability distribution

        Args:
            probabilities (List[float]): A list of probabilities

        Returns:
            float: The entropy value
        """
        entropy_value = 0.0
        for prob in probabilities:
            if prob > 0.0:
                entropy_value -= prob * math.log(prob)
        return entropy_value

    @staticmethod
    def get_mse(errors: List[float]) -> float:
        """
        Compute the Mean Squared Error (MSE) of an error array

        Args:
            errors (List[float]): A list of error values

        Returns:
            float: The MSE value
        """
        sum_squared = sum(error * error for error in errors)
        return sum_squared / len(errors)


class MaxEntropy:
    """
    Maximum entropy loss function for optimization
    """

    def __init__(self, target_moments, grid_size):
        self.dim = len(target_moments)
        self.grid_size = grid_size
        self.target_moments = np.array(target_moments)

        # Create grid points
        self.grid_points = np.linspace(-1, 1, grid_size)

        # Pre-compute Chebyshev polynomial values
        self.chebyshev_values = np.zeros((2 * self.dim, grid_size))
        self.chebyshev_values[0] = 1.0
        self.chebyshev_values[1] = self.grid_points

        for j in range(2, 2 * self.dim):
            self.chebyshev_values[j] = 2 * self.grid_points * self.chebyshev_values[j - 1] - self.chebyshev_values[j - 2]

        # Initialize containers
        self.weights = np.zeros(grid_size)
        self.moments = np.zeros(2 * self.dim)
        self.gradient = np.zeros(self.dim)
        self.hessian = np.zeros((self.dim, self.dim))
        self.lambda_values = None

    def set_lambda(self, new_lambda):
        """Set lambda values"""
        self.lambda_values = new_lambda

    def compute_all(self, point, precision=1e-10):
        """Compute weights, moments, gradient, and hessian"""
        self.set_lambda(point)

        # Vectorized computation of weights - with numerical stability
        exponents = np.zeros(self.grid_size)
        for j in range(self.dim):
            exponents += self.lambda_values[j] * self.chebyshev_values[j]
            
        # Apply numerical stability measures to prevent overflows
        exponents = np.clip(exponents, -50, 50)  # Clip to reasonable range
        
        # Compute stable softmax-like weights
        max_exp = np.max(exponents)
        safe_exp = exponents - max_exp  # Subtract max for numerical stability
        self.weights = np.exp(safe_exp)
        
        # Avoid division by zero
        sum_weights = np.sum(self.weights)
        if sum_weights > 0:
            self.weights = self.weights / sum_weights
        else:
            # If weights sum to zero, use uniform distribution
            self.weights = np.ones(self.grid_size) / self.grid_size

        # Vectorized computation of moments
        for i in range(2 * self.dim):
            self.moments[i] = np.sum(self.chebyshev_values[i] * self.weights)

        # Compute gradient
        self.gradient = self.moments[:self.dim] - self.target_moments

        # Compute Hessian with stability check
        for i in range(self.dim):
            for j in range(self.dim):
                val = 0.5 * (self.moments[i + j] + self.moments[abs(i - j)])
                # Make sure hessian values are finite
                if np.isfinite(val):
                    self.hessian[i, j] = val
                else:
                    # Use a small positive value for stability
                    self.hessian[i, j] = 1e-8
        
        # Ensure Hessian is positive definite
        min_eig = np.min(np.linalg.eigvalsh(self.hessian))
        if min_eig <= 0:
            # Add a small regularization
            self.hessian += (abs(min_eig) + 1e-6) * np.eye(self.dim)

        return self.get_value()

    def get_value(self):
        """Get the function value"""
        return float(self.moments[0] - np.sum(self.lambda_values * self.target_moments))

    def get_gradient(self):
        """Get the gradient"""
        return self.gradient

    def get_hessian(self):
        """Get the Hessian matrix"""
        return self.hessian


class QuadraticFunction:
    """
    Simple quadratic function for testing optimization algorithms
    """

    def __init__(self, dimension):
        """
        Initialize with dimension

        Args:
            dimension: Dimension of the quadratic function
        """
        self.dimension = dimension
        self.function_value = 0
        self.gradient = np.zeros(dimension)
        self.hessian = np.zeros((dimension, dimension))

    def compute_only_value(self, point, precision):
        """Compute only the function value"""
        self.function_value = np.sum(np.square(point))

    def compute_all(self, point, precision):
        """Compute function value, gradient, and Hessian"""
        self.function_value = np.sum(np.square(point))
        
        # Correctly set the gradient to 2 * point
        self.gradient = 2 * np.array(point)
        self.hessian = 2 * np.eye(self.dimension)

        return self.function_value

    def dim(self):
        """Get function dimension"""
        return self.dimension

    def get_value(self):
        """Get function value"""
        return self.function_value

    def get_gradient(self):
        """Get function gradient"""
        return self.gradient

    def get_hessian(self):
        """Get function Hessian"""
        return self.hessian