"""
Optimization algorithms for the MomentSketch implementation.

This module provides optimization algorithms used to solve the maximum entropy problem
in the MomentSketch implementation. It includes:

- BaseOptimizer: An abstract base class defining the common interface for optimizers
- NewtonOptimizer: An implementation of damped Newton's method for convex optimization

The optimizers handle numerical stability issues that can arise in the maximum entropy
optimization problem, including matrix conditioning problems and numerical precision issues.
"""

import numpy as np
from .utils import Util
from scipy.linalg import svd, solve


class BaseOptimizer:
    """
    Base class for optimization algorithms
    """

    def set_verbose(self, flag):
        """Set verbose output flag"""
        pass

    def set_max_iterations(self, max_iterations):
        """Set maximum iterations"""
        pass

    def is_converged(self):
        """Check if optimization has converged"""
        pass

    def get_iteration_count(self):
        """Get the number of steps taken"""
        pass

    def get_function(self):
        """Get the function being optimized"""
        pass

    def solve(self, initial_point, gradient_tolerance):
        """Solve the optimization problem"""
        pass


class NewtonOptimizer(BaseOptimizer):
    """
    Minimizes a convex function using damped Newton's
    """

    def __init__(self, objective_function):
        """
        Initialize the optimizer

        Args:
            objective_function: FunctionWithHessian to optimize
        """
        self.objective_function = objective_function
        self.max_iterations = 200
        self.iteration_count = 0
        self.steps = 0
        self.converged = False

        self.alpha = 0.3
        self.backtracking_rate = 0.25
        self.verbose = False

    def set_verbose(self, flag):
        """Set verbose output flag"""
        self.verbose = flag

    def set_max_iterations(self, max_iterations):
        """Set maximum iterations"""
        self.max_iterations = max_iterations

    def get_iteration_count(self):
        """Get the number of steps taken"""
        return self.iteration_count

    def is_converged(self):
        """Check if optimization has converged"""
        return self.converged

    def get_backtracking_count(self):
        """Get the number of damped steps"""
        return self.steps

    def get_function(self):
        """Get the function being optimized"""
        return self.objective_function

    def solve(self, initial_point, grad_tolerance):
        """
        Solve the optimization problem

        Args:
            initial_point: Initial point
            grad_tolerance: Grad tolerance for convergence

        Returns:
            Optimal point
        """
        dimension = self.objective_function.dim

        current_point = np.array(initial_point).copy()

        required_precision = grad_tolerance / 10
        self.objective_function.compute_all(current_point, required_precision)

        squared_tolerance = grad_tolerance * grad_tolerance
        self.converged = False

        for iteration in range(self.max_iterations):
            function_value = self.objective_function.get_value()
            gradient = self.objective_function.get_gradient()
            hessian = self.objective_function.get_hessian()
            
            # Check for NaN or Inf in gradient or hessian
            if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
                if self.verbose:
                    print("Warning: NaN or Inf detected in gradient or Hessian. Using fallback approach.")
                # Fallback to diagonal Hessian
                hessian = np.eye(dimension)
                # Clean gradient if needed
                gradient = np.nan_to_num(gradient, nan=0.0, posinf=1.0, neginf=-1.0)
            
            mean_squared_error = Util.get_mse(gradient)

            if self.verbose:
                print(f"Iteration: {iteration:3d} GradRMSE: {np.sqrt(mean_squared_error):10.5g} Value: {function_value:10.5g}")

            if mean_squared_error < squared_tolerance:
                self.converged = True
                break

            # Try to solve using Cholesky decomposition
            try:
                # Add small regularization to ensure positive definiteness if needed
                if np.any(np.diag(hessian) <= 0):
                    if self.verbose:
                        print("Adding regularization to Hessian")
                    hessian = hessian + 1e-8 * np.eye(dimension)
                    
                newton_direction = solve(hessian, gradient, assume_a='pos')
            except np.linalg.LinAlgError:
                # Fall back to SVD if Cholesky fails
                if self.verbose:
                    print("Cholesky decomposition failed, falling back to SVD")
                u, singular_values, vh = svd(hessian)
                pseudoinverse_values = np.array([1 / x if x > 1e-10 else 0 for x in singular_values])
                newton_direction = (vh.T * pseudoinverse_values) @ (u.T @ gradient)

            newton_direction = -newton_direction

            # Directional derivative
            directional_derivative = np.sum(newton_direction * gradient)

            step_size = 1.0
            candidate_point = current_point + step_size * newton_direction

            # Warning: this overwrites gradient and hessian
            try:
                self.objective_function.compute_all(candidate_point, required_precision)
            except Exception as e:
                if self.verbose:
                    print(f"Error computing objective at candidate point: {e}")
                step_size = 0.1
                candidate_point = current_point + step_size * newton_direction
                self.objective_function.compute_all(candidate_point, required_precision)

            # Do not look for damped steps if we are near stationary point
            if directional_derivative * directional_derivative > squared_tolerance:
                max_backtracking_steps = 10
                backtracking_steps = 0
                
                while True:
                    new_function_value = self.objective_function.get_value()
                    improvement = function_value + self.alpha * step_size * directional_derivative - new_function_value

                    if improvement >= -grad_tolerance or step_size < 1e-3 or backtracking_steps >= max_backtracking_steps:
                        break
                    else:
                        step_size *= self.backtracking_rate
                        backtracking_steps += 1

                    candidate_point = current_point + step_size * newton_direction
                    try:
                        self.objective_function.compute_all(candidate_point, required_precision)
                    except Exception:
                        # If computation fails, just use the current step size and break
                        break

            if step_size < 1.0:
                self.steps += 1

            if self.verbose and step_size < 1.0:
                print(f"Step Size: {step_size}")

            current_point = candidate_point

        self.iteration_count = iteration + 1
        return current_point