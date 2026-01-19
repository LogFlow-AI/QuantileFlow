"""
Tests for the Optimizer module.

This module contains unit tests for the optimization algorithms used in MomentSketch:
- BaseOptimizer: Abstract base class defining the optimizer interface
- NewtonOptimizer: Implementation of damped Newton's method for convex optimization
"""
import numpy as np
from unittest.mock import MagicMock

from QuantileFlow.momentsketch.optimizer import BaseOptimizer, NewtonOptimizer
from QuantileFlow.momentsketch.utils import QuadraticFunction


class TestBaseOptimizer:
    """Test suite for the BaseOptimizer class."""
    
    def test_interface(self):
        """Test that the base optimizer defines the expected interface."""
        optimizer = BaseOptimizer()
        
        # All methods should be defined but do nothing
        optimizer.set_verbose(True)
        optimizer.set_max_iterations(100)
        assert optimizer.is_converged() is None
        assert optimizer.get_iteration_count() is None
        assert optimizer.get_function() is None
        assert optimizer.solve(None, None) is None


class TestNewtonOptimizer:
    """Test suite for the NewtonOptimizer class."""
    
    def setup_method(self):
        """Set up a test instance with a quadratic function."""
        self.dimension = 3
        self.quadratic = QuadraticFunction(self.dimension)
        self.optimizer = NewtonOptimizer(self.quadratic)
        
    def test_init(self):
        """Test initialization."""
        assert self.optimizer.objective_function == self.quadratic
        assert self.optimizer.max_iterations == 200
        assert not self.optimizer.verbose
        assert self.optimizer.iteration_count == 0
        assert self.optimizer.steps == 0
        assert not self.optimizer.converged
        
    def test_set_verbose(self):
        """Test setting verbose flag."""
        self.optimizer.set_verbose(True)
        assert self.optimizer.verbose
        
    def test_set_max_iterations(self):
        """Test setting max iterations."""
        self.optimizer.set_max_iterations(50)
        assert self.optimizer.max_iterations == 50
        
    def test_get_iteration_count(self):
        """Test getting iteration count."""
        assert self.optimizer.get_iteration_count() == 0
        
        # Simulate some iterations
        self.optimizer.iteration_count = 10
        assert self.optimizer.get_iteration_count() == 10
        
    def test_is_converged(self):
        """Test checking convergence status."""
        assert not self.optimizer.is_converged()
        
        # Simulate convergence
        self.optimizer.converged = True
        assert self.optimizer.is_converged()
        
    def test_get_backtracking_count(self):
        """Test getting backtracking step count."""
        assert self.optimizer.get_backtracking_count() == 0
        
        # Simulate some backtracking steps
        self.optimizer.steps = 5
        assert self.optimizer.get_backtracking_count() == 5
        
    def test_get_function(self):
        """Test getting the objective function."""
        assert self.optimizer.get_function() == self.quadratic
        
    def test_solve_quadratic(self):
        """Test solving a quadratic function (should converge to origin)."""
        # Start at a non-zero point
        initial_point = np.array([1.0, 2.0, 3.0])
        
        # Solve
        result = self.optimizer.solve(initial_point, 1e-6)
        
        # Should converge to origin
        assert np.allclose(result, np.zeros(self.dimension), atol=1e-5)
        
        # Should have converged
        assert self.optimizer.is_converged()
        
        # Iteration count should be positive but much less than max
        assert 0 < self.optimizer.get_iteration_count() < self.optimizer.max_iterations
        
    def test_solve_with_hessian_failure(self):
        """Test solving when Cholesky decomposition fails (SVD fallback)."""
        # Create a mock function that has a non-PD Hessian in first iteration
        mock_func = MagicMock()
        mock_func.dim = 2
        
        # First call, return degenerate Hessian
        first_call = True
        
        def compute_all_side_effect(point, precision):
            nonlocal first_call
            if first_call:
                # Degenerate Hessian (not positive definite)
                mock_func.get_value.return_value = 10.0
                mock_func.get_gradient.return_value = np.array([2.0, 2.0])
                mock_func.get_hessian.return_value = np.array([[1.0, 1.0], [1.0, 1.0]])
                first_call = False
            else:
                # Good Hessian
                mock_func.get_value.return_value = 1.0
                mock_func.get_gradient.return_value = np.array([0.1, 0.1])
                mock_func.get_hessian.return_value = np.array([[2.0, 0.0], [0.0, 2.0]])
            return mock_func.get_value()
            
        mock_func.compute_all.side_effect = compute_all_side_effect
        
        # Create optimizer with mock function
        optimizer = NewtonOptimizer(mock_func)
        
        # Solve
        initial_point = np.array([1.0, 1.0])
        result = optimizer.solve(initial_point, 1e-6)
        
        # Should complete without errors
        assert result is not None
        
    def test_solve_with_backtracking(self):
        """Test solving with backtracking line search."""
        # Create a mock function that requires backtracking
        mock_func = MagicMock()
        mock_func.dim = 2
        
        # Mock a function where full steps don't improve enough
        call_count = 0
        
        def compute_all_side_effect(point, precision):
            nonlocal call_count
            if call_count == 0:
                # Initial point
                mock_func.get_value.return_value = 10.0
                mock_func.get_gradient.return_value = np.array([2.0, 2.0])
                mock_func.get_hessian.return_value = np.array([[2.0, 0.0], [0.0, 2.0]])
            elif call_count == 1:
                # First step - doesn't improve enough
                mock_func.get_value.return_value = 9.5  # Not enough decrease
                mock_func.get_gradient.return_value = np.array([1.5, 1.5])
                mock_func.get_hessian.return_value = np.array([[2.0, 0.0], [0.0, 2.0]])
            elif call_count == 2:
                # Backtracking step - much better
                mock_func.get_value.return_value = 5.0
                mock_func.get_gradient.return_value = np.array([1.0, 1.0])
                mock_func.get_hessian.return_value = np.array([[2.0, 0.0], [0.0, 2.0]])
            else:
                # Remaining steps - converging
                mock_func.get_value.return_value = 0.1
                mock_func.get_gradient.return_value = np.array([0.01, 0.01])
                mock_func.get_hessian.return_value = np.array([[2.0, 0.0], [0.0, 2.0]])
            
            call_count += 1
            return mock_func.get_value()
            
        mock_func.compute_all.side_effect = compute_all_side_effect
        
        # Create optimizer with mock function
        optimizer = NewtonOptimizer(mock_func)
        
        # Solve
        initial_point = np.array([1.0, 1.0])
        optimizer.solve(initial_point, 1e-3)
        
        # Should have required at least one backtracking step
        assert optimizer.get_backtracking_count() > 0
        
    def test_solve_max_iterations(self):
        """Test that optimization stops at max iterations if not converged."""
        # Create a mock function that never converges
        mock_func = MagicMock()
        mock_func.dim = 2
        
        # Always report large gradient
        mock_func.get_value.return_value = 10.0
        mock_func.get_gradient.return_value = np.array([2.0, 2.0])
        mock_func.get_hessian.return_value = np.array([[0.1, 0.0], [0.0, 0.1]])  # Small Hessian to slow convergence
        
        # Create optimizer with low max iterations
        optimizer = NewtonOptimizer(mock_func)
        optimizer.set_max_iterations(5)
        
        # Solve
        initial_point = np.array([1.0, 1.0])
        optimizer.solve(initial_point, 1e-10)  # Very tight tolerance
        
        # Should not have converged
        assert not optimizer.is_converged()
        
        # Should have reached max iterations
        assert optimizer.get_iteration_count() == optimizer.max_iterations 