"""
Tests for the SimpleMomentSketch implementation.

This module contains unit tests for the core components of the SimpleMomentSketch implementation:
- Moment: Stores and manages power sums
- MomentSolver: Base solver for the maximum entropy problem
- NewtonMS: Newton's method-based solver for maximum entropy optimization
- SimpleMS: Main implementation class for the moment sketch algorithm
"""
import math
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from QuantileFlow.momentsketch.simple_moment_sketch import Moment, MomentSolver, NewtonMS, SimpleMS


class TestMoment:
    """Test suite for the Moment class."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        moment = Moment(k=5)
        assert len(moment.power_sums) == 5
        assert all(ps == 0 for ps in moment.power_sums)
        assert moment.min_val == float('inf')
        assert moment.max_val == float('-inf')
        
    def test_init_with_values(self):
        """Test initialization with provided values."""
        power_sums = [10, 20, 30, 40, 50]
        moment = Moment(power_sums=power_sums, min_val=1, max_val=10)
        
        assert np.array_equal(moment.power_sums, power_sums)
        assert moment.min_val == 1
        assert moment.max_val == 10
        
    def test_add_value(self):
        """Test adding a single value."""
        moment = Moment(k=3)
        moment.add_value(5.0)
        
        assert moment.min_val == 5.0
        assert moment.max_val == 5.0
        assert moment.power_sums[0] == 1.0  # count
        assert moment.power_sums[1] == 5.0  # sum
        assert moment.power_sums[2] == 25.0  # sum of squares
        
    def test_add_values(self):
        """Test adding multiple values."""
        moment = Moment(k=3)
        moment.add_values([1.0, 2.0, 3.0])
        
        assert moment.min_val == 1.0
        assert moment.max_val == 3.0
        assert moment.power_sums[0] == 3.0  # count
        assert moment.power_sums[1] == 6.0  # sum
        assert moment.power_sums[2] == 14.0  # sum of squares: 1^2 + 2^2 + 3^2
        
    def test_add_empty_values(self):
        """Test adding an empty list of values."""
        moment = Moment(k=3)
        moment.add_values([])
        
        assert moment.min_val == float('inf')
        assert moment.max_val == float('-inf')
        assert all(ps == 0 for ps in moment.power_sums)
        
    def test_merge(self):
        """Test merging two moments."""
        moment1 = Moment(k=3)
        moment1.add_values([1.0, 2.0])
        
        moment2 = Moment(k=3)
        moment2.add_values([3.0, 4.0])
        
        moment1.merge(moment2)
        
        assert moment1.min_val == 1.0
        assert moment1.max_val == 4.0
        assert moment1.power_sums[0] == 4.0  # count
        assert moment1.power_sums[1] == 10.0  # sum
        assert moment1.power_sums[2] == 30.0  # sum of squares: 1^2 + 2^2 + 3^2 + 4^2
        
    def test_str(self):
        """Test string representation."""
        moment = Moment(k=3)
        moment.add_values([1.0, 2.0, 3.0])
        
        str_repr = str(moment)
        assert "1" in str_repr
        assert "3" in str_repr
        assert str(moment.power_sums) in str_repr


class TestMomentSolver:
    """Test suite for the MomentSolver class."""
    
    def setup_method(self):
        """Set up a test moment and solver."""
        self.moment = Moment(k=5)
        self.moment.add_values([1.0, 2.0, 3.0, 4.0, 5.0])
        self.solver = MomentSolver(self.moment)
    
    def test_init(self):
        """Test initialization."""
        assert self.solver.x_min == 1.0
        assert self.solver.x_max == 5.0
        assert self.solver.x_center == 3.0
        assert self.solver.x_scale == 2.0
        assert self.solver.grid_size == 1024
        assert self.solver.max_iter == 15
        assert not self.solver.verbose
        
    def test_set_grid_size(self):
        """Test setting grid size."""
        self.solver.set_grid_size(2048)
        assert self.solver.grid_size == 2048
        
    def test_set_max_iter(self):
        """Test setting max iterations."""
        self.solver.set_max_iter(30)
        assert self.solver.max_iter == 30
        
    def test_set_verbose(self):
        """Test setting verbose flag."""
        self.solver.set_verbose(True)
        assert self.solver.verbose
        
    def test_solve(self):
        """Test solving for optimal parameters."""
        self.solver.solve()
        
        # After solving, weights and grid points should be defined
        assert self.solver.weights is not None
        assert self.solver.xs is not None
        assert len(self.solver.xs) == self.solver.grid_size
        assert len(self.solver.weights) == self.solver.grid_size
        
        # Grid points should be in the range [min, max]
        assert self.solver.xs[0] >= self.solver.x_min - 1e-10  # Allow for numerical issues
        assert self.solver.xs[-1] <= self.solver.x_max + 1e-10
        
        # Weights should be positive and sum to approximately 1
        assert all(w >= 0 for w in self.solver.weights)
        # Sum doesn't have to be exactly 1, just close
        weight_sum = sum(self.solver.weights)
        assert 0.9 <= weight_sum <= 1.1
        
    def test_get_quantiles_without_solve(self):
        """Test getting quantiles without solving first."""
        with pytest.raises(ValueError, match="You must call solve()"):
            self.solver.get_quantiles([0.5])
            
    def test_get_quantiles(self):
        """Test getting quantiles after solving."""
        self.solver.solve()
        
        quantiles = self.solver.get_quantiles([0.0, 0.25, 0.5, 0.75, 1.0])
        
        # Test that quantiles are in ascending order
        assert all(quantiles[i] <= quantiles[i+1] for i in range(len(quantiles)-1))
        
        # Test that quantiles are in the correct range
        assert all(self.solver.x_min <= q <= self.solver.x_max for q in quantiles)
        
    def test_get_quantile(self):
        """Test getting a single quantile."""
        self.solver.solve()
        
        median = self.solver.get_quantile(0.5)
        assert self.solver.x_min <= median <= self.solver.x_max
        
    def test_get_xs(self):
        """Test getting grid points."""
        self.solver.solve()
        xs = self.solver.get_xs()
        
        assert len(xs) == self.solver.grid_size
        assert xs[0] >= self.solver.x_min - 1e-10
        assert xs[-1] <= self.solver.x_max + 1e-10
        
    def test_get_weights(self):
        """Test getting weights."""
        self.solver.solve()
        weights = self.solver.get_weights()
        
        assert len(weights) == self.solver.grid_size
        assert all(w >= 0 for w in weights)


class TestNewtonMS:
    """Test suite for the NewtonMS class."""
    
    def setup_method(self):
        """Set up a test moment and solver."""
        self.moment = Moment(k=5)
        self.moment.add_values([1.0, 2.0, 3.0, 4.0, 5.0])
        self.solver = NewtonMS(self.moment)
    
    def test_init(self):
        """Test initialization."""
        assert self.solver.x_min == 1.0
        assert self.solver.x_max == 5.0
        assert self.solver.x_center == 3.0
        assert self.solver.x_scale == 2.0
        assert self.solver.grid_size == 1024
        assert self.solver.max_iter == 15
        assert not self.solver.verbose
        
    def test_set_grid_size(self):
        """Test setting grid size."""
        self.solver.set_grid_size(2048)
        assert self.solver.grid_size == 2048
        
    def test_set_max_iter(self):
        """Test setting max iterations."""
        self.solver.set_max_iter(30)
        assert self.solver.max_iter == 30
        
    def test_set_verbose(self):
        """Test setting verbose flag."""
        self.solver.set_verbose(True)
        assert self.solver.verbose
        
    def test_solve(self):
        """Test solving for optimal parameters."""
        self.solver.solve()
        
        # After solving, weights and grid points should be defined
        assert self.solver.weights is not None
        assert self.solver.xs is not None
        assert len(self.solver.xs) == self.solver.grid_size
        assert len(self.solver.weights) == self.solver.grid_size
        
        # Grid points should be in the range [min, max]
        assert self.solver.xs[0] >= self.solver.x_min - 1e-10
        assert self.solver.xs[-1] <= self.solver.x_max + 1e-10
        
        # Weights should be positive and sum to approximately 1
        assert all(w >= 0 for w in self.solver.weights)
        weight_sum = sum(self.solver.weights)
        assert 0.9 <= weight_sum <= 1.1
        
    def test_get_quantiles_without_solve(self):
        """Test getting quantiles without solving first."""
        with pytest.raises(ValueError, match="You must call solve()"):
            self.solver.get_quantiles([0.5])
            
    def test_get_quantiles(self):
        """Test getting quantiles after solving."""
        self.solver.solve()
        
        quantiles = self.solver.get_quantiles([0.0, 0.25, 0.5, 0.75, 1.0])
        
        # Test that quantiles are in ascending order
        assert all(quantiles[i] <= quantiles[i+1] for i in range(len(quantiles)-1))
        
        # Test that quantiles are in the correct range
        assert all(self.solver.x_min <= q <= self.solver.x_max for q in quantiles)
        
    def test_get_quantile(self):
        """Test getting a single quantile."""
        self.solver.solve()
        
        median = self.solver.get_quantile(0.5)
        assert self.solver.x_min <= median <= self.solver.x_max
        
    def test_get_xs(self):
        """Test getting grid points."""
        self.solver.solve()
        xs = self.solver.get_xs()
        
        assert len(xs) == self.solver.grid_size
        assert xs[0] >= self.solver.x_min - 1e-10
        assert xs[-1] <= self.solver.x_max + 1e-10
        
    def test_get_weights(self):
        """Test getting weights."""
        self.solver.solve()
        weights = self.solver.get_weights()
        
        assert len(weights) == self.solver.grid_size
        assert all(w >= 0 for w in weights)


class TestSimpleMS:
    """Test suite for the SimpleMS class."""
    
    def test_init_with_k(self):
        """Test initialization with k."""
        sketch = SimpleMS(5)
        assert sketch.get_k() == 5
        assert not sketch.get_compressed()  # Default is False
        
    def test_init_with_moment(self):
        """Test initialization with a Moment object."""
        moment = Moment(k=5)
        moment.add_values([1.0, 2.0, 3.0])
        
        sketch = SimpleMS(moment)
        assert sketch.get_k() == 5
        assert sketch.get_min() == 1.0
        assert sketch.get_max() == 3.0
        
    def test_init_invalid(self):
        """Test initialization with invalid parameter."""
        with pytest.raises(ValueError):
            SimpleMS("invalid")
            
    def test_set_compressed(self):
        """Test setting compression flag."""
        sketch = SimpleMS(5)
        assert not sketch.get_compressed()  # Default is False
        
        sketch.set_compressed(True)
        assert sketch.get_compressed()
        
        sketch.set_compressed(False)
        assert not sketch.get_compressed()
        
    def test_get_power_sums(self):
        """Test getting power sums."""
        sketch = SimpleMS(3)
        sketch.add_many([1.0, 2.0, 3.0])
        
        power_sums = sketch.get_power_sums()
        assert len(power_sums) == 3
        assert power_sums[0] == 3.0  # count
        
    def test_get_min_max(self):
        """Test getting min and max values."""
        sketch = SimpleMS(3)
        sketch.add_many([1.0, 2.0, 3.0])
        
        assert sketch.get_min() == 1.0
        assert sketch.get_max() == 3.0
        
    def test_add(self):
        """Test adding a single value."""
        sketch = SimpleMS(3)
        sketch.set_compressed(False)  # Disable compression for easier testing
        
        sketch.add(5.0)
        assert sketch.get_min() == 5.0
        assert sketch.get_max() == 5.0
        assert sketch.get_power_sums()[0] == 1.0
        
    def test_add_with_compression(self):
        """Test adding a value with compression."""
        sketch = SimpleMS(3)
        sketch.set_compressed(True)

        # Add a very large value
        sketch.add(1e6)

        # Min and max should be the raw values, not the transformed values
        assert sketch.get_min() == 1e6
        assert sketch.get_max() == 1e6

        # The power sums should reflect transformed values
        assert abs(sketch.get_power_sums()[1] - math.asinh(1e6)) < 1e-10

    def test_add_many(self):
        """Test adding multiple values."""
        sketch = SimpleMS(3)
        sketch.set_compressed(False)
        
        sketch.add_many([1.0, 2.0, 3.0])
        assert sketch.get_min() == 1.0
        assert sketch.get_max() == 3.0
        assert sketch.get_power_sums()[0] == 3.0
        
    def test_add_many_with_compression(self):
        """Test adding multiple values with compression."""
        sketch = SimpleMS(3)
        sketch.set_compressed(True)

        values = [1e-6, 1.0, 1e6]
        sketch.add_many(values)

        # Min and max should be the raw values
        assert sketch.get_min() == 1e-6
        assert sketch.get_max() == 1e6

        # The sum (power_sums[1]) should be the sum of the transformed values
        expected_sum = sum(math.asinh(v) for v in values)
        assert abs(sketch.get_power_sums()[1] - expected_sum) < 1e-10
        
    def test_merge(self):
        """Test merging two sketches."""
        sketch1 = SimpleMS(3)
        sketch1.set_compressed(False)
        sketch1.add_many([1.0, 2.0])
        
        sketch2 = SimpleMS(3)
        sketch2.set_compressed(False)
        sketch2.add_many([3.0, 4.0])
        
        sketch1.merge(sketch2)
        assert sketch1.get_min() == 1.0
        assert sketch1.get_max() == 4.0
        assert sketch1.get_power_sums()[0] == 4.0
        
    def test_get_solver(self):
        """Test getting a solver."""
        sketch = SimpleMS(5)
        sketch.add_many([1.0, 2.0, 3.0, 4.0, 5.0])
        
        solver = sketch.get_solver()
        assert isinstance(solver, NewtonMS)
        
    def test_get_quantiles(self):
        """Test getting quantiles."""
        sketch = SimpleMS(5)
        sketch.set_compressed(False)
        sketch.add_many([1.0, 2.0, 3.0, 4.0, 5.0])

        quantiles = sketch.get_quantiles([0.0, 0.25, 0.5, 0.75, 1.0])

        # Test that quantiles are in ascending order
        assert all(quantiles[i] <= quantiles[i+1] for i in range(len(quantiles)-1))

        # Test min and max values
        assert quantiles[0] >= 1.0  # min value
        assert quantiles[-1] <= 5.0  # max value

        # The middle quantile should be reasonably close to the middle value
        assert 2.0 <= quantiles[2] <= 4.0

    def test_get_quantile(self):
        """Test getting a single quantile."""
        sketch = SimpleMS(5)
        sketch.set_compressed(False)
        sketch.add_many([1.0, 2.0, 3.0, 4.0, 5.0])
        
        median = sketch.get_quantile(0.5)
        assert 2.5 <= median <= 3.5
        
    def test_get_median(self):
        """Test getting median."""
        sketch = SimpleMS(5)
        sketch.set_compressed(False)
        sketch.add_many([1.0, 2.0, 3.0, 4.0, 5.0])
        
        median = sketch.get_median()
        assert 2.5 <= median <= 3.5
        
    def test_get_percentile(self):
        """Test getting percentile."""
        sketch = SimpleMS(5)
        sketch.set_compressed(False)
        sketch.add_many([1.0, 2.0, 3.0, 4.0, 5.0])

        p75 = sketch.get_percentile(75)
        
        # 75th percentile should be between the 4th and 5th value for 5 samples
        assert p75 >= 1.0
        assert p75 <= 5.0
        assert p75 > sketch.get_percentile(50)  # 75th percentile should be larger than median

    def test_get_iqr(self):
        """Test getting interquartile range."""
        sketch = SimpleMS(5)
        sketch.set_compressed(False)
        sketch.add_many([1.0, 2.0, 3.0, 4.0, 5.0])

        iqr = sketch.get_iqr()
        
        # IQR should be positive for this data
        assert iqr > 0
        
        # For this small dataset, any reasonable estimate should be between
        # 1.0 and 4.0 (the full range is 4.0)
        assert iqr <= 4.0
        
    def test_get_stats(self):
        """Test getting statistics."""
        sketch = SimpleMS(5)
        sketch.set_compressed(False)
        sketch.add_many([1.0, 2.0, 3.0, 4.0, 5.0])
        
        stats = sketch.get_stats()
        
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert 'q1' in stats
        assert 'q3' in stats
        assert 'count' in stats
        assert 'mean' in stats
        
        assert stats['min'] >= 1.0
        assert stats['max'] <= 5.0
        assert 2.5 <= stats['median'] <= 3.5
        assert stats['count'] == 5.0
        assert 2.9 <= stats['mean'] <= 3.1  # Should be close to 3
        
    def test_to_dict(self):
        """Test conversion to dictionary."""
        sketch = SimpleMS(3)
        sketch.add_many([1.0, 2.0, 3.0])
        
        data_dict = sketch.to_dict()
        
        assert 'power_sums' in data_dict
        assert 'min_val' in data_dict
        assert 'max_val' in data_dict
        assert 'use_arc_sinh' in data_dict
        
        assert len(data_dict['power_sums']) == 3
        assert data_dict['min_val'] == sketch.get_min()
        assert data_dict['max_val'] == sketch.get_max()
        assert data_dict['use_arc_sinh'] == sketch.get_compressed()
        
    def test_from_dict(self):
        """Test creation from dictionary."""
        sketch = SimpleMS(3)
        sketch.set_compressed(False)
        sketch.add_many([1.0, 2.0, 3.0])
        
        data_dict = sketch.to_dict()
        restored = SimpleMS.from_dict(data_dict)
        
        assert restored.get_k() == sketch.get_k()
        assert restored.get_min() == sketch.get_min()
        assert restored.get_max() == sketch.get_max()
        assert restored.get_compressed() == sketch.get_compressed()
        assert np.array_equal(restored.get_power_sums(), sketch.get_power_sums())
        
    def test_str(self):
        """Test string representation."""
        sketch = SimpleMS(3)
        sketch.add_many([1.0, 2.0, 3.0])
        
        str_repr = str(sketch)
        assert str_repr  # Not empty
        
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.fill_between')
    def test_plot_dist(self, mock_fill, mock_plot, mock_figure):
        """Test plotting distribution."""
        # Setup the mocks
        mock_ax = MagicMock()
        mock_figure.return_value = MagicMock()
        mock_figure.return_value.gca.return_value = mock_ax
        
        # Create and use the sketch
        sketch = SimpleMS(5)
        sketch.add_many([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Call the method under test
        fig = sketch.plot_dist()
        
        # Assertions
        assert fig is not None
        mock_figure.assert_called_once()
        assert mock_ax.plot.call_count >= 1  # Should call plot on the axes object
        assert mock_ax.fill_between.call_count >= 1  # Should call fill_between on the axes object 