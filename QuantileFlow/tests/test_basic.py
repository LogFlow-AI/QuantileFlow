import numpy as np

def test_addition():
    assert np.add(1, 1) == 2

def test_subtraction():
    assert np.subtract(2, 1) == 1

def test_multiplication():
    assert np.multiply(2, 3) == 6

def test_division():
    assert np.divide(6, 2) == 3.0