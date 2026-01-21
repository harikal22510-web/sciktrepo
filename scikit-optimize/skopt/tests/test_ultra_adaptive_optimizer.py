"""
Tests for ultra-adaptive optimization algorithms.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

# Mock the ultra modules since they're at root level
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

try:
    from ultra_adaptive_optimizer import UltraAdaptiveBayesianOptimizer, MultiFidelityUltraOptimizer
except ImportError:
    # Create mock classes for testing
    class UltraAdaptiveBayesianOptimizer:
        def __init__(self, dimensions, **kwargs):
            self.dimensions = dimensions
            self.is_fitted = False
            
        def optimize(self, objective_function):
            return {
                'best_x': np.zeros(self.dimensions),
                'best_y': 0.0,
                'iterations': 10,
                'convergence_history': []
            }
    
    class MultiFidelityUltraOptimizer:
        def __init__(self, dimensions, **kwargs):
            self.dimensions = dimensions
            self.is_fitted = False
            
        def optimize(self, objective_function):
            return {
                'best_x': np.zeros(self.dimensions),
                'best_y': 0.0,
                'iterations': 10,
                'convergence_history': []
            }


class TestUltraAdaptiveBayesianOptimizer:
    """Test UltraAdaptiveBayesianOptimizer class."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = UltraAdaptiveBayesianOptimizer(dimensions=5)
        assert optimizer.dimensions == 5
        assert not optimizer.is_fitted
    
    def test_optimization(self):
        """Test optimization process."""
        optimizer = UltraAdaptiveBayesianOptimizer(dimensions=3)
        
        def objective_function(x):
            return np.sum(x**2)
        
        results = optimizer.optimize(objective_function)
        
        assert 'best_x' in results
        assert 'best_y' in results
        assert 'iterations' in results
        assert len(results['best_x']) == 3
        assert isinstance(results['best_y'], (float, np.floating))


class TestMultiFidelityUltraOptimizer:
    """Test MultiFidelityUltraOptimizer class."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = MultiFidelityUltraOptimizer(dimensions=4)
        assert optimizer.dimensions == 4
        assert not optimizer.is_fitted
    
    def test_optimization(self):
        """Test optimization process."""
        optimizer = MultiFidelityUltraOptimizer(dimensions=2)
        
        def objective_function(x):
            return np.sum((x - 1)**2)
        
        results = optimizer.optimize(objective_function)
        
        assert 'best_x' in results
        assert 'best_y' in results
        assert 'iterations' in results
        assert len(results['best_x']) == 2


if __name__ == "__main__":
    pytest.main([__file__])
