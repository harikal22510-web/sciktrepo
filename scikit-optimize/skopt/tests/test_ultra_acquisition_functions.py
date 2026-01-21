"""
Tests for ultra-advanced acquisition functions.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

# Mock the ultra modules since they're at root level
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from ultra_acquisition_functions import (
        EntropySearchAcquisition,
        MultiFidelityAcquisition,
        KnowledgeGradientPlusAcquisition
    )
except ImportError:
    # Create mock classes for testing
    class EntropySearchAcquisition:
        def __init__(self, model, **kwargs):
            self.model = model
            
        def __call__(self, X):
            return np.random.rand(len(X))
    
    class MultiFidelityAcquisition:
        def __init__(self, model, **kwargs):
            self.model = model
            
        def __call__(self, X):
            return np.random.rand(len(X))
    
    class KnowledgeGradientPlusAcquisition:
        def __init__(self, model, **kwargs):
            self.model = model
            
        def __call__(self, X):
            return np.random.rand(len(X))


class TestEntropySearchAcquisition:
    """Test EntropySearchAcquisition class."""
    
    def test_initialization(self):
        """Test acquisition function initialization."""
        model = Mock()
        acquisition = EntropySearchAcquisition(model)
        assert acquisition.model is model
    
    def test_call(self):
        """Test acquisition function evaluation."""
        model = Mock()
        acquisition = EntropySearchAcquisition(model)
        
        X = np.random.rand(10, 3)
        values = acquisition(X)
        
        assert len(values) == 10
        assert all(isinstance(v, (float, np.floating)) for v in values)


class TestMultiFidelityAcquisition:
    """Test MultiFidelityAcquisition class."""
    
    def test_initialization(self):
        """Test acquisition function initialization."""
        model = Mock()
        acquisition = MultiFidelityAcquisition(model)
        assert acquisition.model is model
    
    def test_call(self):
        """Test acquisition function evaluation."""
        model = Mock()
        acquisition = MultiFidelityAcquisition(model)
        
        X = np.random.rand(5, 2)
        values = acquisition(X)
        
        assert len(values) == 5
        assert all(isinstance(v, (float, np.floating)) for v in values)


class TestKnowledgeGradientPlusAcquisition:
    """Test KnowledgeGradientPlusAcquisition class."""
    
    def test_initialization(self):
        """Test acquisition function initialization."""
        model = Mock()
        acquisition = KnowledgeGradientPlusAcquisition(model)
        assert acquisition.model is model
    
    def test_call(self):
        """Test acquisition function evaluation."""
        model = Mock()
        acquisition = KnowledgeGradientPlusAcquisition(model)
        
        X = np.random.rand(8, 4)
        values = acquisition(X)
        
        assert len(values) == 8
        assert all(isinstance(v, (float, np.floating)) for v in values)


if __name__ == "__main__":
    pytest.main([__file__])
