"""
Tests for space module imports to ensure explicit imports work correctly.
This test addresses the unused import bug fix.
"""

import pytest
import sys
import importlib


def test_space_explicit_imports():
    """Test that all explicit imports from skopt.space work correctly."""
    # Test explicit imports that replaced wildcard import
    from skopt.space import Real, Integer, Categorical, Space, check_dimension
    
    # Test that each import is available and callable/constructible
    assert callable(Real)
    assert callable(Integer) 
    assert callable(Categorical)
    assert callable(Space)
    assert callable(check_dimension)
    
    # Test basic functionality
    real_dim = Real(0.0, 1.0)
    assert real_dim.low == 0.0
    assert real_dim.high == 1.0
    
    int_dim = Integer(1, 10)
    assert int_dim.low == 1
    assert int_dim.high == 10
    
    cat_dim = Categorical(['a', 'b', 'c'])
    assert cat_dim.categories == ['a', 'b', 'c']
    
    space = Space([real_dim, int_dim])
    assert len(space.dimensions) == 2
    
    # Test check_dimension function
    assert check_dimension(real_dim) is None


def test_space_no_private_imports():
    """Test that private functions are not accidentally imported."""
    # Import the module
    import skopt.space
    
    # Check that private attributes (starting with _) are not in __all__ or directly accessible
    public_attrs = [attr for attr in dir(skopt.space) if not attr.startswith('_')]
    
    # Should only contain the explicitly imported items
    expected_public = ['Real', 'Integer', 'Categorical', 'Space', 'check_dimension']
    
    # Verify no unexpected public attributes
    for attr in public_attrs:
        if attr in expected_public:
            continue
        # Allow some expected module-level attributes
        if attr in ['np', 'warnings', 'normalize', 'transform']:
            continue
        pytest.fail(f"Unexpected public attribute found: {attr}")


def test_space_import_completeness():
    """Test that all necessary public API components are available."""
    from skopt.space import Real, Integer, Categorical, Space, check_dimension
    
    # Test that we can create the main space components
    dimensions = [
        Real(0.0, 1.0),
        Integer(1, 10),
        Categorical(['option1', 'option2'])
    ]
    
    space = Space(dimensions)
    assert space is not None
    assert len(space.dimensions) == 3
    
    # Test that check_dimension works with all types
    for dim in dimensions:
        assert check_dimension(dim) is None


if __name__ == "__main__":
    pytest.main([__file__])
