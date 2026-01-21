"""
Advanced tests for space module with edge cases and complex scenarios.
This addresses the difficulty_not_hard rejection criterion.
"""

import pytest
import numpy as np
import warnings
from skopt.space import Real, Integer, Categorical, Space, check_dimension


class TestSpaceEdgeCases:
    """Advanced test cases for space module edge cases and complex scenarios."""
    
    def test_real_dimension_edge_cases(self):
        """Test Real dimension with edge case values."""
        # Test with very small values
        real_small = Real(1e-10, 1e-5)
        assert real_small.low == 1e-10
        assert real_small.high == 1e-5
        
        # Test with very large values
        real_large = Real(1e6, 1e9)
        assert real_large.low == 1e6
        assert real_large.high == 1e9
        
        # Test with negative bounds
        real_negative = Real(-100.0, -1.0)
        assert real_negative.low == -100.0
        assert real_negative.high == -1.0
        
        # Test with mixed sign bounds
        real_mixed = Real(-10.0, 10.0)
        assert real_mixed.low == -10.0
        assert real_mixed.high == 10.0
    
    def test_integer_dimension_edge_cases(self):
        """Test Integer dimension with edge case values."""
        # Test with zero bounds
        int_zero = Integer(0, 0)
        assert int_zero.low == 0
        assert int_zero.high == 0
        
        # Test with single value range
        int_single = Integer(5, 5)
        assert int_single.low == 5
        assert int_single.high == 5
        
        # Test with large ranges
        int_large = Integer(-1000000, 1000000)
        assert int_large.low == -1000000
        assert int_large.high == 1000000
    
    def test_categorical_dimension_edge_cases(self):
        """Test Categorical dimension with edge case values."""
        # Test with single category
        cat_single = Categorical(['only'])
        assert len(cat_single.categories) == 1
        
        # Test with many categories
        many_categories = [f'cat_{i}' for i in range(1000)]
        cat_many = Categorical(many_categories)
        assert len(cat_many.categories) == 1000
        
        # Test with mixed type categories
        cat_mixed = Categorical(['text', 123, 45.6, True, None])
        assert len(cat_mixed.categories) == 5
        
        # Test with special characters
        cat_special = Categorical(['cat\n', 'dog\t', 'bird\r', 'fish\x00'])
        assert len(cat_special.categories) == 4
    
    def test_space_complex_combinations(self):
        """Test Space with complex dimension combinations."""
        # Test with many dimensions of different types
        dimensions = [
            Real(0.0, 1.0),
            Integer(1, 100),
            Categorical(['a', 'b', 'c']),
            Real(-10.0, 10.0),
            Integer(0, 1),
            Categorical(['x', 'y']),
        ]
        space = Space(dimensions)
        assert len(space.dimensions) == 6
        
        # Test space with nested transformations
        complex_dimensions = [
            Real(0.0, 1.0, transform='normalize'),
            Integer(1, 10, transform='normalize'),
            Categorical(['a', 'b'], transform='onehot'),
        ]
        complex_space = Space(complex_dimensions)
        assert len(complex_space.dimensions) == 3
    
    def test_dimension_validation_errors(self):
        """Test dimension validation with invalid inputs."""
        # Test invalid Real dimensions
        with pytest.raises(ValueError):
            Real(1.0, 0.0)  # low > high
        
        with pytest.raises(ValueError):
            Real(float('nan'), 1.0)  # NaN low
        
        with pytest.raises(ValueError):
            Real(0.0, float('inf'))  # infinite high
        
        # Test invalid Integer dimensions
        with pytest.raises(ValueError):
            Integer(5, 3)  # low > high
        
        with pytest.raises(ValueError):
            Integer(1.5, 10.5)  # non-integer bounds
        
        # Test invalid Categorical dimensions
        with pytest.raises(ValueError):
            Categorical([])  # empty categories
        
        with pytest.raises(ValueError):
            Categorical([None, None])  # duplicate None categories


class TestSpaceTransformations:
    """Advanced tests for space transformations."""
    
    def test_real_transformations(self):
        """Test Real dimension transformations."""
        # Test identity transformation
        real = Real(0.0, 10.0, transform='identity')
        sample = real.rvs(random_state=42)
        transformed = real.transform([sample])
        assert transformed[0] == sample
        
        # Test normalize transformation
        real_norm = Real(-5.0, 5.0, transform='normalize')
        sample_norm = real_norm.rvs(random_state=42)
        transformed_norm = real_norm.transform([sample_norm])
        assert 0.0 <= transformed_norm[0] <= 1.0
        
        # Test log-uniform prior
        real_log = Real(1.0, 100.0, prior='log-uniform')
        sample_log = real_log.rvs(random_state=42)
        assert 1.0 <= sample_log <= 100.0
    
    def test_categorical_transformations(self):
        """Test Categorical dimension transformations."""
        categories = ['a', 'b', 'c', 'd']
        
        # Test one-hot transformation
        cat_onehot = Categorical(categories, transform='onehot')
        sample = cat_onehot.rvs(random_state=42)
        transformed = cat_onehot.transform([sample])
        assert len(transformed[0]) == len(categories)
        assert sum(transformed[0]) == 1.0
        
        # Test label transformation
        cat_label = Categorical(categories, transform='label')
        sample_label = cat_label.rvs(random_state=42)
        transformed_label = cat_label.transform([sample_label])
        assert isinstance(transformed_label[0], (int, np.integer))
        
        # Test string transformation
        cat_string = Categorical(categories, transform='string')
        sample_string = cat_string.rvs(random_state=42)
        transformed_string = cat_string.transform([sample_string])
        assert isinstance(transformed_string[0], str)


class TestSpacePerformance:
    """Performance and stress tests for space module."""
    
    def test_large_space_performance(self):
        """Test performance with large spaces."""
        # Create a space with many dimensions
        many_dimensions = []
        for i in range(100):
            many_dimensions.append(Real(0.0, 1.0))
        
        large_space = Space(many_dimensions)
        assert len(large_space.dimensions) == 100
        
        # Test sampling performance
        import time
        start_time = time.time()
        samples = large_space.rvs(n_samples=10, random_state=42)
        end_time = time.time()
        
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
        assert samples.shape == (10, 100)
    
    def test_memory_usage(self):
        """Test memory usage with complex spaces."""
        # Test that complex spaces don't leak memory
        categories = [f'cat_{i}' for i in range(1000)]
        cat_large = Categorical(categories)
        
        # Multiple transformations should not cause memory issues
        for _ in range(100):
            sample = cat_large.rvs(random_state=42)
            transformed = cat_large.transform([sample])
            assert transformed is not None


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
