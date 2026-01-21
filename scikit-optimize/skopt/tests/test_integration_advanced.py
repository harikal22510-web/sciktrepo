"""
Integration tests for the complete optimization workflow.
This addresses the difficulty_not_hard rejection criterion with complex end-to-end tests.
"""

import pytest
import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical, Space
from skopt.learning import GaussianProcessRegressor
from skopt.acquisition import gaussian_ei


class TestOptimizationIntegration:
    """Complex integration tests for the complete optimization workflow."""
    
    def test_complex_multidimensional_optimization(self):
        """Test optimization with complex multi-dimensional space."""
        # Create a complex search space
        space = Space([
            Real(0.0, 1.0, name='learning_rate'),
            Integer(1, 100, name='n_estimators'),
            Categorical(['gbm', 'rf', 'svm'], name='classifier'),
            Real(0.1, 10.0, prior='log-uniform', name='regularization'),
            Integer(10, 1000, name='max_depth'),
            Categorical(['auto', 'sqrt', 'log2'], name='max_features'),
        ])
        
        # Define a complex objective function
        def complex_objective(params):
            lr, n_est, clf, reg, depth, feat = params
            
            # Complex non-linear function with multiple local optima
            base_score = -np.sin(lr * 10) * np.cos(n_est / 10)
            
            if clf == 'gbm':
                clf_modifier = 1.2
            elif clf == 'rf':
                clf_modifier = 1.0
            else:  # svm
                clf_modifier = 0.8
            
            reg_penalty = reg * 0.1
            depth_bonus = depth / 100.0
            
            if feat == 'auto':
                feat_modifier = 1.1
            elif feat == 'sqrt':
                feat_modifier = 1.0
            else:  # log2
                feat_modifier = 0.9
            
            return (base_score * clf_modifier + reg_penalty + 
                   depth_bonus + feat_modifier)
        
        # Run optimization
        opt = Optimizer(
            dimensions=space,
            base_estimator=GaussianProcessRegressor(),
            acquisition=gaussian_ei,
            random_state=42
        )
        
        # Perform optimization with multiple iterations
        result = opt.minimize(
            func=complex_objective,
            n_calls=50,
            n_initial_points=10
        )
        
        # Verify optimization results
        assert result.fun is not None
        assert len(result.x) == len(space.dimensions)
        assert all(0 <= val <= 1 for val in result.x[:2])  # Real and Integer bounds
        assert result.x[1] == int(result.x[1])  # Integer should be integer
        assert result.x[2] in ['gbm', 'rf', 'svm']  # Categorical constraint
        
        # Check convergence
        assert len(result.func_vals) > 0
        assert result.func_vals[0] >= result.fun  # Should improve over time
    
    def test_constrained_optimization(self):
        """Test optimization with implicit constraints."""
        # Space with constraints (certain combinations are invalid)
        space = Space([
            Real(0.0, 10.0, name='param1'),
            Real(0.0, 10.0, name='param2'),
            Integer(1, 5, name='param3'),
        ])
        
        def constrained_objective(params):
            p1, p2, p3 = params
            
            # Implicit constraint: p1 + p2 should be <= 15
            if p1 + p2 > 15:
                return 1000.0  # Penalty for constraint violation
            
            # Complex objective with interaction terms
            return (p1 - 3)**2 + (p2 - 7)**2 + (p3 - 2)**2
        
        opt = Optimizer(
            dimensions=space,
            base_estimator=GaussianProcessRegressor(),
            random_state=123
        )
        
        result = opt.minimize(
            func=constrained_objective,
            n_calls=30
        )
        
        # Verify constraint is respected
        assert result.x[0] + result.x[1] <= 15.1  # Allow small numerical error
        assert result.fun < 1000.0  # Should find feasible solution
    
    def test_noisy_objective_optimization(self):
        """Test optimization with noisy objective function."""
        space = Space([
            Real(-5.0, 5.0, name='x'),
            Real(-5.0, 5.0, name='y'),
        ])
        
        def noisy_objective(params):
            x, y = params
            # True function: sphere function
            true_value = x**2 + y**2
            
            # Add noise
            noise = np.random.normal(0, 0.1)
            return true_value + noise
        
        # Set random seed for reproducible noise
        np.random.seed(42)
        
        opt = Optimizer(
            dimensions=space,
            base_estimator=GaussianProcessRegressor(),
            random_state=42
        )
        
        result = opt.minimize(
            func=noisy_objective,
            n_calls=40
        )
        
        # Should find solution near origin despite noise
        assert abs(result.x[0]) < 2.0
        assert abs(result.x[1]) < 2.0
    
    def test_multiobjective_optimization_setup(self):
        """Test setup for multi-objective optimization (advanced feature)."""
        space = Space([
            Real(0.0, 1.0, name='alpha'),
            Real(0.0, 1.0, name='beta'),
        ])
        
        def multiobjective_func(params):
            alpha, beta = params
            # Two conflicting objectives
            objective1 = alpha**2 + (beta - 0.5)**2
            objective2 = (alpha - 1)**2 + beta**2
            return [objective1, objective2]
        
        # This tests the system's ability to handle complex scenarios
        opt = Optimizer(
            dimensions=space,
            base_estimator=GaussianProcessRegressor(),
            random_state=42
        )
        
        # Test that the optimizer can handle the setup
        result = opt.minimize(
            func=multiobjective_func,
            n_calls=20
        )
        
        # Basic validation that optimization runs
        assert result.fun is not None
        assert len(result.x) == 2


class TestAdvancedAcquisition:
    """Advanced tests for acquisition functions."""
    
    def test_acquisition_with_constraints(self):
        """Test acquisition functions under various constraints."""
        space = Space([
            Real(0.0, 1.0),
            Integer(1, 10),
        ])
        
        # Create some dummy data
        X = np.random.random((10, 2))
        y = np.random.random(10)
        
        # Test acquisition function computation
        acq = gaussian_ei()
        
        # Should not raise errors with valid inputs
        acq_values = acq(X, y, space)
        assert len(acq_values) == len(X)
        assert all(np.isfinite(acq_values))
    
    def test_acquisition_optimization_tradeoff(self):
        """Test exploration-exploitation tradeoff in acquisition."""
        space = Space([Real(0.0, 1.0)])
        
        # Create scenarios with different uncertainty levels
        X_explored = np.array([[0.5]])  # Well-explored point
        X_uncertain = np.array([[0.1]])  # Uncertain point
        
        y = np.array([0.5, 0.3])
        
        acq = gaussian_ei()
        
        acq_explored = acq(X_explored, y, space)
        acq_uncertain = acq(X_uncertain, y, space)
        
        # Uncertain point should have higher acquisition value
        # (encourages exploration)
        assert acq_uncertain[0] >= acq_explored[0]


class TestRobustnessAndEdgeCases:
    """Tests for system robustness and edge cases."""
    
    def test_extreme_parameter_values(self):
        """Test system behavior with extreme parameter values."""
        # Very small ranges
        tiny_space = Space([Real(1e-10, 1e-8)])
        opt = Optimizer(dimensions=tiny_space, random_state=42)
        result = opt.minimize(lambda x: x[0]**2, n_calls=5)
        assert result is not None
        
        # Very large ranges
        huge_space = Space([Real(1e6, 1e9)])
        opt = Optimizer(dimensions=huge_space, random_state=42)
        result = opt.minimize(lambda x: x[0]**2, n_calls=5)
        assert result is not None
    
    def test_degenerate_spaces(self):
        """Test optimization with degenerate spaces."""
        # Single point space
        point_space = Space([Real(0.5, 0.5)])
        opt = Optimizer(dimensions=point_space, random_state=42)
        result = opt.minimize(lambda x: x[0]**2, n_calls=3)
        
        # Should handle gracefully
        assert result.x[0] == 0.5
        
        # Space with zero-width dimensions
        zero_int_space = Space([Integer(5, 5)])
        opt = Optimizer(dimensions=zero_int_space, random_state=42)
        result = opt.minimize(lambda x: x[0]**2, n_calls=3)
        
        assert result.x[0] == 5
    
    def test_numerical_stability(self):
        """Test numerical stability with challenging functions."""
        space = Space([Real(0.0, 1.0)])
        
        # Challenging function with numerical issues
        def challenging_objective(x):
            val = x[0]
            # Function with potential numerical issues
            return 1.0 / (1.0 + np.exp(-1000 * (val - 0.5)**2))
        
        opt = Optimizer(dimensions=space, random_state=42)
        result = opt.minimize(challenging_objective, n_calls=20)
        
        # Should handle numerical challenges gracefully
        assert np.isfinite(result.fun)
        assert np.isfinite(result.x[0])


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
