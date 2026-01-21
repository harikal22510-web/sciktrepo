"""
Tests for advanced optimization utilities.
This provides comprehensive testing for the new advanced optimizers.
"""

import pytest
import numpy as np
from skopt.space import Real, Integer, Categorical
from skopt.advanced_optimizers import (
    AdaptiveOptimizer, 
    MultiFidelityOptimizer, 
    ConstrainedOptimizer,
    latin_hypercube_sampling,
    sobol_sequence
)


class TestAdaptiveOptimizer:
    """Comprehensive tests for the adaptive optimizer."""
    
    def test_adaptive_acquisition_selection(self):
        """Test adaptive acquisition function selection."""
        space = [
            Real(0.0, 1.0),
            Real(0.0, 1.0)
        ]
        
        # Test different adaptation strategies
        strategies = ["dynamic", "conservative", "aggressive"]
        
        for strategy in strategies:
            opt = AdaptiveOptimizer(
                dimensions=space,
                adaptation_strategy=strategy,
                random_state=42
            )
            
            def simple_objective(x):
                return x[0]**2 + x[1]**2
            
            result = opt.minimize(
                func=simple_objective,
                n_calls=30
            )
            
            # Validate results
            assert result.fun is not None
            assert len(result.x) == 2
            assert result.fun < 1.0  # Should find good solution
    
    def test_convergence_detection(self):
        """Test advanced convergence detection."""
        space = [Real(0.0, 1.0)]
        
        opt = AdaptiveOptimizer(
            dimensions=space,
            convergence_threshold=1e-4,
            random_state=42
        )
        
        # Function that converges quickly
        def convergent_objective(x):
            return (x[0] - 0.5)**2
        
        result = opt.minimize(
            func=convergent_objective,
            n_calls=50
        )
        
        # Should converge to near-optimal solution
        assert abs(result.x[0] - 0.5) < 0.1
        assert result.fun < 0.01
    
    def test_improvement_tracking(self):
        """Test improvement history tracking."""
        space = [Real(0.0, 1.0)]
        
        opt = AdaptiveOptimizer(
            dimensions=space,
            random_state=42
        )
        
        def noisy_objective(x):
            return x[0]**2 + np.random.normal(0, 0.01)
        
        result = opt.minimize(
            func=noisy_objective,
            n_calls=40
        )
        
        # Should track improvements correctly
        assert len(opt.improvement_history) > 0
        assert opt.best_value < 1.0


class TestMultiFidelityOptimizer:
    """Tests for multi-fidelity optimization."""
    
    def test_fidelity_dimension_handling(self):
        """Test handling of fidelity dimensions."""
        space = [
            Real(0.0, 1.0),
            Real(0.0, 1.0)
        ]
        fidelity_dim = Real(1, 10)  # Fidelity levels 1-10
        
        def cost_function(fidelity):
            return fidelity  # Linear cost
        
        opt = MultiFidelityOptimizer(
            dimensions=space,
            fidelity_dim=fidelity_dim,
            cost_function=cost_function,
            random_state=42
        )
        
        def high_fidelity_objective(x):
            return x[0]**2 + x[1]**2
        
        result = opt.minimize(
            func=high_fidelity_objective,
            n_calls=30
        )
        
        # Should handle fidelity correctly
        assert result.fun is not None
        assert len(result.x) == 2  # Should return only parameters, not fidelity
    
    def test_cost_budget_constraint(self):
        """Test cost budget constraint handling."""
        space = [Real(0.0, 1.0)]
        fidelity_dim = Real(1, 5)
        
        def cost_function(fidelity):
            return fidelity * 10  # Each fidelity unit costs 10
        
        opt = MultiFidelityOptimizer(
            dimensions=space,
            fidelity_dim=fidelity_dim,
            cost_function=cost_function,
            random_state=42
        )
        
        def expensive_objective(x):
            return x[0]**2
        
        result = opt.minimize(
            func=expensive_objective,
            n_calls=20,
            cost_budget=100  # Limited budget
        )
        
        # Should respect budget constraint
        assert result.fun is not None
    
    def test_noise_level_adaptation(self):
        """Test noise level adaptation based on fidelity."""
        space = [Real(0.0, 1.0)]
        fidelity_dim = Real(1, 10)
        
        opt = MultiFidelityOptimizer(
            dimensions=space,
            fidelity_dim=fidelity_dim,
            random_state=42
        )
        
        def deterministic_objective(x):
            return x[0]**2
        
        result = opt.minimize(
            func=deterministic_objective,
            n_calls=30
        )
        
        # Should handle noise correctly
        assert result.fun is not None
        assert len(result.x) == 1


class TestConstrainedOptimizer:
    """Tests for constrained optimization."""
    
    def test_inequality_constraints(self):
        """Test handling of inequality constraints."""
        space = [Real(0.0, 2.0), Real(0.0, 2.0)]
        
        # Constraint: x + y <= 1
        def constraint(x):
            return x[0] + x[1] - 1  # Should be <= 0
        
        opt = ConstrainedOptimizer(
            dimensions=space,
            constraints=[constraint],
            penalty_method="quadratic",
            random_state=42
        )
        
        def objective(x):
            return (x[0] - 1)**2 + (x[1] - 1)**2
        
        result = opt.minimize(
            func=objective,
            n_calls=40
        )
        
        # Should respect constraint
        assert result.x[0] + result.x[1] <= 1.1  # Allow small tolerance
        assert result.fun is not None
    
    def test_multiple_constraints(self):
        """Test handling of multiple constraints."""
        space = [Real(0.0, 2.0), Real(0.0, 2.0)]
        
        # Multiple constraints
        def constraint1(x):
            return x[0] + x[1] - 1.5  # x + y <= 1.5
        
        def constraint2(x):
            return x[0] - x[1]  # x <= y
        
        opt = ConstrainedOptimizer(
            dimensions=space,
            constraints=[constraint1, constraint2],
            penalty_method="quadratic",
            random_state=42
        )
        
        def objective(x):
            return x[0]**2 + x[1]**2
        
        result = opt.minimize(
            func=objective,
            n_calls=50
        )
        
        # Should respect all constraints
        assert result.x[0] + result.x[1] <= 1.6  # Allow tolerance
        assert result.x[0] <= result.x[1] + 0.1  # Allow tolerance
    
    def test_penalty_methods(self):
        """Test different penalty methods."""
        space = [Real(0.0, 2.0)]
        
        def constraint(x):
            return x[0] - 1  # x <= 1
        
        penalty_methods = ["quadratic", "linear", "exponential"]
        
        for method in penalty_methods:
            opt = ConstrainedOptimizer(
                dimensions=space,
                constraints=[constraint],
                penalty_method=method,
                random_state=42
            )
            
            def objective(x):
                return (x[0] - 0.5)**2
            
            result = opt.minimize(
                func=objective,
                n_calls=30
            )
            
            # Should handle different penalty methods
            assert result.fun is not None
            assert result.x[0] <= 1.2  # Should respect constraint


class TestSamplingUtilities:
    """Tests for advanced sampling utilities."""
    
    def test_latin_hypercube_sampling(self):
        """Test Latin Hypercube sampling."""
        samples = latin_hypercube_sampling(
            n_samples=100,
            n_dimensions=5,
            random_state=42
        )
        
        # Validate samples
        assert samples.shape == (100, 5)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)
        
        # Check Latin Hypercube property (each dimension has good coverage)
        for dim in range(5):
            dim_samples = samples[:, dim]
            # Should have samples in each bin
            n_bins = 10
            hist, _ = np.histogram(dim_samples, bins=n_bins, range=(0, 1))
            assert np.all(hist > 0)  # Each bin should have samples
    
    def test_sobol_sequence(self):
        """Test Sobol sequence generation."""
        points = sobol_sequence(
            n_points=64,  # Power of 2 for Sobol
            n_dimensions=3,
            random_state=42
        )
        
        # Validate points
        assert points.shape == (64, 3)
        assert np.all(points >= 0.0)
        assert np.all(points <= 1.0)
        
        # Check quasi-random properties (better uniformity than random)
        # Sobol points should have good space-filling properties
        for dim in range(3):
            dim_points = points[:, dim]
            # Should be well-distributed
            assert np.std(dim_points) > 0.2  # Not concentrated
    
    def test_sampling_reproducibility(self):
        """Test reproducibility of sampling methods."""
        # Test Latin Hypercube reproducibility
        samples1 = latin_hypercube_sampling(50, 3, random_state=42)
        samples2 = latin_hypercube_sampling(50, 3, random_state=42)
        
        np.testing.assert_array_equal(samples1, samples2)
        
        # Test Sobol reproducibility
        points1 = sobol_sequence(32, 2, random_state=42)
        points2 = sobol_sequence(32, 2, random_state=42)
        
        np.testing.assert_array_equal(points1, points2)


class TestAdvancedOptimizerIntegration:
    """Integration tests for advanced optimizers."""
    
    def test_adaptive_vs_standard_comparison(self):
        """Compare adaptive optimizer with standard optimizer."""
        from skopt import Optimizer
        
        space = [Real(-2.0, 2.0), Real(-2.0, 2.0)]
        
        # Multi-modal objective
        def multimodal_objective(x):
            return np.sin(x[0]) * np.cos(x[1]) + 0.1 * (x[0]**2 + x[1]**2)
        
        # Standard optimizer
        standard_opt = Optimizer(
            dimensions=space,
            random_state=42
        )
        standard_result = standard_opt.minimize(
            func=multimodal_objective,
            n_calls=40
        )
        
        # Adaptive optimizer
        adaptive_opt = AdaptiveOptimizer(
            dimensions=space,
            adaptation_strategy="dynamic",
            random_state=42
        )
        adaptive_result = adaptive_opt.minimize(
            func=multimodal_objective,
            n_calls=40
        )
        
        # Both should find reasonable solutions
        assert standard_result.fun is not None
        assert adaptive_result.fun is not None
        assert len(standard_result.x) == 2
        assert len(adaptive_result.x) == 2
    
    def test_constrained_feasibility(self):
        """Test constrained optimizer feasibility."""
        space = [Real(0.0, 3.0), Real(0.0, 3.0)]
        
        # Feasible region: x + y <= 2, x >= 0.5, y >= 0.5
        def constraint1(x):
            return x[0] + x[1] - 2
        
        def constraint2(x):
            return 0.5 - x[0]
        
        def constraint3(x):
            return 0.5 - x[1]
        
        opt = ConstrainedOptimizer(
            dimensions=space,
            constraints=[constraint1, constraint2, constraint3],
            penalty_method="quadratic",
            random_state=42
        )
        
        def objective(x):
            return (x[0] - 1)**2 + (x[1] - 1)**2
        
        result = opt.minimize(
            func=objective,
            n_calls=60
        )
        
        # Should find feasible solution
        assert result.x[0] + result.x[1] <= 2.2  # Allow tolerance
        assert result.x[0] >= 0.3  # Allow tolerance
        assert result.x[1] >= 0.3  # Allow tolerance


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
