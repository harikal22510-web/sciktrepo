"""
Comprehensive test suite for enterprise optimization framework.
This file provides extensive testing for all advanced optimization components.
"""

import pytest
import numpy as np
import time
import warnings
from unittest.mock import Mock, patch

# Import enterprise optimization suite
from enterprise_optimization_suite import (
    EnterpriseOptimizationSuite,
    quick_optimize,
    robust_optimize,
    multi_objective_optimize
)

# Import advanced components
from scikit_optimize.skopt.advanced_optimizer_module1 import (
    AdaptiveBayesianOptimizer,
    MultiObjectiveOptimizer,
    ConstrainedOptimizer
)
from scikit_optimize.skopt.advanced_models_module2 import (
    HeteroscedasticGaussianProcess,
    MultiFidelityGaussianProcess,
    DeepEnsembleRegressor,
    RobustGaussianProcess,
    AdaptiveRandomForest
)
from scikit_optimize.skopt.enhanced_acquisition_module3 import (
    ExpectedImprovementPlus,
    ProbabilityOfImprovementPlus,
    LowerConfidenceBoundPlus,
    KnowledgeGradient,
    ThompsonSampling,
    MaxValueEntropySearch
)
from scikit_optimize.skopt.advanced_transformations_module4 import (
    AdaptiveSpaceTransformer,
    HierarchicalSpaceTransformer,
    ConditionalSpaceTransformer,
    MultiObjectiveSpaceTransformer
)


class TestEnterpriseOptimizationSuite:
    """Comprehensive tests for the enterprise optimization suite."""
    
    def test_suite_initialization(self):
        """Test suite initialization with different parameters."""
        # Test default initialization
        suite = EnterpriseOptimizationSuite()
        assert suite.random_state is None
        assert len(suite._available_optimizers) == 3
        assert len(suite._available_models) == 5
        assert len(suite._available_acquisitions) == 6
        
        # Test initialization with random state
        suite_fixed = EnterpriseOptimizationSuite(random_state=42)
        assert suite_fixed.random_state == 42
    
    def test_create_optimizer(self):
        """Test optimizer creation with different types."""
        suite = EnterpriseOptimizationSuite(random_state=42)
        
        # Test adaptive Bayesian optimizer
        optimizer = suite.create_optimizer(
            optimizer_type='adaptive_bayesian',
            dimensions=[(0.0, 1.0), (0.0, 1.0)]
        )
        assert isinstance(optimizer, AdaptiveBayesianOptimizer)
        assert optimizer.random_state == 42
        
        # Test multi-objective optimizer
        multi_opt = suite.create_optimizer(
            optimizer_type='multi_objective',
            dimensions=[(0.0, 1.0), (0.0, 1.0)]
        )
        assert isinstance(multi_opt, MultiObjectiveOptimizer)
        
        # Test constrained optimizer
        constrained_opt = suite.create_optimizer(
            optimizer_type='constrained',
            dimensions=[(0.0, 1.0), (0.0, 1.0)]
        )
        assert isinstance(constrained_opt, ConstrainedOptimizer)
        
        # Test invalid optimizer type
        with pytest.raises(ValueError):
            suite.create_optimizer(optimizer_type='invalid')
    
    def test_create_model(self):
        """Test model creation with different types."""
        suite = EnterpriseOptimizationSuite(random_state=42)
        
        # Test heteroscedastic GP
        model = suite.create_model(model_type='heteroscedastic_gp')
        assert isinstance(model, HeteroscedasticGaussianProcess)
        assert model.random_state == 42
        
        # Test deep ensemble
        ensemble = suite.create_model(model_type='deep_ensemble', n_estimators=3)
        assert isinstance(ensemble, DeepEnsembleRegressor)
        assert ensemble.n_estimators == 3
        
        # Test invalid model type
        with pytest.raises(ValueError):
            suite.create_model(model_type='invalid')
    
    def test_create_acquisition(self):
        """Test acquisition function creation."""
        suite = EnterpriseOptimizationSuite(random_state=42)
        
        # Test Expected Improvement Plus
        ei = suite.create_acquisition(acquisition_type='ei_plus')
        assert isinstance(ei, ExpectedImprovementPlus)
        assert ei.random_state == 42
        
        # Test Thompson Sampling
        ts = suite.create_acquisition(acquisition_type='ts', n_samples=5)
        assert isinstance(ts, ThompsonSampling)
        assert ts.n_samples == 5
        
        # Test invalid acquisition type
        with pytest.raises(ValueError):
            suite.create_acquisition(acquisition_type='invalid')
    
    def test_optimize_integration(self):
        """Test full optimization workflow integration."""
        suite = EnterpriseOptimizationSuite(random_state=42)
        
        # Define test objective
        def test_objective(x):
            return x[0]**2 + x[1]**2
        
        dimensions = [(0.0, 1.0), (0.0, 1.0)]
        
        # Test optimization
        result = suite.optimize(
            objective=test_objective,
            dimensions=dimensions,
            n_calls=20
        )
        
        # Validate result structure
        assert 'x' in result
        assert 'fun' in result
        assert 'x_iters' in result
        assert 'func_vals' in result
        assert 'nit' in result
        assert 'success' in result
        
        # Validate result values
        assert len(result['x']) == 2
        assert result['fun'] < 1.0  # Should find good solution
        assert result['nit'] == 20
        assert result['success'] is True
    
    def test_benchmark_optimizers(self):
        """Test optimizer benchmarking functionality."""
        suite = EnterpriseOptimizationSuite(random_state=42)
        
        def test_objective(x):
            return x[0]**2 + x[1]**2
        
        dimensions = [(0.0, 1.0), (0.0, 1.0)]
        
        # Test benchmarking
        results = suite.benchmark_optimizers(
            objective=test_objective,
            dimensions=dimensions,
            n_calls=15
        )
        
        # Validate results
        assert len(results) == 3  # Should have results for all optimizers
        assert 'adaptive_bayesian' in results
        assert 'multi_objective' in results
        assert 'constrained' in results
        
        # Validate result structure
        for opt_type, result in results.items():
            assert 'best_value' in result
            assert 'n_iterations' in result
            assert 'success' in result
    
    def test_performance_analysis(self):
        """Test performance analysis functionality."""
        suite = EnterpriseOptimizationSuite(random_state=42)
        
        # Create mock results
        mock_results = {
            'adaptive_bayesian': {'best_value': 0.1, 'n_iterations': 20, 'success': True},
            'multi_objective': {'best_value': 0.2, 'n_iterations': 15, 'success': True},
            'constrained': {'best_value': 0.05, 'n_iterations': 25, 'success': True}
        }
        
        # Test analysis
        analysis = suite.analyze_performance(mock_results)
        
        # Validate analysis
        assert 'best_optimizer' in analysis
        assert 'best_value' in analysis
        assert analysis['best_optimizer'] == 'constrained'
        assert analysis['best_value'] == 0.05


class TestQuickOptimize:
    """Tests for quick optimization convenience function."""
    
    def test_quick_optimize_basic(self):
        """Test basic quick optimization."""
        def test_objective(x):
            return x[0]**2 + x[1]**2
        
        dimensions = [(0.0, 1.0), (0.0, 1.0)]
        
        result = quick_optimize(
            objective=test_objective,
            dimensions=dimensions,
            n_calls=20,
            random_state=42
        )
        
        assert result['fun'] < 1.0
        assert len(result['x']) == 2
        assert result['nit'] == 20
    
    def test_quick_optimize_with_noise(self):
        """Test quick optimization with noisy objective."""
        def noisy_objective(x):
            return x[0]**2 + x[1]**2 + np.random.normal(0, 0.01)
        
        dimensions = [(0.0, 1.0), (0.0, 1.0)]
        
        result = quick_optimize(
            objective=noisy_objective,
            dimensions=dimensions,
            n_calls=30,
            random_state=42
        )
        
        assert result['fun'] is not None
        assert len(result['x']) == 2


class TestRobustOptimize:
    """Tests for robust optimization convenience function."""
    
    def test_robust_optimize_basic(self):
        """Test basic robust optimization."""
        def test_objective(x):
            return x[0]**2 + x[1]**2
        
        dimensions = [(0.0, 1.0), (0.0, 1.0)]
        
        result = robust_optimize(
            objective=test_objective,
            dimensions=dimensions,
            n_calls=20,
            random_state=42
        )
        
        assert result['fun'] < 1.0
        assert len(result['x']) == 2
        assert result['nit'] == 20
    
    def test_robust_optimize_with_outliers(self):
        """Test robust optimization with outliers."""
        def outlier_objective(x):
            # Add occasional outliers
            if np.random.random() < 0.1:  # 10% chance of outlier
                return 100.0  # Large outlier
            return x[0]**2 + x[1]**2
        
        dimensions = [(0.0, 1.0), (0.0, 1.0)]
        
        result = robust_optimize(
            objective=outlier_objective,
            dimensions=dimensions,
            n_calls=30,
            random_state=42
        )
        
        # Should handle outliers gracefully
        assert result['fun'] < 10.0  # Should not be fooled by outliers
        assert len(result['x']) == 2


class TestMultiObjectiveOptimize:
    """Tests for multi-objective optimization convenience function."""
    
    def test_multi_objective_basic(self):
        """Test basic multi-objective optimization."""
        def obj1(x):
            return x[0]**2 + x[1]**2
        
        def obj2(x):
            return (x[0] - 1)**2 + (x[1] - 1)**2
        
        objectives = [obj1, obj2]
        dimensions = [(0.0, 1.0), (0.0, 1.0)]
        
        result = multi_objective_optimize(
            objectives=objectives,
            dimensions=dimensions,
            n_calls=30,
            random_state=42
        )
        
        assert 'pareto_front' in result
        assert len(result['x']) == 2
        assert result['nit'] == 30
    
    def test_multi_objective_conflicting(self):
        """Test multi-objective optimization with conflicting objectives."""
        def obj1(x):
            return x[0]  # Minimize x[0]
        
        def obj2(x):
            return -x[0]  # Maximize x[0] (conflicting)
        
        objectives = [obj1, obj2]
        dimensions = [(0.0, 1.0), (0.0, 1.0)]
        
        result = multi_objective_optimize(
            objectives=objectives,
            dimensions=dimensions,
            n_calls=20,
            random_state=42
        )
        
        assert 'pareto_front' in result
        assert len(result['pareto_front']) > 0


class TestAdvancedComponentsIntegration:
    """Integration tests for advanced components."""
    
    def test_adaptive_optimizer_with_advanced_model(self):
        """Test adaptive optimizer with advanced surrogate models."""
        dimensions = [(0.0, 1.0), (0.0, 1.0)]
        
        def test_objective(x):
            return x[0]**2 + x[1]**2 + np.random.normal(0, 0.01)
        
        # Test with heteroscedastic GP
        optimizer = AdaptiveBayesianOptimizer(
            dimensions=dimensions,
            base_estimator=HeteroscedasticGaussianProcess(random_state=42),
            random_state=42
        )
        
        result = optimizer.minimize(func=test_objective, n_calls=20)
        
        assert result['fun'] is not None
        assert len(result['x']) == 2
    
    def test_constrained_optimizer_with_constraints(self):
        """Test constrained optimizer with actual constraints."""
        dimensions = [(0.0, 2.0), (0.0, 2.0)]
        
        def test_objective(x):
            return x[0]**2 + x[1]**2
        
        def constraint1(x):
            return x[0] + x[1] - 1.5  # x + y <= 1.5
        
        def constraint2(x):
            return x[0] - x[1]  # x <= y
        
        optimizer = ConstrainedOptimizer(
            dimensions=dimensions,
            constraints=[constraint1, constraint2],
            penalty_method='quadratic',
            random_state=42
        )
        
        result = optimizer.minimize(func=test_objective, n_calls=30)
        
        assert result['fun'] is not None
        assert result['constraints_satisfied'] is True
    
    def test_acquisition_function_adaptivity(self):
        """Test acquisition function adaptivity."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        
        # Create test data
        X = np.random.random((20, 2))
        y = np.random.random(20)
        
        # Fit GP model
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        # Test adaptive EI
        ei = ExpectedImprovementPlus(adaptive=True, random_state=42)
        ei.improvement_history = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
        
        # Should adapt to low improvement
        acq_values = ei(X, model, y_opt=np.min(y))
        assert len(acq_values) == 20
        assert np.all(np.isfinite(acq_values))
    
    def test_space_transformation_integration(self):
        """Test space transformation integration."""
        # Create test data
        X = np.random.random((30, 5))
        
        # Test adaptive transformer
        transformer = AdaptiveSpaceTransformer(method='auto', random_state=42)
        transformer.fit(X)
        
        X_transformed = transformer.transform(X)
        X_recovered = transformer.inverse_transform(X_transformed)
        
        # Should preserve data structure
        assert X_transformed.shape[0] == X.shape[0]
        assert X_recovered.shape == X.shape
        np.testing.assert_array_almost_equal(X, X_recovered, rtol=1e-10)


class TestPerformanceAndScalability:
    """Performance and scalability tests."""
    
    def test_suite_performance(self):
        """Test suite performance with reasonable problem sizes."""
        suite = EnterpriseOptimizationSuite(random_state=42)
        
        def test_objective(x):
            return sum(xi**2 for xi in x)
        
        # Test with different problem sizes
        dimensions_list = [
            [(0.0, 1.0) for _ in range(2)],   # 2D
            [(0.0, 1.0) for _ in range(5)],   # 5D
            [(0.0, 1.0) for _ in range(10)]   # 10D
        ]
        
        for dimensions in dimensions_list:
            start_time = time.time()
            
            result = suite.optimize(
                objective=test_objective,
                dimensions=dimensions,
                n_calls=20
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete in reasonable time
            assert execution_time < 30.0, f"Too slow for {len(dimensions)}D problem"
            assert result['success'] is True
    
    def test_memory_usage(self):
        """Test memory usage with optimization."""
        import psutil
        import gc
        
        suite = EnterpriseOptimizationSuite(random_state=42)
        
        def memory_intensive_objective(x):
            # Create temporary data to stress memory
            temp_data = [np.random.random(100) for _ in range(5)]
            result = sum(xi**2 for xi in x) + sum(np.sum(d) for d in temp_data)
            del temp_data
            gc.collect()
            return result
        
        dimensions = [(0.0, 1.0) for _ in range(5)]
        
        # Monitor memory
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = suite.optimize(
            objective=memory_intensive_objective,
            dimensions=dimensions,
            n_calls=15
        )
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Should not use excessive memory
        assert memory_increase < 100, f"Memory usage too high: {memory_increase}MB"
        assert result['success'] is True


class TestErrorHandling:
    """Error handling and edge case tests."""
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        suite = EnterpriseOptimizationSuite()
        
        # Test invalid optimizer type
        with pytest.raises(ValueError):
            suite.create_optimizer(optimizer_type='invalid')
        
        # Test invalid model type
        with pytest.raises(ValueError):
            suite.create_model(model_type='invalid')
        
        # Test invalid acquisition type
        with pytest.raises(ValueError):
            suite.create_acquisition(acquisition_type='invalid')
    
    def test_optimization_failures(self):
        """Test handling of optimization failures."""
        suite = EnterpriseOptimizationSuite(random_state=42)
        
        # Define problematic objective
        def failing_objective(x):
            if x[0] > 0.5:
                raise ValueError("Invalid parameter")
            return x[0]**2 + x[1]**2
        
        dimensions = [(0.0, 1.0), (0.0, 1.0)]
        
        # Should handle failures gracefully
        with warnings.catch_warnings():
            result = suite.optimize(
                objective=failing_objective,
                dimensions=dimensions,
                n_calls=10
            )
        
        # Should still return a result structure
        assert 'success' in result
        assert 'error' not in result  # Should not crash
    
    def test_empty_dimensions(self):
        """Test handling of empty dimensions."""
        suite = EnterpriseOptimizationSuite()
        
        def test_objective(x):
            return sum(x**2 for x in x)
        
        # Empty dimensions should raise error
        with pytest.raises((ValueError, TypeError)):
            suite.optimize(
                objective=test_objective,
                dimensions=[],
                n_calls=10
            )


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
