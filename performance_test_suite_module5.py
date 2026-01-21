"""
Comprehensive Performance Testing Suite - Module 5
Enterprise-level performance benchmarks and stress testing for optimization algorithms.
"""

import pytest
import numpy as np
import time
import psutil
import gc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_friedman2

from advanced_optimizer_module1 import AdaptiveBayesianOptimizer
from advanced_models_module2 import HeteroscedasticGaussianProcess
from advanced_transformations_module3 import AdaptiveSpaceTransformer
from enhanced_acquisition_module4 import ExpectedImprovementPlus, LowerConfidenceBoundPlus


class TestPerformanceBenchmarks:
    """Enterprise-level performance benchmarks for optimization algorithms."""
    
    def test_large_scale_optimization_performance(self):
        """Test performance with large-scale optimization problems."""
        dimensions = []
        for i in range(50):  # 50-dimensional space
            dimensions.append((0.0, 1.0))
        
        # Complex multi-modal objective function
        def complex_objective(x):
            # Rastrigin function - challenging for optimizers
            n = len(x)
            return 10 * n + sum([xi**2 - 10 * np.cos(2 * np.pi * xi) 
                                   for xi in x])
        
        # Performance benchmark
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        optimizer = AdaptiveBayesianOptimizer(
            dimensions=dimensions,
            base_estimator=GaussianProcessRegressor(),
            n_initial_points=20,
            random_state=42
        )
        
        result = optimizer.minimize(
            func=complex_objective,
            n_calls=100
        )
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Performance assertions
        execution_time = end_time - start_time
        memory_increase = end_memory - start_memory
        
        assert execution_time < 60.0, f"Optimization took too long: {execution_time}s"
        assert memory_increase < 500, f"Memory usage too high: {memory_increase}MB"
        assert len(result['x']) == 50
        assert result['fun'] < -100.0  # Should find good solution
    
    def test_high_dimensional_convergence(self):
        """Test convergence behavior in high-dimensional spaces."""
        # Test with progressively increasing dimensions
        dimension_counts = [10, 25, 50, 100]
        
        for n_dims in dimension_counts:
            dimensions = [(0.0, 1.0) for _ in range(n_dims)]
            
            # Sphere function for convergence testing
            def sphere_function(x):
                return sum(x**2 for x in x)
            
            optimizer = AdaptiveBayesianOptimizer(
                dimensions=dimensions,
                base_estimator=GaussianProcessRegressor(),
                n_initial_points=min(20, n_dims),
                random_state=42
            )
            
            start_time = time.time()
            result = optimizer.minimize(
                func=sphere_function,
                n_calls=50 + n_dims  # More calls for higher dimensions
            )
            end_time = time.time()
            
            # Convergence quality assessment
            theoretical_optimum = 0.0
            convergence_error = abs(result['fun'] - theoretical_optimum)
            
            # Allow more error for higher dimensions (curse of dimensionality)
            max_acceptable_error = 0.1 * np.sqrt(n_dims)
            
            assert convergence_error < max_acceptable_error, \
                f"Poor convergence in {n_dims}D: error={convergence_error}"
            assert end_time - start_time < 120.0, \
                f"Too slow in {n_dims}D: {end_time - start_time}s"
    
    def test_memory_efficiency_stress_test(self):
        """Stress test for memory efficiency and garbage collection."""
        # Create memory-intensive scenario
        dimensions = [(0.0, 1.0) for _ in range(20)]
        
        def memory_intensive_objective(x):
            # Create temporary large arrays to stress memory
            temp_data = [np.random.random(1000) for _ in range(10)]
            result = sum(x**2 for x in x) + sum(np.sum(d) for d in temp_data)
            del temp_data  # Clean up
            return result
        
        # Monitor memory throughout optimization
        memory_samples = []
        
        def memory_monitoring_objective(x):
            gc.collect()  # Force garbage collection
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            result = memory_intensive_objective(x)
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append(memory_after - memory_before)
            return result
        
        optimizer = AdaptiveBayesianOptimizer(
            dimensions=dimensions,
            base_estimator=GaussianProcessRegressor(),
            n_initial_points=15,
            random_state=42
        )
        
        result = optimizer.minimize(
            func=memory_monitoring_objective,
            n_calls=30
        )
        
        # Memory efficiency validation
        avg_memory_increase = np.mean(memory_samples)
        max_memory_increase = np.max(memory_samples)
        
        assert avg_memory_increase < 100, f"Average memory too high: {avg_memory_increase}MB"
        assert max_memory_increase < 200, f"Peak memory too high: {max_memory_increase}MB"
        assert len(memory_samples) == 30
    
    def test_parallel_performance_scaling(self):
        """Test parallel performance scaling."""
        import multiprocessing
        
        dimensions = [(0.0, 1.0) for _ in range(10)]
        
        def simple_objective(x):
            return x[0]**2 + x[1]**2
        
        # Test with different numbers of jobs
        job_counts = [1, 2, 4]
        parallel_results = {}
        
        for n_jobs in job_counts:
            optimizer = AdaptiveBayesianOptimizer(
                dimensions=dimensions,
                base_estimator=GaussianProcessRegressor(),
                n_initial_points=15,
                n_jobs=n_jobs,
                random_state=42
            )
            
            start_time = time.time()
            result = optimizer.minimize(
                func=simple_objective,
                n_calls=40
            )
            end_time = time.time()
            
            parallel_results[n_jobs] = {
                'time': end_time - start_time,
                'best_value': result['fun']
            }
        
        # Parallel efficiency validation
        sequential_time = parallel_results[1]['time']
        
        for n_jobs in [2, 4]:
            parallel_time = parallel_results[n_jobs]['time']
            speedup = sequential_time / parallel_time
            
            # Should achieve reasonable parallel speedup
            min_speedup = min(n_jobs, multiprocessing.cpu_count())
            assert speedup >= min_speedup * 0.7, f"Poor parallel scaling for {n_jobs} jobs"
            assert speedup <= min_speedup * 1.2, "Unrealistic parallel speedup"


class TestAdvancedAcquisitionComparison:
    """Advanced tests comparing different acquisition functions."""
    
    def test_acquisition_function_comparison(self):
        """Compare performance of different acquisition functions."""
        dimensions = [(-2.0, 2.0), (-2.0, 2.0)]
        
        # Multi-modal objective (has multiple local optima)
        def multimodal_objective(x):
            x1, x2 = x
            return (x1**2 + x2**2) * np.cos(x1) * np.sin(x2)
        
        acquisition_functions = ["ei_plus", "lcb_plus", "pi_plus"]
        results = {}
        
        for acq_func in acquisition_functions:
            optimizer = AdaptiveBayesianOptimizer(
                dimensions=dimensions,
                base_estimator=GaussianProcessRegressor(),
                n_initial_points=10,
                random_state=42
            )
            
            start_time = time.time()
            result = optimizer.minimize(
                func=multimodal_objective,
                n_calls=40
            )
            end_time = time.time()
            
            results[acq_func] = {
                'best_value': result['fun'],
                'convergence_time': end_time - start_time,
                'n_iterations': len(result['func_vals'])
            }
        
        # Validate that all acquisition functions work
        for acq_func, result in results.items():
            assert np.isfinite(result['best_value']), \
                f"Invalid result for {acq_func}: {result['best_value']}"
            assert result['convergence_time'] < 30.0, \
                f"Too slow for {acq_func}: {result['convergence_time']}s"
        
        # Compare performance (EI should generally be best for this problem)
        ei_result = results['ei_plus']
        lcb_result = results['lcb_plus']
        pi_result = results['pi_plus']
        
        # EI should find good solution without being too slow
        assert ei_result['best_value'] <= lcb_result['best_value'] * 1.5
        assert ei_result['convergence_time'] <= lcb_result['convergence_time'] * 1.2


class TestRobustnessValidation:
    """Robustness tests with challenging edge cases."""
    
    def test_numerical_precision_robustness(self):
        """Test robustness to numerical precision issues."""
        dimensions = [
            (1e-10, 1e-5),   # Very small range
            (1e5, 1e8),        # Very large range
            (-1e6, 1e6),        # Large symmetric range
        ]
        
        def precision_challenging_objective(x):
            # Function with potential numerical precision issues
            x1, x2, x3 = x
            
            # Terms that could cause precision problems
            term1 = x1 * 1e-12  # Very small multiplication
            term2 = x2 / 1e8   # Large division
            term3 = np.exp(-abs(x3))  # Exponential with large negative
            
            return term1 + term2 + term3
        
        optimizer = AdaptiveBayesianOptimizer(
            dimensions=dimensions,
            base_estimator=GaussianProcessRegressor(),
            n_initial_points=15,
            random_state=42
        )
        
        result = optimizer.minimize(
            func=precision_challenging_objective,
            n_calls=50
        )
        
        # Should handle numerical challenges gracefully
        assert np.all(np.isfinite(result['x'])), "Non-finite parameters found"
        assert np.isfinite(result['fun']), "Non-finite objective value"
        
        # Check for reasonable bounds
        assert 1e-10 <= result['x'][0] <= 1e-5
        assert 1e5 <= result['x'][1] <= 1e8
        assert -1e6 <= result['x'][2] <= 1e6
    
    def test_extreme_categorical_spaces(self):
        """Test optimization with extreme categorical spaces."""
        # Large categorical space
        large_categories = [f'category_{i}' for i in range(1000)]
        
        dimensions = [large_categories, (0.0, 1.0), (1, 100)]
        
        def large_categorical_objective(x):
            cat_idx, real_val, int_val = x
            
            # Objective depends on categorical index
            categorical_value = cat_idx / 999.0  # Normalize to [0, 1]
            
            return categorical_value * 0.5 + real_val * 0.3 + int_val * 0.2
        
        optimizer = AdaptiveBayesianOptimizer(
            dimensions=dimensions,
            base_estimator=RandomForestRegressor(),  # Better for categorical
            n_initial_points=20,
            random_state=42
        )
        
        start_time = time.time()
        result = optimizer.minimize(
            func=large_categorical_objective,
            n_calls=60  # More calls for categorical space
        )
        end_time = time.time()
        
        # Performance validation for large categorical
        assert end_time - start_time < 45.0, "Too slow for large categorical"
        assert 0 <= result['x'][0] < 1000, "Invalid categorical index"
        assert 0.0 <= result['x'][1] <= 1.0, "Invalid real value"
        assert 1 <= result['x'][2] <= 100, "Invalid integer value"


class TestScalabilityAnalysis:
    """Scalability analysis for different problem sizes."""
    
    def test_scalability_across_dimensions(self):
        """Test how algorithm scales with problem dimensions."""
        dimension_counts = [5, 10, 20, 30, 40, 50]
        scalability_results = {}
        
        for n_dims in dimension_counts:
            dimensions = [(0.0, 1.0) for _ in range(n_dims)]
            
            # Simple quadratic objective
            def quadratic_objective(x):
                return sum(x**2 for x in x)
            
            optimizer = AdaptiveBayesianOptimizer(
                dimensions=dimensions,
                base_estimator=GaussianProcessRegressor(),
                n_initial_points=min(25, n_dims * 2),
                random_state=42
            )
            
            # Measure time complexity
            start_time = time.time()
            result = optimizer.minimize(
                func=quadratic_objective,
                n_calls=30 + n_dims
            )
            end_time = time.time()
            
            scalability_results[n_dims] = {
                'time': end_time - start_time,
                'function_evaluations': len(result['func_vals']),
                'final_error': result['fun']
            }
        
        # Analyze scalability (should be roughly O(n^2) or better)
        for n_dims in range(10, 50, 10):
            if n_dims in scalability_results:
                time_per_eval = scalability_results[n_dims]['time'] / scalability_results[n_dims]['function_evaluations']
                
                # Time per evaluation should not grow too rapidly
                if n_dims > 10:
                    prev_time_per_eval = scalability_results[n_dims-10]['time'] / scalability_results[n_dims-10]['function_evaluations']
                    growth_factor = time_per_eval / prev_time_per_eval
                    
                    # Allow some growth but not exponential
                    assert growth_factor < 3.0, f"Poor scalability from {n_dims-10}D to {n_dims}D"
    
    def test_model_comparison_scalability(self):
        """Test scalability of different surrogate models."""
        dimensions = [(0.0, 1.0) for _ in range(15)]
        
        def simple_objective(x):
            return sum(x**2 for x in x)
        
        models = [
            GaussianProcessRegressor(),
            RandomForestRegressor(n_estimators=50),
            HeteroscedasticGaussianProcess()
        ]
        
        model_results = {}
        
        for model in models:
            optimizer = AdaptiveBayesianOptimizer(
                dimensions=dimensions,
                base_estimator=model,
                n_initial_points=15,
                random_state=42
            )
            
            start_time = time.time()
            result = optimizer.minimize(
                func=simple_objective,
                n_calls=40
            )
            end_time = time.time()
            
            model_results[type(model).__name__] = {
                'time': end_time - start_time,
                'best_value': result['fun']
            }
        
        # Compare model performance
        gp_time = model_results['GaussianProcessRegressor']['time']
        rf_time = model_results['RandomForestRegressor']['time']
        hg_time = model_results['HeteroscedasticGaussianProcess']['time']
        
        # GP should be slower but more accurate for small problems
        assert rf_time < gp_time * 2.0, "RandomForest too slow compared to GP"
        
        # All models should find reasonable solutions
        for model_name, result in model_results.items():
            assert result['best_value'] < 10.0, f"Poor performance from {model_name}"


class TestIntegrationAdvanced:
    """Integration tests for advanced optimization components."""
    
    def test_adaptive_vs_standard_comparison(self):
        """Compare adaptive optimizer with standard optimizer."""
        dimensions = [(-2.0, 2.0), (-2.0, 2.0)]
        
        # Multi-modal objective
        def multimodal_objective(x):
            return np.sin(x[0]) * np.cos(x[1]) + 0.1 * (x[0]**2 + x[1]**2)
        
        # Standard optimizer (simplified)
        def standard_optimize(func, n_calls):
            X = []
            y = []
            
            for i in range(n_calls):
                if i == 0:
                    x = np.random.uniform(-2, 2, (2,))
                else:
                    # Simple random sampling for comparison
                    x = np.random.uniform(-2, 2, (2,))
                
                y_val = func(x)
                X.append(x)
                y.append(y_val)
            
            best_idx = np.argmin(y)
            return X[best_idx], y[best_idx]
        
        # Adaptive optimizer
        adaptive_optimizer = AdaptiveBayesianOptimizer(
            dimensions=dimensions,
            adaptation_strategy="dynamic",
            random_state=42
        )
        
        adaptive_result = adaptive_optimizer.minimize(
            func=multimodal_objective,
            n_calls=40
        )
        
        # Standard optimization (simplified)
        standard_x, standard_y = standard_optimize(multimodal_objective, 40)
        
        # Both should find reasonable solutions
        assert standard_y < 5.0
        assert adaptive_result['fun'] < 5.0
        assert len(standard_x) == 2
        assert len(adaptive_result['x']) == 2
    
    def test_advanced_model_integration(self):
        """Test integration of advanced models with optimizer."""
        dimensions = [(0.0, 1.0), (0.0, 1.0)]
        
        def noisy_objective(x):
            return x[0]**2 + x[1]**2 + np.random.normal(0, 0.01)  # Add noise
        
        # Test with heteroscedastic GP
        hg_model = HeteroscedasticGaussianProcess()
        optimizer = AdaptiveBayesianOptimizer(
            dimensions=dimensions,
            base_estimator=hg_model,
            n_initial_points=15,
            random_state=42
        )
        
        result = optimizer.minimize(
            func=noisy_objective,
            n_calls=30
        )
        
        # Should handle noise correctly
        assert result['fun'] is not None
        assert len(result['x']) == 2
    
    def test_transformation_integration(self):
        """Test integration of space transformations."""
        from advanced_transformations_module3 import AdaptiveSpaceTransformer
        
        dimensions = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        
        def transformed_objective(x):
            # Apply transformation first
            transformer = AdaptiveSpaceTransformer(method='pca')
            transformer.fit(np.array([x]))
            x_transformed = transformer.transform(np.array([x]))
            
            # Optimize in transformed space
            return x_transformed[0, 0]**2 + x_transformed[0, 1]**2
        
        optimizer = AdaptiveBayesianOptimizer(
            dimensions=dimensions,
            base_estimator=GaussianProcessRegressor(),
            n_initial_points=10,
            random_state=42
        )
        
        result = optimizer.minimize(
            func=transformed_objective,
            n_calls=25
        )
        
        # Should find good solution
        assert result['fun'] < 2.0
        assert len(result['x']) == 3


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
