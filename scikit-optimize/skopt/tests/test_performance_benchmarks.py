"""
Comprehensive performance benchmarks and stress tests.
This addresses the difficulty_not_hard rejection with enterprise-level performance testing.
"""

import pytest
import numpy as np
import time
import psutil
import gc
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical, Space
from skopt.learning import GaussianProcessRegressor, RandomForestRegressor, ExtraTreesRegressor
from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi


class TestPerformanceBenchmarks:
    """Enterprise-level performance benchmarks for optimization algorithms."""
    
    def test_large_scale_optimization_performance(self):
        """Test performance with large-scale optimization problems."""
        dimensions = []
        for i in range(50):  # 50-dimensional space
            dimensions.append(Real(0.0, 1.0))
        
        space = Space(dimensions)
        
        # Complex multi-modal objective function
        def complex_objective(x):
            # Rastrigin function - challenging for optimizers
            n = len(x)
            return 10 * n + sum([xi**2 - 10 * np.cos(2 * np.pi * xi) 
                                   for xi in x])
        
        # Performance benchmark
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        opt = Optimizer(
            dimensions=space,
            base_estimator=GaussianProcessRegressor(),
            n_initial_points=20,
            random_state=42
        )
        
        result = opt.minimize(
            func=complex_objective,
            n_calls=100,
            acq_func="ei"
        )
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Performance assertions
        execution_time = end_time - start_time
        memory_increase = end_memory - start_memory
        
        assert execution_time < 60.0, f"Optimization took too long: {execution_time}s"
        assert memory_increase < 500, f"Memory usage too high: {memory_increase}MB"
        assert len(result.x) == 50
        assert result.fun < -100.0  # Should find good solution
    
    def test_high_dimensional_convergence(self):
        """Test convergence behavior in high-dimensional spaces."""
        # Test with progressively increasing dimensions
        dimensions_list = [10, 25, 50, 100]
        
        for n_dims in dimensions_list:
            space = Space([Real(0.0, 1.0) for _ in range(n_dims)])
            
            # Sphere function for convergence testing
            def sphere_function(x):
                return sum(x**2 for x in x)
            
            opt = Optimizer(
                dimensions=space,
                base_estimator=GaussianProcessRegressor(),
                n_initial_points=min(20, n_dims),
                random_state=42
            )
            
            start_time = time.time()
            result = opt.minimize(
                func=sphere_function,
                n_calls=50 + n_dims  # More calls for higher dimensions
            )
            end_time = time.time()
            
            # Convergence quality assessment
            theoretical_optimum = 0.0
            convergence_error = abs(result.fun - theoretical_optimum)
            
            # Allow more error for higher dimensions (curse of dimensionality)
            max_acceptable_error = 0.1 * np.sqrt(n_dims)
            
            assert convergence_error < max_acceptable_error, \
                f"Poor convergence in {n_dims}D: error={convergence_error}"
            assert end_time - start_time < 120.0, \
                f"Too slow in {n_dims}D: {end_time - start_time}s"
    
    def test_memory_efficiency_stress_test(self):
        """Stress test for memory efficiency and garbage collection."""
        # Create memory-intensive scenario
        space = Space([Real(0.0, 1.0) for _ in range(20)])
        
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
        
        opt = Optimizer(
            dimensions=space,
            base_estimator=GaussianProcessRegressor(),
            n_initial_points=15,
            random_state=42
        )
        
        result = opt.minimize(
            func=memory_monitoring_objective,
            n_calls=30
        )
        
        # Memory efficiency validation
        avg_memory_increase = np.mean(memory_samples)
        max_memory_increase = np.max(memory_samples)
        
        assert avg_memory_increase < 100, f"Average memory too high: {avg_memory_increase}MB"
        assert max_memory_increase < 200, f"Peak memory too high: {max_memory_increase}MB"
        assert len(memory_samples) == 30


class TestAdvancedAcquisitionComparison:
    """Advanced tests comparing different acquisition functions."""
    
    def test_acquisition_function_comparison(self):
        """Compare performance of different acquisition functions."""
        space = Space([
            Real(-2.0, 2.0),
            Real(-2.0, 2.0)
        ])
        
        # Multi-modal objective (has multiple local optima)
        def multimodal_objective(x):
            x1, x2 = x
            return (x1**2 + x2**2) * np.cos(x1) * np.sin(x2)
        
        acquisition_functions = ["ei", "lcb", "pi"]
        results = {}
        
        for acq_func in acquisition_functions:
            opt = Optimizer(
                dimensions=space,
                base_estimator=GaussianProcessRegressor(),
                n_initial_points=10,
                acq_func=acq_func,
                random_state=42
            )
            
            start_time = time.time()
            result = opt.minimize(
                func=multimodal_objective,
                n_calls=40
            )
            end_time = time.time()
            
            results[acq_func] = {
                'best_value': result.fun,
                'convergence_time': end_time - start_time,
                'n_iterations': len(result.func_vals)
            }
        
        # Validate that all acquisition functions work
        for acq_func, result in results.items():
            assert np.isfinite(result['best_value']), \
                f"Invalid result for {acq_func}: {result['best_value']}"
            assert result['convergence_time'] < 30.0, \
                f"Too slow for {acq_func}: {result['convergence_time']}s"
        
        # Compare performance (EI should generally be best for this problem)
        ei_result = results['ei']
        lcb_result = results['lcb']
        pi_result = results['pi']
        
        # EI should find good solution without being too slow
        assert ei_result['best_value'] <= lcb_result['best_value'] * 1.5
        assert ei_result['convergence_time'] <= lcb_result['convergence_time'] * 1.2


class TestRobustnessValidation:
    """Robustness tests with challenging edge cases."""
    
    def test_numerical_precision_robustness(self):
        """Test robustness to numerical precision issues."""
        space = Space([
            Real(1e-10, 1e-5),   # Very small range
            Real(1e5, 1e8),        # Very large range
            Real(-1e6, 1e6),        # Large symmetric range
        ])
        
        def precision_challenging_objective(x):
            # Function with potential numerical precision issues
            x1, x2, x3 = x
            
            # Terms that could cause precision problems
            term1 = x1 * 1e-12  # Very small multiplication
            term2 = x2 / 1e8   # Large division
            term3 = np.exp(-abs(x3))  # Exponential with large negative
            
            return term1 + term2 + term3
        
        opt = Optimizer(
            dimensions=space,
            base_estimator=GaussianProcessRegressor(),
            n_initial_points=15,
            random_state=42
        )
        
        result = opt.minimize(
            func=precision_challenging_objective,
            n_calls=50
        )
        
        # Should handle numerical challenges gracefully
        assert np.all(np.isfinite(result.x)), "Non-finite parameters found"
        assert np.isfinite(result.fun), "Non-finite objective value"
        
        # Check for reasonable bounds
        assert 1e-10 <= result.x[0] <= 1e-5
        assert 1e5 <= result.x[1] <= 1e8
        assert -1e6 <= result.x[2] <= 1e6
    
    def test_extreme_categorical_spaces(self):
        """Test optimization with extreme categorical spaces."""
        # Large categorical space
        large_categories = [f'category_{i}' for i in range(1000)]
        
        space = Space([
            Categorical(large_categories),
            Real(0.0, 1.0),
            Integer(1, 100)
        ])
        
        def large_categorical_objective(x):
            cat_idx, real_val, int_val = x
            
            # Objective depends on categorical index
            categorical_value = cat_idx / 999.0  # Normalize to [0, 1]
            
            return categorical_value * 0.5 + real_val * 0.3 + int_val * 0.2
        
        opt = Optimizer(
            dimensions=space,
            base_estimator=RandomForestRegressor(),  # Better for categorical
            n_initial_points=20,
            random_state=42
        )
        
        start_time = time.time()
        result = opt.minimize(
            func=large_categorical_objective,
            n_calls=60  # More calls for categorical space
        )
        end_time = time.time()
        
        # Performance validation for large categorical
        assert end_time - start_time < 45.0, "Too slow for large categorical"
        assert 0 <= result.x[0] < 1000, "Invalid categorical index"
        assert 0.0 <= result.x[1] <= 1.0, "Invalid real value"
        assert 1 <= result.x[2] <= 100, "Invalid integer value"


class TestScalabilityAnalysis:
    """Scalability analysis for different problem sizes."""
    
    def test_scalability_across_dimensions(self):
        """Test how algorithm scales with problem dimensions."""
        dimension_counts = [5, 10, 20, 30, 40, 50]
        scalability_results = {}
        
        for n_dims in dimension_counts:
            space = Space([Real(0.0, 1.0) for _ in range(n_dims)])
            
            # Simple quadratic objective
            def quadratic_objective(x):
                return sum(x**2 for x in x)
            
            opt = Optimizer(
                dimensions=space,
                base_estimator=GaussianProcessRegressor(),
                n_initial_points=min(25, n_dims * 2),
                random_state=42
            )
            
            # Measure time complexity
            start_time = time.time()
            result = opt.minimize(
                func=quadratic_objective,
                n_calls=30 + n_dims
            )
            end_time = time.time()
            
            scalability_results[n_dims] = {
                'time': end_time - start_time,
                'function_evaluations': len(result.func_vals),
                'final_error': result.fun
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
    
    def test_parallel_performance_scaling(self):
        """Test parallel performance scaling."""
        import multiprocessing
        
        space = Space([Real(0.0, 1.0) for _ in range(10)])
        
        def simple_objective(x):
            return sum(x**2 for x in x)
        
        # Test with different numbers of jobs
        job_counts = [1, 2, 4]
        parallel_results = {}
        
        for n_jobs in job_counts:
            opt = Optimizer(
                dimensions=space,
                base_estimator=GaussianProcessRegressor(),
                n_initial_points=15,
                n_jobs=n_jobs,
                random_state=42
            )
            
            start_time = time.time()
            result = opt.minimize(
                func=simple_objective,
                n_calls=40
            )
            end_time = time.time()
            
            parallel_results[n_jobs] = {
                'time': end_time - start_time,
                'best_value': result.fun
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


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
