"""
Ultra-Advanced Optimization Suite - Examples and Demonstrations

This file contains comprehensive examples demonstrating the capabilities
of the ultra-advanced optimization algorithms and meta-learning techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# Example 1: Ultra-Adaptive Bayesian Optimization
def example_ultra_adaptive_optimization():
    """Demonstrate ultra-adaptive Bayesian optimization."""
    print("=" * 60)
    print("Example 1: Ultra-Adaptive Bayesian Optimization")
    print("=" * 60)
    
    try:
        from ultra_adaptive_optimizer import UltraAdaptiveBayesianOptimizer
        
        # Define a complex objective function
        def complex_objective(x):
            # Multi-modal function with local minima
            term1 = np.sum((x - 2)**2)
            term2 = 10 * np.sin(x[0]) * np.cos(x[1])
            term3 = 5 * np.sum(np.sin(x))
            return term1 + term2 + term3
        
        # Create optimizer
        optimizer = UltraAdaptiveBayesianOptimizer(
            dimensions=3,
            max_iterations=50,
            random_state=42
        )
        
        # Run optimization
        print("Running ultra-adaptive optimization...")
        start_time = time.time()
        results = optimizer.optimize(complex_objective)
        end_time = time.time()
        
        # Display results
        print(f"✅ Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"🎯 Best solution: {results['best_x']}")
        print(f"📊 Best value: {results['best_y']:.6f}")
        print(f"🔄 Iterations: {results['iterations']}")
        print(f"📈 Convergence: {'Yes' if results['iterations'] < 50 else 'No'}")
        
        return results
        
    except ImportError:
        print("❌ Ultra-adaptive optimizer not available")
        return None

# Example 2: Quantum-Inspired Optimization
def example_quantum_optimization():
    """Demonstrate quantum-inspired optimization algorithms."""
    print("\n" + "=" * 60)
    print("Example 2: Quantum-Inspired Optimization")
    print("=" * 60)
    
    try:
        from quantum_optimization_algorithms import create_quantum_optimizer, benchmark_quantum_optimizers
        
        # Define test functions
        def sphere_function(x):
            return np.sum(x**2)
        
        def rastrigin_function(x):
            n = len(x)
            return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        
        # Test different quantum optimizers
        optimizers = ['quantum_annealing', 'qaoa_inspired', 'quantum_walk']
        results = {}
        
        for opt_type in optimizers:
            print(f"\n🔬 Testing {opt_type} optimizer...")
            
            try:
                optimizer = create_quantum_optimizer(
                    opt_type, 
                    n_qubits=3, 
                    max_iterations=30,
                    random_state=42
                )
                
                # Optimize sphere function
                start_time = time.time()
                opt_results = optimizer.optimize(sphere_function)
                end_time = time.time()
                
                results[opt_type] = {
                    'best_energy': opt_results['best_energy'],
                    'time': end_time - start_time,
                    'iterations': opt_results['iterations']
                }
                
                print(f"  ✅ Best energy: {opt_results['best_energy']:.6f}")
                print(f"  ⏱️  Time: {end_time - start_time:.2f}s")
                print(f"  🔄 Iterations: {opt_results['iterations']}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results[opt_type] = {'error': str(e)}
        
        # Compare results
        print(f"\n📊 Quantum Optimization Comparison:")
        print("-" * 40)
        for opt_type, result in results.items():
            if 'error' not in result:
                print(f"{opt_type:20}: {result['best_energy']:10.6f} ({result['time']:.2f}s)")
            else:
                print(f"{opt_type:20}: Failed")
        
        return results
        
    except ImportError:
        print("❌ Quantum optimization algorithms not available")
        return None

# Example 3: Neural Architecture Search
def example_neural_architecture_search():
    """Demonstrate neural architecture search."""
    print("\n" + "=" * 60)
    print("Example 3: Neural Architecture Search")
    print("=" * 60)
    
    try:
        from neural_architecture_optimization import (
            create_nas_optimizer, 
            create_default_search_space,
            benchmark_nas_methods
        )
        
        # Create search space
        search_space = create_default_search_space()
        print(f"🔍 Search space: {len(search_space['layer_types'])} layer types")
        
        # Create synthetic dataset
        np.random.seed(42)
        dataset = {
            'X_train': np.random.randn(100, 10),
            'y_train': np.random.randint(0, 5, 100),
            'X_val': np.random.randn(30, 10),
            'y_val': np.random.randint(0, 5, 30)
        }
        
        print(f"📊 Dataset: {dataset['X_train'].shape[0]} training samples")
        print(f"🏷️  Classes: {len(np.unique(dataset['y_train']))}")
        
        # Test different NAS methods
        methods = ['evolutionary', 'reinforcement_learning', 'gradient_based']
        results = {}
        
        for method in methods:
            print(f"\n🧠 Testing {method} NAS...")
            
            try:
                optimizer = create_nas_optimizer(
                    method,
                    search_space=search_space,
                    max_iterations=20,
                    random_state=42
                )
                
                # Run architecture search
                start_time = time.time()
                search_results = optimizer.search(dataset)
                end_time = time.time()
                
                results[method] = {
                    'best_performance': search_results['best_performance'],
                    'time': end_time - start_time,
                    'evaluations': search_results['evaluations_performed']
                }
                
                print(f"  ✅ Best performance: {search_results['best_performance']:.6f}")
                print(f"  ⏱️  Time: {end_time - start_time:.2f}s")
                print(f"  🔢 Evaluations: {search_results['evaluations_performed']}")
                
                if search_results['best_architecture']:
                    arch = search_results['best_architecture']
                    print(f"  🏗️  Best architecture: {arch.n_layers} layers, {arch.n_parameters} params")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results[method] = {'error': str(e)}
        
        return results
        
    except ImportError:
        print("❌ Neural architecture search not available")
        return None

# Example 4: Meta-Learning for Few-Shot Learning
def example_meta_learning():
    """Demonstrate meta-learning algorithms."""
    print("\n" + "=" * 60)
    print("Example 4: Meta-Learning for Few-Shot Learning")
    print("=" * 60)
    
    try:
        from advanced_meta_learning import (
            create_meta_learner,
            generate_classification_tasks,
            generate_regression_tasks,
            SimpleMLP
        )
        
        # Generate synthetic tasks
        print("🎯 Generating meta-learning tasks...")
        
        # Classification tasks
        train_tasks = generate_classification_tasks(
            n_tasks=15, n_classes=5, n_support=5, n_query=10, 
            input_dim=8, random_state=42
        )
        
        test_tasks = generate_classification_tasks(
            n_tasks=5, n_classes=5, n_support=5, n_query=10,
            input_dim=8, random_state=123
        )
        
        print(f"📊 Generated {len(train_tasks)} training tasks and {len(test_tasks)} test tasks")
        
        # Test different meta-learning methods
        methods = ['maml', 'prototypical_networks', 'reptile']
        results = {}
        
        for method in methods:
            print(f"\n🧠 Testing {method} meta-learning...")
            
            try:
                # Create base model
                base_model = SimpleMLP(input_dim=8, hidden_dim=64, output_dim=5)
                
                # Create meta-learner
                meta_learner = create_meta_learner(
                    method, 
                    base_model=base_model,
                    max_iterations=20,
                    random_state=42
                )
                
                # Meta-train
                start_time = time.time()
                train_results = meta_learner.meta_train(train_tasks, n_epochs=15)
                end_time = time.time()
                
                # Meta-test
                test_results = meta_learner.meta_test(test_tasks)
                
                results[method] = {
                    'train_time': end_time - start_time,
                    'test_accuracy': test_results.get('accuracy', 0),
                    'final_loss': train_results['meta_history'][-1]['metrics'].get('loss', 0)
                }
                
                print(f"  ✅ Test accuracy: {test_results.get('accuracy', 0):.4f}")
                print(f"  ⏱️  Train time: {end_time - start_time:.2f}s")
                print(f"  📉 Final loss: {train_results['meta_history'][-1]['metrics'].get('loss', 0):.6f}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results[method] = {'error': str(e)}
        
        return results
        
    except ImportError:
        print("❌ Meta-learning algorithms not available")
        return None

# Example 5: Ensemble Optimization Methods
def example_ensemble_optimization():
    """Demonstrate ensemble optimization methods."""
    print("\n" + "=" * 60)
    print("Example 5: Ensemble Optimization Methods")
    print("=" * 60)
    
    try:
        from ultra_ensemble_methods import (
            HeterogeneousEnsembleOptimizer,
            DynamicEnsembleOptimizer,
            RobustEnsembleOptimizer,
            EnsembleOptimizerFactory
        )
        
        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 6)
        y_train = np.sum(X_train**2, axis=1) + np.random.randn(100) * 0.1
        X_test = np.random.randn(30, 6)
        y_test = np.sum(X_test**2, axis=1)
        
        print(f"📊 Dataset: {X_train.shape[0]} training samples, {X_train.shape[1]} features")
        
        # Test different ensemble methods
        ensemble_types = ['heterogeneous', 'dynamic', 'robust']
        results = {}
        
        for ensemble_type in ensemble_types:
            print(f"\n🎭 Testing {ensemble_type} ensemble...")
            
            try:
                # Create ensemble optimizer
                ensemble = EnsembleOptimizerFactory.create_ensemble(
                    ensemble_type,
                    ensemble_size=3,
                    random_state=42
                )
                
                # Fit ensemble
                start_time = time.time()
                ensemble.fit(X_train, y_train)
                end_time = time.time()
                
                # Make predictions
                y_pred = ensemble.predict(X_test)
                mse = np.mean((y_test - y_pred)**2)
                
                results[ensemble_type] = {
                    'mse': mse,
                    'time': end_time - start_time,
                    'ensemble_size': len(ensemble.ensemble_members)
                }
                
                print(f"  ✅ Test MSE: {mse:.6f}")
                print(f"  ⏱️  Time: {end_time - start_time:.2f}s")
                print(f"  👥 Ensemble size: {len(ensemble.ensemble_members)}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results[ensemble_type] = {'error': str(e)}
        
        return results
        
    except ImportError:
        print("❌ Ensemble optimization methods not available")
        return None

# Example 6: Performance Analytics
def example_performance_analytics():
    """Demonstrate performance analytics and monitoring."""
    print("\n" + "=" * 60)
    print("Example 6: Performance Analytics and Monitoring")
    print("=" * 60)
    
    try:
        from ultra_performance_analytics import (
            SystemResourceMonitor,
            OptimizationPerformanceMonitor,
            ConvergenceAnalyzer,
            PerformanceBenchmark
        )
        
        print("🖥️  System Resource Monitoring")
        print("-" * 30)
        
        # Monitor system resources
        system_monitor = SystemResourceMonitor(sampling_interval=0.5, max_history=10)
        system_monitor.start_monitoring()
        
        # Simulate some work
        time.sleep(2)
        
        system_monitor.stop_monitoring()
        
        # Get system metrics
        system_metrics = system_monitor.get_metrics_dataframe()
        if not system_metrics.empty:
            print(f"  📊 CPU usage: {system_metrics[system_metrics['name'] == 'cpu_percent']['value'].mean():.2f}%")
            print(f"  💾 Memory usage: {system_metrics[system_metrics['name'] == 'memory_percent']['value'].mean():.2f}%")
        
        print("\n📈 Optimization Performance Monitoring")
        print("-" * 40)
        
        # Monitor optimization performance
        opt_monitor = OptimizationPerformanceMonitor()
        opt_monitor.start_optimization()
        
        # Simulate optimization iterations
        for i in range(10):
            x = np.random.randn(5)
            y = np.sum(x**2) + np.random.randn() * 0.1
            opt_monitor.record_iteration(x, y)
        
        opt_metrics = opt_monitor.get_metrics_dataframe()
        if not opt_metrics.empty:
            print(f"  🎯 Best value: {opt_metrics[opt_metrics['name'] == 'best_value']['value'].min():.6f}")
            print(f"  🔄 Iterations: {len(opt_metrics[opt_metrics['name'] == 'iteration_count'])}")
        
        print("\n📊 Convergence Analysis")
        print("-" * 25)
        
        # Analyze convergence
        analyzer = ConvergenceAnalyzer()
        convergence_values = [10.0, 8.0, 6.0, 5.0, 4.5, 4.2, 4.1, 4.05, 4.02, 4.01]
        
        analysis = analyzer.analyze_convergence(convergence_values)
        
        print(f"  ✅ Converged: {analysis['converged']}")
        print(f"  🎯 Best value: {analysis['best_value']:.6f}")
        print(f"  📈 Convergence rate: {analysis['convergence_rate']:.6f}")
        
        return {
            'system_metrics': system_metrics,
            'opt_metrics': opt_metrics,
            'convergence_analysis': analysis
        }
        
    except ImportError:
        print("❌ Performance analytics not available")
        return None

# Example 7: Space Transformations
def example_space_transformations():
    """Demonstrate advanced space transformation techniques."""
    print("\n" + "=" * 60)
    print("Example 7: Advanced Space Transformations")
    print("=" * 60)
    
    try:
        from ultra_space_manipulations import (
            AdaptiveManifoldTransformer,
            TopologyAwareTransformer,
            MultiScaleSpaceTransformer,
            analyze_space_complexity
        )
        
        # Create high-dimensional data
        np.random.seed(42)
        X = np.random.randn(100, 15)
        y = np.sum(X[:, :5]**2, axis=1) + np.sum(X[:, 5:10], axis=1)
        
        print(f"📊 Original data: {X.shape[0]} samples, {X.shape[1]} dimensions")
        
        # Analyze space complexity
        print("\n🔍 Space Complexity Analysis")
        print("-" * 30)
        
        complexity = analyze_space_complexity(X)
        for metric, value in complexity.items():
            print(f"  📊 {metric}: {value:.4f}")
        
        # Test different transformers
        transformers = {
            'Adaptive Manifold': AdaptiveManifoldTransformer(random_state=42),
            'Topology Aware': TopologyAwareTransformer(random_state=42),
            'Multi-Scale': MultiScaleSpaceTransformer(random_state=42)
        }
        
        results = {}
        
        for name, transformer in transformers.items():
            print(f"\n🔄 {name} Transformation")
            print("-" * 30)
            
            try:
                # Fit and transform
                start_time = time.time()
                X_transformed = transformer.fit_transform(X, y)
                end_time = time.time()
                
                results[name] = {
                    'output_shape': X_transformed.shape,
                    'time': end_time - start_time,
                    'intrinsic_dim': getattr(transformer, 'intrinsic_dim_', None)
                }
                
                print(f"  ✅ Output shape: {X_transformed.shape}")
                print(f"  ⏱️  Time: {end_time - start_time:.3f}s")
                
                if hasattr(transformer, 'intrinsic_dim_'):
                    print(f"  🎯 Intrinsic dimension: {transformer.intrinsic_dim_}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results[name] = {'error': str(e)}
        
        return results
        
    except ImportError:
        print("❌ Space transformation methods not available")
        return None

# Example 8: Comprehensive Benchmark
def example_comprehensive_benchmark():
    """Run comprehensive benchmark of all algorithms."""
    print("\n" + "=" * 60)
    print("Example 8: Comprehensive Benchmark")
    print("=" * 60)
    
    benchmark_results = {}
    
    # Test ultra-adaptive optimization
    print("🚀 Benchmarking Ultra-Adaptive Optimization...")
    try:
        ultra_results = example_ultra_adaptive_optimization()
        if ultra_results:
            benchmark_results['ultra_adaptive'] = ultra_results['best_y']
    except Exception as e:
        print(f"❌ Ultra-adaptive benchmark failed: {e}")
    
    # Test quantum optimization
    print("\n🔬 Benchmarking Quantum Optimization...")
    try:
        quantum_results = example_quantum_optimization()
        if quantum_results:
            quantum_best = min([r['best_energy'] for r in quantum_results.values() if 'error' not in r])
            benchmark_results['quantum'] = quantum_best
    except Exception as e:
        print(f"❌ Quantum benchmark failed: {e}")
    
    # Test NAS
    print("\n🧠 Benchmarking Neural Architecture Search...")
    try:
        nas_results = example_neural_architecture_search()
        if nas_results:
            nas_best = min([r['best_performance'] for r in nas_results.values() if 'error' not in r])
            benchmark_results['nas'] = nas_best
    except Exception as e:
        print(f"❌ NAS benchmark failed: {e}")
    
    # Test meta-learning
    print("\n🎯 Benchmarking Meta-Learning...")
    try:
        meta_results = example_meta_learning()
        if meta_results:
            meta_best = max([r['test_accuracy'] for r in meta_results.values() if 'error' not in r])
            benchmark_results['meta_learning'] = meta_best
    except Exception as e:
        print(f"❌ Meta-learning benchmark failed: {e}")
    
    # Test ensemble methods
    print("\n🎭 Benchmarking Ensemble Methods...")
    try:
        ensemble_results = example_ensemble_optimization()
        if ensemble_results:
            ensemble_best = min([r['mse'] for r in ensemble_results.values() if 'error' not in r])
            benchmark_results['ensemble'] = ensemble_best
    except Exception as e:
        print(f"❌ Ensemble benchmark failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    for method, result in benchmark_results.items():
        if isinstance(result, (int, float)):
            print(f"{method:20}: {result:10.6f}")
        else:
            print(f"{method:20}: {result}")
    
    return benchmark_results

# Main function to run all examples
def run_all_examples():
    """Run all examples and demonstrate the ultra-advanced optimization suite."""
    print("🚀 ULTRA-ADVANCED OPTIMIZATION SUITE - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("This demonstration showcases all cutting-edge optimization algorithms")
    print("and meta-learning techniques available in the suite.")
    print("=" * 80)
    
    # Track results
    all_results = {}
    
    # Run all examples
    examples = [
        ("Ultra-Adaptive Optimization", example_ultra_adaptive_optimization),
        ("Quantum Optimization", example_quantum_optimization),
        ("Neural Architecture Search", example_neural_architecture_search),
        ("Meta-Learning", example_meta_learning),
        ("Ensemble Optimization", example_ensemble_optimization),
        ("Performance Analytics", example_performance_analytics),
        ("Space Transformations", example_space_transformations)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n🔄 Running {name} example...")
            result = example_func()
            all_results[name] = result
            print(f"✅ {name} example completed successfully")
        except Exception as e:
            print(f"❌ {name} example failed: {e}")
            all_results[name] = {'error': str(e)}
    
    # Run comprehensive benchmark
    print("\n" + "=" * 80)
    print("🏁 RUNNING COMPREHENSIVE BENCHMARK")
    print("=" * 80)
    
    try:
        benchmark_results = example_comprehensive_benchmark()
        all_results['comprehensive_benchmark'] = benchmark_results
    except Exception as e:
        print(f"❌ Comprehensive benchmark failed: {e}")
        all_results['comprehensive_benchmark'] = {'error': str(e)}
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("Summary of all examples and benchmarks:")
    
    successful_examples = 0
    total_examples = len(examples) + 1  # +1 for comprehensive benchmark
    
    for name, result in all_results.items():
        if 'error' not in result:
            print(f"✅ {name}: SUCCESS")
            successful_examples += 1
        else:
            print(f"❌ {name}: FAILED")
    
    print(f"\n📊 Success Rate: {successful_examples}/{total_examples} ({successful_examples/total_examples*100:.1f}%)")
    
    if successful_examples == total_examples:
        print("\n🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("The Ultra-Advanced Optimization Suite is fully functional and ready for use!")
    else:
        print(f"\n⚠️  {total_examples - successful_examples} examples failed.")
        print("Please check the error messages above for troubleshooting.")
    
    print("\n📚 For more information, see README_ULTRA_OPTIMIZATION.md")
    print("🔧 For API documentation, check the docstrings in each module")
    print("🧪 For testing, run the test files: test_*.py")
    
    return all_results

if __name__ == "__main__":
    # Run all examples
    results = run_all_examples()
    
    # Save results to file (optional)
    try:
        import json
        with open('examples_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for k, v in value.items():
                        if hasattr(v, 'tolist'):
                            json_results[key][k] = v.tolist()
                        elif isinstance(v, (int, float, str, bool)):
                            json_results[key][k] = v
                        else:
                            json_results[key][k] = str(v)
                else:
                    json_results[key] = str(value)
            
            json.dump(json_results, f, indent=2)
        print(f"\n💾 Results saved to examples_results.json")
    except Exception as e:
        print(f"⚠️  Could not save results to file: {e}")
