"""
Comprehensive Test Suite for Neural Architecture Search

This module provides extensive testing for neural architecture search algorithms,
including unit tests, integration tests, and performance benchmarks.
"""

import unittest
import numpy as np
import time
import warnings
from unittest.mock import patch, MagicMock
import tempfile
import os
import json

# Import NAS modules
from neural_architecture_optimization import (
    NeuralArchitecture, NASMethod, BaseNASOptimizer,
    ReinforcementLearningNAS, EvolutionaryNAS, GradientBasedNAS,
    BayesianOptimizationNAS, MultiObjectiveNAS,
    create_nas_optimizer, benchmark_nas_methods,
    create_default_search_space
)


class TestNeuralArchitecture(unittest.TestCase):
    """Test suite for NeuralArchitecture class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.layers = [
            {'type': 'conv', 'filters': 32, 'kernel_size': 3},
            {'type': 'dense', 'units': 128},
            {'type': 'dropout', 'rate': 0.3}
        ]
        self.connections = [(0, 1), (1, 2)]
        self.parameters = {'test': 'value'}
        
        self.architecture = NeuralArchitecture(
            layers=self.layers,
            connections=self.connections,
            parameters=self.parameters
        )
        
    def test_initialization(self):
        """Test NeuralArchitecture initialization."""
        self.assertEqual(self.architecture.n_layers, 3)
        self.assertEqual(len(self.architecture.connections), 2)
        self.assertEqual(self.architecture.parameters['test'], 'value')
        self.assertIsNotNone(self.architecture.encoding)
        self.assertEqual(len(self.architecture.encoding), 20)
        
    def test_parameter_counting(self):
        """Test parameter counting."""
        # Should count parameters from conv and dense layers
        self.assertGreater(self.architecture.n_parameters, 0)
        
    def test_architecture_encoding(self):
        """Test architecture encoding."""
        encoding = self.architecture.encoding
        
        self.assertEqual(len(encoding), 20)
        self.assertTrue(all(isinstance(x, (int, float)) for x in encoding))
        
    def test_performance_metrics(self):
        """Test performance metrics storage."""
        metrics = {'validation_loss': 0.5, 'accuracy': 0.9}
        self.architecture.performance_metrics = metrics
        
        self.assertEqual(self.architecture.performance_metrics['validation_loss'], 0.5)
        self.assertEqual(self.architecture.performance_metrics['accuracy'], 0.9)


class TestReinforcementLearningNAS(unittest.TestCase):
    """Test suite for Reinforcement Learning NAS."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search_space = create_default_search_space()
        self.optimizer = ReinforcementLearningNAS(
            search_space=self.search_space,
            max_iterations=20,
            random_state=42
        )
        
        # Mock dataset
        self.dataset = {
            'X_train': np.random.randn(100, 10),
            'y_train': np.random.randint(0, 5, 100),
            'X_val': np.random.randn(20, 10),
            'y_val': np.random.randint(0, 5, 20)
        }
        
    def test_initialization(self):
        """Test RL-NAS initialization."""
        self.assertEqual(self.optimizer.learning_rate, 0.01)
        self.assertEqual(self.optimizer.gamma, 0.99)
        self.assertEqual(len(self.optimizer.policy_weights), 20)
        self.assertFalse(self.optimizer.is_fitted)
        
    def test_architecture_sampling(self):
        """Test architecture sampling."""
        architecture = self.optimizer.sample_architecture()
        
        self.assertIsInstance(architecture, NeuralArchitecture)
        self.assertGreater(len(architecture.layers), 0)
        self.assertGreater(architecture.n_parameters, 0)
        
        # Check layer types are from search space
        layer_types = [layer['type'] for layer in architecture.layers]
        for layer_type in layer_types:
            self.assertIn(layer_type, self.search_space['layer_types'])
            
    def test_policy_update(self):
        """Test policy update mechanism."""
        # Create mock architecture
        architecture = NeuralArchitecture(
            layers=[{'type': 'conv', 'filters': 32}],
            connections=[],
            parameters={}
        )
        
        initial_weights = self.optimizer.policy_weights.copy()
        
        # Update policy
        self.optimizer.update_search_strategy(architecture, 0.5)
        
        # Weights should have been updated
        self.assertFalse(np.allclose(initial_weights, self.optimizer.policy_weights))
        
    def test_full_search(self):
        """Test full NAS search process."""
        results = self.optimizer.search(self.dataset)
        
        self.assertIn('best_architecture', results)
        self.assertIn('best_performance', results)
        self.assertIn('search_history', results)
        self.assertIn('evaluations_performed', results)
        self.assertTrue(self.optimizer.is_fitted)
        
        # Check that search was performed
        self.assertGreater(results['evaluations_performed'], 0)
        self.assertLessEqual(results['evaluations_performed'], self.optimizer.evaluation_budget)


class TestEvolutionaryNAS(unittest.TestCase):
    """Test suite for Evolutionary NAS."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search_space = create_default_search_space()
        self.optimizer = EvolutionaryNAS(
            search_space=self.search_space,
            population_size=10,
            max_iterations=15,
            random_state=42
        )
        
        self.dataset = {
            'X_train': np.random.randn(50, 8),
            'y_train': np.random.randint(0, 3, 50),
            'X_val': np.random.randn(10, 8),
            'y_val': np.random.randint(0, 3, 10)
        }
        
    def test_initialization(self):
        """Test Evolutionary NAS initialization."""
        self.assertEqual(self.optimizer.population_size, 10)
        self.assertEqual(self.optimizer.mutation_rate, 0.1)
        self.assertEqual(self.optimizer.crossover_rate, 0.7)
        self.assertEqual(len(self.optimizer.population), 10)
        
    def test_population_initialization(self):
        """Test population initialization."""
        self.assertEqual(len(self.optimizer.population), 10)
        
        for individual in self.optimizer.population:
            self.assertIsInstance(individual, NeuralArchitecture)
            self.assertGreater(len(individual.layers), 0)
            
    def test_mutation_operation(self):
        """Test mutation operation."""
        parent = self.optimizer.population[0]
        mutated = self.optimizer.mutate_architecture(parent)
        
        self.assertIsInstance(mutated, NeuralArchitecture)
        self.assertEqual(mutated.n_layers, parent.n_layers)
        
        # Should be similar but not identical
        parent_encoding = parent.encoding
        mutated_encoding = mutated.encoding
        similarity = np.corrcoef(parent_encoding, mutated_encoding)[0, 1]
        self.assertGreater(similarity, 0.5)  # Similar but not identical
        self.assertLess(similarity, 1.0)    # Not identical
        
    def test_crossover_operation(self):
        """Test crossover operation."""
        parent1 = self.optimizer.population[0]
        parent2 = self.optimizer.population[1]
        
        child = self.optimizer.crossover_architectures(parent1, parent2)
        
        self.assertIsInstance(child, NeuralArchitecture)
        self.assertGreater(len(child.layers), 0)
        
    def test_evolutionary_search(self):
        """Test evolutionary search process."""
        results = self.optimizer.search(self.dataset)
        
        self.assertIn('best_architecture', results)
        self.assertIn('best_performance', results)
        self.assertTrue(self.optimizer.is_fitted)
        
        # Check evolution occurred
        self.assertGreater(results['evaluations_performed'], 0)


class TestGradientBasedNAS(unittest.TestCase):
    """Test suite for Gradient-based NAS."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search_space = create_default_search_space()
        self.optimizer = GradientBasedNAS(
            search_space=self.search_space,
            max_iterations=15,
            random_state=42
        )
        
        self.dataset = {
            'X_train': np.random.randn(30, 6),
            'y_train': np.random.randint(0, 2, 30),
            'X_val': np.random.randn(10, 6),
            'y_val': np.random.randint(0, 2, 10)
        }
        
    def test_initialization(self):
        """Test Gradient-based NAS initialization."""
        self.assertEqual(len(self.optimizer.architecture_weights), 20)
        self.assertEqual(self.optimizer.weight_decay, 1e-3)
        
    def test_weight_updates(self):
        """Test architecture weight updates."""
        initial_weights = self.optimizer.architecture_weights.copy()
        
        # Create mock architecture
        architecture = NeuralArchitecture(
            layers=[{'type': 'dense', 'units': 64}],
            connections=[],
            parameters={}
        )
        
        # Update weights
        self.optimizer.update_search_strategy(architecture, 0.3)
        
        # Weights should have been updated
        self.assertFalse(np.allclose(initial_weights, self.optimizer.architecture_weights))
        
    def test_gradient_based_search(self):
        """Test gradient-based search process."""
        results = self.optimizer.search(self.dataset)
        
        self.assertIn('best_architecture', results)
        self.assertIn('best_performance', results)
        self.assertTrue(self.optimizer.is_fitted)


class TestBayesianOptimizationNAS(unittest.TestCase):
    """Test suite for Bayesian Optimization NAS."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search_space = create_default_search_space()
        self.optimizer = BayesianOptimizationNAS(
            search_space=self.search_space,
            max_iterations=15,
            random_state=42
        )
        
        self.dataset = {
            'X_train': np.random.randn(25, 5),
            'y_train': np.random.randint(0, 4, 25),
            'X_val': np.random.randn(10, 5),
            'y_val': np.random.randint(0, 4, 10)
        }
        
    def test_initialization(self):
        """Test Bayesian Optimization NAS initialization."""
        self.assertEqual(self.optimizer.acquisition_function, 'ei')
        self.assertEqual(self.optimizer.surrogate_model, 'gp')
        
    def test_surrogate_model_training(self):
        """Test surrogate model training."""
        # Add some training data
        for _ in range(5):
            architecture = self.optimizer._random_architecture()
            performance = self.optimizer.evaluate_architecture(architecture, self.dataset)
            self.optimizer.update_search_strategy(architecture, performance)
            
        # Should have training data
        self.assertGreater(len(self.optimizer.X_train), 0)
        self.assertGreater(len(self.optimizer.y_train), 0)
        
    def test_acquisition_function(self):
        """Test acquisition function evaluation."""
        # Add training data
        for _ in range(5):
            architecture = self.optimizer._random_architecture()
            performance = self.optimizer.evaluate_architecture(architecture, self.dataset)
            self.optimizer.update_search_strategy(architecture, performance)
            
        # Sample architecture using acquisition function
        architecture = self.optimizer.sample_architecture()
        
        self.assertIsInstance(architecture, NeuralArchitecture)
        
    def test_bayesian_search(self):
        """Test Bayesian optimization search."""
        results = self.optimizer.search(self.dataset)
        
        self.assertIn('best_architecture', results)
        self.assertIn('best_performance', results)
        self.assertTrue(self.optimizer.is_fitted)


class TestMultiObjectiveNAS(unittest.TestCase):
    """Test suite for Multi-Objective NAS."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search_space = create_default_search_space()
        self.optimizer = MultiObjectiveNAS(
            search_space=self.search_space,
            objectives=['performance', 'complexity', 'inference_time'],
            max_iterations=15,
            random_state=42
        )
        
        self.dataset = {
            'X_train': np.random.randn(20, 4),
            'y_train': np.random.randint(0, 3, 20),
            'X_val': np.random.randn(8, 4),
            'y_val': np.random.randint(0, 3, 8)
        }
        
    def test_initialization(self):
        """Test Multi-Objective NAS initialization."""
        self.assertEqual(len(self.optimizer.objectives), 3)
        self.assertEqual(len(self.optimizer.weights), 3)
        self.assertEqual(len(self.optimizer.pareto_front), 0)
        
    def test_multi_objective_evaluation(self):
        """Test multi-objective evaluation."""
        architecture = NeuralArchitecture(
            layers=[{'type': 'conv', 'filters': 32}, {'type': 'dense', 'units': 64}],
            connections=[(0, 1)],
            parameters={}
        )
        
        performance = self.optimizer.evaluate_architecture(architecture, self.dataset)
        
        self.assertIsInstance(performance, (float, np.floating))
        self.assertTrue(np.isfinite(performance))
        
        # Check that all objectives were computed
        for obj in self.optimizer.objectives:
            self.assertIn(obj, architecture.performance_metrics)
            
    def test_pareto_front_update(self):
        """Test Pareto front update."""
        # Add some architectures
        for i in range(5):
            architecture = NeuralArchitecture(
                layers=[{'type': 'dense', 'units': 32 * (i + 1)}],
                connections=[],
                parameters={}
            )
            
            objectives = {
                'performance': 0.1 * i,
                'complexity': 0.05 * i,
                'inference_time': 0.02 * i
            }
            
            self.optimizer._update_pareto_front(architecture, objectives)
            
        # Should have Pareto front
        self.assertGreater(len(self.optimizer.pareto_front), 0)
        
    def test_multi_objective_search(self):
        """Test multi-objective search process."""
        results = self.optimizer.search(self.dataset)
        
        self.assertIn('best_architecture', results)
        self.assertIn('best_performance', results)
        self.assertTrue(self.optimizer.is_fitted)
        
        # Should have Pareto front
        pareto_front = self.optimizer.get_pareto_front()
        self.assertGreater(len(pareto_front), 0)


class TestNASOptimizerFactory(unittest.TestCase):
    """Test suite for NAS optimizer factory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search_space = create_default_search_space()
        
    def test_create_optimizers(self):
        """Test NAS optimizer creation."""
        optimizer_types = [
            'reinforcement_learning',
            'evolutionary',
            'gradient_based',
            'bayesian_optimization',
            'multi_objective'
        ]
        
        for opt_type in optimizer_types:
            optimizer = create_nas_optimizer(
                opt_type, 
                search_space=self.search_space, 
                max_iterations=10
            )
            
            self.assertIsInstance(optimizer, BaseNASOptimizer)
            self.assertEqual(optimizer.search_space, self.search_space)
            self.assertFalse(optimizer.is_fitted)
            
    def test_invalid_optimizer_type(self):
        """Test invalid optimizer type handling."""
        with self.assertRaises(ValueError):
            create_nas_optimizer('invalid_type', self.search_space)


class TestNASIntegration(unittest.TestCase):
    """Integration tests for NAS algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search_space = create_default_search_space()
        self.dataset = {
            'X_train': np.random.randn(30, 5),
            'y_train': np.random.randint(0, 3, 30),
            'X_val': np.random.randn(10, 5),
            'y_val': np.random.randint(0, 3, 10)
        }
        
    def test_optimizer_comparison(self):
        """Test comparison of different NAS optimizers."""
        optimizer_types = ['reinforcement_learning', 'evolutionary', 'gradient_based']
        results = {}
        
        for opt_type in optimizer_types:
            optimizer = create_nas_optimizer(
                opt_type, 
                search_space=self.search_space, 
                max_iterations=10,
                random_state=42
            )
            
            try:
                search_results = optimizer.search(self.dataset)
                results[opt_type] = search_results['best_performance']
            except Exception as e:
                results[opt_type] = float('inf')
                
        # All should find reasonable solutions
        for opt_type, performance in results.items():
            self.assertLess(performance, 10.0, f"{opt_type} failed to find good solution")
            
    def test_different_dataset_sizes(self):
        """Test optimizers on different dataset sizes."""
        dataset_sizes = [10, 20, 50]
        
        for size in dataset_sizes:
            dataset = {
                'X_train': np.random.randn(size, 4),
                'y_train': np.random.randint(0, 2, size),
                'X_val': np.random.randn(size//4, 4),
                'y_val': np.random.randint(0, 2, size//4)
            }
            
            optimizer = create_nas_optimizer(
                'evolutionary',
                search_space=self.search_space,
                max_iterations=5,
                evaluation_budget=size
            )
            
            results = optimizer.search(dataset)
            
            self.assertIn('best_architecture', results)
            self.assertTrue(results['evaluations_performed'] <= size)
            
    def test_search_space_variations(self):
        """Test optimizers with different search spaces."""
        # Create limited search space
        limited_space = {
            'layer_types': ['dense'],
            'dense_units': [32, 64],
            'max_layers': 3,
            'min_layers': 1
        }
        
        optimizer = create_nas_optimizer(
            'gradient_based',
            search_space=limited_space,
            max_iterations=10
        )
        
        results = optimizer.search(self.dataset)
        
        # Best architecture should respect search space constraints
        best_arch = results['best_architecture']
        for layer in best_arch.layers:
            self.assertEqual(layer['type'], 'dense')
            self.assertIn(layer['units'], [32, 64])


class TestNASPerformance(unittest.TestCase):
    """Performance tests for NAS algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search_space = create_default_search_space()
        self.dataset = {
            'X_train': np.random.randn(40, 6),
            'y_train': np.random.randint(0, 4, 40),
            'X_val': np.random.randn(15, 6),
            'y_val': np.random.randint(0, 4, 15)
        }
        
    def test_optimization_speed(self):
        """Test optimization speed."""
        optimizer = create_nas_optimizer(
            'evolutionary',
            search_space=self.search_space,
            max_iterations=20,
            population_size=5
        )
        
        start_time = time.time()
        results = optimizer.search(self.dataset)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(execution_time, 60.0)  # 60 seconds max
        self.assertTrue(results['evaluations_performed'] > 0)
        
    def test_memory_usage(self):
        """Test memory usage patterns."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and run optimizer
        optimizer = create_nas_optimizer(
            'reinforcement_learning',
            search_space=self.search_space,
            max_iterations=15
        )
        
        results = optimizer.search(self.dataset)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024**2)  # MB
        
        # Memory increase should be reasonable
        self.assertLess(memory_increase, 200)  # Less than 200MB increase
        
    def test_scalability_with_search_space(self):
        """Test scalability with search space size."""
        # Create different sized search spaces
        small_space = {
            'layer_types': ['dense'],
            'dense_units': [32],
            'max_layers': 2,
            'min_layers': 1
        }
        
        large_space = {
            'layer_types': ['conv', 'dense', 'pool', 'dropout'],
            'conv_filters': [16, 32, 64, 128, 256],
            'conv_kernels': [3, 5, 7, 9],
            'dense_units': [32, 64, 128, 256, 512],
            'pool_sizes': [2, 3],
            'dropout_rates': [0.2, 0.3, 0.4, 0.5],
            'max_layers': 10,
            'min_layers': 1
        }
        
        spaces = [small_space, self.search_space, large_space]
        execution_times = []
        
        for space in spaces:
            optimizer = create_nas_optimizer(
                'gradient_based',
                search_space=space,
                max_iterations=10
            )
            
            start_time = time.time()
            results = optimizer.search(self.dataset)
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
            
        # Time should increase with search space size (but not too much)
        self.assertLess(execution_times[-1], execution_times[0] * 20)


class TestNASRobustness(unittest.TestCase):
    """Robustness tests for NAS algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search_space = create_default_search_space()
        
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        optimizer = create_nas_optimizer(
            'evolutionary',
            search_space=self.search_space,
            max_iterations=5,
            population_size=3
        )
        
        # Test with minimal dataset
        minimal_dataset = {
            'X_train': np.random.randn(5, 3),
            'y_train': np.random.randint(0, 2, 5),
            'X_val': np.random.randn(2, 3),
            'y_val': np.random.randint(0, 2, 2)
        }
        
        results = optimizer.search(minimal_dataset)
        
        self.assertIn('best_architecture', results)
        self.assertTrue(np.isfinite(results['best_performance']))
        
    def test_random_state_reproducibility(self):
        """Test reproducibility with random state."""
        optimizer1 = create_nas_optimizer(
            'reinforcement_learning',
            search_space=self.search_space,
            max_iterations=10,
            random_state=42
        )
        
        optimizer2 = create_nas_optimizer(
            'reinforcement_learning',
            search_space=self.search_space,
            max_iterations=10,
            random_state=42
        )
        
        dataset = {
            'X_train': np.random.randn(20, 4),
            'y_train': np.random.randint(0, 3, 20),
            'X_val': np.random.randn(8, 4),
            'y_val': np.random.randint(0, 3, 8)
        }
        
        results1 = optimizer1.search(dataset)
        results2 = optimizer2.search(dataset)
        
        # Results should be similar (though not necessarily identical due to randomness)
        self.assertLess(
            abs(results1['best_performance'] - results2['best_performance']), 
            1.0
        )
        
    def test_invalid_search_space(self):
        """Test handling of invalid search spaces."""
        # Empty search space
        empty_space = {}
        
        with self.assertRaises(Exception):
            create_nas_optimizer('evolutionary', search_space=empty_space)
            
    def test_evaluation_budget_exhaustion(self):
        """Test behavior when evaluation budget is exhausted."""
        optimizer = create_nas_optimizer(
            'gradient_based',
            search_space=self.search_space,
            max_iterations=20,
            evaluation_budget=5
        )
        
        dataset = {
            'X_train': np.random.randn(30, 5),
            'y_train': np.random.randint(0, 3, 30),
            'X_val': np.random.randn(10, 5),
            'y_val': np.random.randint(0, 3, 10)
        }
        
        results = optimizer.search(dataset)
        
        # Should stop at budget limit
        self.assertLessEqual(results['evaluations_performed'], 5)


def run_nas_tests():
    """Run comprehensive test suite for NAS algorithms."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestNeuralArchitecture,
        TestReinforcementLearningNAS,
        TestEvolutionaryNAS,
        TestGradientBasedNAS,
        TestBayesianOptimizationNAS,
        TestMultiObjectiveNAS,
        TestNASOptimizerFactory,
        TestNASIntegration,
        TestNASPerformance,
        TestNASRobustness
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
        
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate report
    report = {
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save report
    with open('nas_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\n{'='*60}")
    print("NEURAL ARCHITECTURE SEARCH TEST REPORT")
    print(f"{'='*60}")
    print(f"Total Tests: {report['total_tests']}")
    print(f"Failures: {report['failures']}")
    print(f"Errors: {report['errors']}")
    print(f"Success Rate: {report['success_rate']:.2f}%")
    print(f"Timestamp: {report['timestamp']}")
    print(f"{'='*60}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Suppress warnings for cleaner test output
    warnings.filterwarnings('ignore')
    
    print("Neural Architecture Search - Comprehensive Test Suite")
    print("=" * 65)
    print("Testing all NAS algorithms...")
    print()
    
    # Run comprehensive tests
    success = run_nas_tests()
    
    if success:
        print("\n🎉 ALL NAS TESTS PASSED!")
    else:
        print("\n⚠️  Some tests failed. Please review the output above for details.")
        
    print("\nNAS testing completed.")
