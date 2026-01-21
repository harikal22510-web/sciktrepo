"""
Comprehensive Test Suite for Quantum Optimization Algorithms

This module provides extensive testing for quantum-inspired optimization algorithms,
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

# Import quantum optimization modules
from quantum_optimization_algorithms import (
    QuantumAnnealingOptimizer, QAOAInspiredOptimizer, QuantumWalkOptimizer,
    QuantumNeuralOptimizer, QuantumEvolutionaryOptimizer, QuantumSwarmOptimizer,
    create_quantum_optimizer, benchmark_quantum_optimizers
)


class TestQuantumAnnealingOptimizer(unittest.TestCase):
    """Test suite for Quantum Annealing optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_qubits = 4
        self.optimizer = QuantumAnnealingOptimizer(
            n_qubits=self.n_qubits,
            max_iterations=50,
            random_state=42
        )
        
        # Simple test objective function
        def sphere_function(x):
            return np.sum(x**2)
        
        self.objective_function = sphere_function
        
    def test_initialization(self):
        """Test QuantumAnnealingOptimizer initialization."""
        self.assertEqual(self.optimizer.n_qubits, self.n_qubits)
        self.assertEqual(self.optimizer.max_iterations, 50)
        self.assertEqual(self.optimizer.initial_temperature, 10.0)
        self.assertEqual(self.optimizer.final_temperature, 0.01)
        self.assertFalse(self.optimizer.is_fitted)
        
    def test_quantum_state_initialization(self):
        """Test quantum state initialization."""
        self.optimizer.initialize_quantum_state()
        
        self.assertIsNotNone(self.optimizer.quantum_state)
        self.assertEqual(self.optimizer.quantum_state.dimension, 2**self.n_qubits)
        self.assertAlmostEqual(
            np.sum(np.abs(self.optimizer.quantum_state.amplitudes)**2), 
            1.0, 
            places=6
        )
        
    def test_hamiltonian_construction(self):
        """Test Hamiltonian construction."""
        self.optimizer.initialize_quantum_state()
        hamiltonian = self.optimizer.construct_hamiltonian(self.objective_function)
        
        self.assertEqual(hamiltonian.shape, (2**self.n_qubits, 2**self.n_qubits))
        self.assertTrue(np.allclose(hamiltonian, hamiltonian.conj().T))  # Hermitian
        
    def test_optimization_step(self):
        """Test single optimization step."""
        self.optimizer.initialize_quantum_state()
        self.optimizer.hamiltonian = self.optimizer.construct_hamiltonian(self.objective_function)
        
        energy = self.optimizer.optimize_step(self.objective_function)
        
        self.assertIsInstance(energy, (float, np.floating))
        self.assertTrue(np.isfinite(energy))
        self.assertGreaterEqual(energy, 0)  # Sphere function is non-negative
        
    def test_full_optimization(self):
        """Test full optimization process."""
        results = self.optimizer.optimize(self.objective_function)
        
        self.assertIn('best_solution', results)
        self.assertIn('best_energy', results)
        self.assertIn('iterations', results)
        self.assertIn('convergence_history', results)
        self.assertTrue(self.optimizer.is_fitted)
        
        # Check that best energy is reasonable for sphere function
        self.assertGreaterEqual(results['best_energy'], 0)
        self.assertLessEqual(results['iterations'], self.optimizer.max_iterations)
        
    def test_temperature_schedule(self):
        """Test temperature annealing schedule."""
        schedules = ['exponential', 'linear']
        
        for schedule in schedules:
            optimizer = QuantumAnnealingOptimizer(
                n_qubits=3,
                temperature_schedule=schedule,
                max_iterations=20,
                random_state=42
            )
            
            optimizer.optimize(self.objective_function)
            
            # Check that temperature decreases
            initial_temp = optimizer.initial_temperature
            final_temp = optimizer.current_temperature
            self.assertLessEqual(final_temp, initial_temp)


class TestQAOAInspiredOptimizer(unittest.TestCase):
    """Test suite for QAOA-inspired optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_qubits = 3
        self.optimizer = QAOAInspiredOptimizer(
            n_qubits=self.n_qubits,
            depth=2,
            max_iterations=30,
            random_state=42
        )
        
        def simple_function(x):
            return np.sum(x**2) - 2 * np.sum(x)
        
        self.objective_function = simple_function
        
    def test_initialization(self):
        """Test QAOAInspiredOptimizer initialization."""
        self.assertEqual(self.optimizer.n_qubits, self.n_qubits)
        self.assertEqual(self.optimizer.depth, 2)
        self.assertEqual(len(self.optimizer.gamma_parameters), 2)
        self.assertEqual(len(self.optimizer.beta_parameters), 2)
        
    def test_qaoa_layer_application(self):
        """Test QAOA layer application."""
        self.optimizer.initialize_quantum_state()
        self.optimizer.hamiltonian = self.optimizer.construct_hamiltonian(self.objective_function)
        
        # Store initial state
        initial_amplitudes = self.optimizer.quantum_state.amplitudes.copy()
        
        # Apply QAOA layer
        self.optimizer.apply_qaoa_layer(0.5, 0.3)
        
        # State should have changed
        self.assertFalse(np.allclose(initial_amplitudes, self.optimizer.quantum_state.amplitudes))
        
    def test_parameter_updates(self):
        """Test parameter updates during optimization."""
        initial_gamma = self.optimizer.gamma_parameters.copy()
        initial_beta = self.optimizer.beta_parameters.copy()
        
        self.optimizer.optimize(self.objective_function)
        
        # Parameters should have been updated
        self.assertFalse(np.allclose(initial_gamma, self.optimizer.gamma_parameters))
        self.assertFalse(np.allclose(initial_beta, self.optimizer.beta_parameters))


class TestQuantumWalkOptimizer(unittest.TestCase):
    """Test suite for Quantum Walk optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_qubits = 3
        self.optimizer = QuantumWalkOptimizer(
            n_qubits=self.n_qubits,
            walk_type='continuous',
            max_iterations=30,
            random_state=42
        )
        
        def test_function(x):
            return np.sum((x - 1)**2)
        
        self.objective_function = test_function
        
    def test_initialization(self):
        """Test QuantumWalkOptimizer initialization."""
        self.assertEqual(self.optimizer.walk_type, 'continuous')
        self.assertEqual(self.optimizer.step_size, 0.1)
        
    def test_adjacency_matrix_construction(self):
        """Test adjacency matrix construction."""
        self.optimizer.initialize_quantum_state()
        self.optimizer.hamiltonian = self.optimizer.construct_hamiltonian(self.objective_function)
        
        # Check that adjacency matrix was created
        self.assertIsNotNone(self.optimizer.adjacency_matrix)
        dimension = 2**self.n_qubits
        self.assertEqual(self.optimizer.adjacency_matrix.shape, (dimension, dimension))
        
        # Check that it's symmetric
        self.assertTrue(np.allclose(self.optimizer.adjacency_matrix, self.optimizer.adjacency_matrix.T))
        
    def test_discrete_walk(self):
        """Test discrete quantum walk."""
        optimizer = QuantumWalkOptimizer(
            n_qubits=self.n_qubits,
            walk_type='discrete',
            max_iterations=20,
            random_state=42
        )
        
        optimizer.initialize_quantum_state()
        optimizer.hamiltonian = optimizer.construct_hamiltonian(self.objective_function)
        
        initial_amplitudes = optimizer.quantum_state.amplitudes.copy()
        energy = optimizer.optimize_step(self.objective_function)
        
        # State should have changed
        self.assertFalse(np.allclose(initial_amplitudes, optimizer.quantum_state.amplitudes))
        self.assertTrue(np.isfinite(energy))


class TestQuantumNeuralOptimizer(unittest.TestCase):
    """Test suite for Quantum Neural optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_qubits = 3
        self.optimizer = QuantumNeuralOptimizer(
            n_qubits=self.n_qubits,
            n_layers=2,
            max_iterations=20,
            random_state=42
        )
        
        def neural_function(x):
            return np.sum(x**3) - np.sum(x)
        
        self.objective_function = neural_function
        
    def test_initialization(self):
        """Test QuantumNeuralOptimizer initialization."""
        self.assertEqual(self.optimizer.n_layers, 2)
        self.assertEqual(self.optimizer.weights.shape, (2, self.n_qubits))
        self.assertEqual(len(self.optimizer.biases), 2)
        
    def test_neural_layer_application(self):
        """Test quantum neural layer application."""
        self.optimizer.initialize_quantum_state()
        self.optimizer.hamiltonian = self.optimizer.construct_hamiltonian(self.objective_function)
        
        initial_amplitudes = self.optimizer.quantum_state.amplitudes.copy()
        
        # Apply neural layer
        self.optimizer.quantum_neural_layer(0)
        
        # State should have changed
        self.assertFalse(np.allclose(initial_amplitudes, self.optimizer.quantum_state.amplitudes))
        
    def test_weight_updates(self):
        """Test weight updates during optimization."""
        initial_weights = self.optimizer.weights.copy()
        initial_biases = self.optimizer.biases.copy()
        
        self.optimizer.optimize(self.objective_function)
        
        # Weights and biases should have been updated
        self.assertFalse(np.allclose(initial_weights, self.optimizer.weights))
        self.assertFalse(np.allclose(initial_biases, self.optimizer.biases))


class TestQuantumEvolutionaryOptimizer(unittest.TestCase):
    """Test suite for Quantum Evolutionary optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_qubits = 3
        self.optimizer = QuantumEvolutionaryOptimizer(
            n_qubits=self.n_qubits,
            population_size=10,
            max_iterations=20,
            random_state=42
        )
        
        def evolutionary_function(x):
            return np.sum(x**4) - 2 * np.sum(x**2)
        
        self.objective_function = evolutionary_function
        
    def test_initialization(self):
        """Test QuantumEvolutionaryOptimizer initialization."""
        self.assertEqual(self.optimizer.population_size, 10)
        self.assertEqual(self.optimizer.mutation_rate, 0.1)
        self.assertEqual(self.optimizer.crossover_rate, 0.7)
        
    def test_population_initialization(self):
        """Test quantum population initialization."""
        self.optimizer.initialize_quantum_state()
        
        self.assertEqual(len(self.optimizer.quantum_population), 10)
        self.assertEqual(len(self.optimizer.personal_best), 10)
        
        # Check that all individuals have correct dimension
        for individual in self.optimizer.quantum_population:
            self.assertEqual(individual.dimension, 2**self.n_qubits)
            
    def test_quantum_crossover(self):
        """Test quantum crossover operation."""
        self.optimizer.initialize_quantum_state()
        
        parent1 = self.optimizer.quantum_population[0]
        parent2 = self.optimizer.quantum_population[1]
        
        child = self.optimizer.quantum_crossover(parent1, parent2)
        
        self.assertEqual(child.dimension, parent1.dimension)
        self.assertTrue(np.isfinite(np.sum(child.amplitudes)))
        
    def test_quantum_mutation(self):
        """Test quantum mutation operation."""
        self.optimizer.initialize_quantum_state()
        
        parent = self.optimizer.quantum_population[0]
        mutated = self.optimizer.quantum_mutation(parent)
        
        self.assertEqual(mutated.dimension, parent.dimension)
        self.assertTrue(np.isfinite(np.sum(mutated.amplitudes)))
        
        # Should be different from parent (with high probability)
        similarity = np.corrcoef(
            np.abs(parent.amplitudes)**2, 
            np.abs(mutated.amplitudes)**2
        )[0, 1]
        self.assertLess(similarity, 0.99)  # Not identical


class TestQuantumSwarmOptimizer(unittest.TestCase):
    """Test suite for Quantum Swarm optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_qubits = 3
        self.optimizer = QuantumSwarmOptimizer(
            n_qubits=self.n_qubits,
            swarm_size=8,
            max_iterations=20,
            random_state=42
        )
        
        def swarm_function(x):
            return np.sum((x - 0.5)**2)
        
        self.objective_function = swarm_function
        
    def test_initialization(self):
        """Test QuantumSwarmOptimizer initialization."""
        self.assertEqual(self.optimizer.swarm_size, 8)
        self.assertEqual(self.optimizer.inertia_weight, 0.7)
        self.assertEqual(self.optimizer.cognitive_weight, 1.5)
        self.assertEqual(self.optimizer.social_weight, 1.5)
        
    def test_swarm_initialization(self):
        """Test quantum swarm initialization."""
        self.optimizer.initialize_quantum_state()
        
        self.assertEqual(len(self.optimizer.quantum_particles), 8)
        self.assertEqual(len(self.optimizer.personal_best), 8)
        self.assertIsNotNone(self.optimizer.global_best)
        
    def test_particle_updates(self):
        """Test particle position updates."""
        self.optimizer.initialize_quantum_state()
        self.optimizer.hamiltonian = self.optimizer.construct_hamiltonian(self.objective_function)
        
        initial_particles = [p.amplitudes.copy() for p in self.optimizer.quantum_particles]
        
        # Perform optimization step
        energy = self.optimizer.optimize_step(self.objective_function)
        
        # Particles should have been updated
        for i, particle in enumerate(self.optimizer.quantum_particles):
            self.assertFalse(np.allclose(initial_particles[i], particle.amplitudes))
            
        self.assertTrue(np.isfinite(energy))


class TestQuantumOptimizerFactory(unittest.TestCase):
    """Test suite for quantum optimizer factory."""
    
    def test_create_quantum_optimizer(self):
        """Test quantum optimizer creation."""
        n_qubits = 3
        
        optimizer_types = [
            'quantum_annealing',
            'qaoa_inspired',
            'quantum_walk',
            'quantum_neural',
            'quantum_evolutionary',
            'quantum_swarm'
        ]
        
        for opt_type in optimizer_types:
            optimizer = create_quantum_optimizer(opt_type, n_qubits, max_iterations=10)
            
            self.assertEqual(optimizer.n_qubits, n_qubits)
            self.assertEqual(optimizer.max_iterations, 10)
            self.assertFalse(optimizer.is_fitted)
            
    def test_invalid_optimizer_type(self):
        """Test invalid optimizer type handling."""
        with self.assertRaises(ValueError):
            create_quantum_optimizer('invalid_type', 3)


class TestQuantumOptimizationIntegration(unittest.TestCase):
    """Integration tests for quantum optimization algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_qubits = 3
        self.max_iterations = 20
        
        # Test functions
        self.test_functions = {
            'sphere': lambda x: np.sum(x**2),
            'rastrigin': lambda x: np.sum(x**2 - 10 * np.cos(2 * np.pi * x)),
            'rosenbrock': lambda x: np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        }
        
    def test_optimizer_convergence(self):
        """Test that optimizers converge to reasonable solutions."""
        optimizer = QuantumAnnealingOptimizer(
            n_qubits=self.n_qubits,
            max_iterations=self.max_iterations,
            random_state=42
        )
        
        results = optimizer.optimize(self.test_functions['sphere'])
        
        # Should find a reasonable solution for sphere function
        self.assertLess(results['best_energy'], 5.0)  # Not too high
        self.assertGreater(results['iterations'], 5)   # Some iterations occurred
        
    def test_multiple_optimizers_comparison(self):
        """Test comparison of multiple optimizers."""
        optimizer_types = ['quantum_annealing', 'qaoa_inspired', 'quantum_walk']
        results = {}
        
        for opt_type in optimizer_types:
            optimizer = create_quantum_optimizer(
                opt_type, self.n_qubits, 
                max_iterations=15, random_state=42
            )
            
            try:
                opt_results = optimizer.optimize(self.test_functions['sphere'])
                results[opt_type] = opt_results['best_energy']
            except Exception as e:
                results[opt_type] = float('inf')
                
        # All should find reasonable solutions
        for opt_type, energy in results.items():
            self.assertLess(energy, 10.0, f"{opt_type} failed to find good solution")
            
    def test_different_objective_functions(self):
        """Test optimizers on different objective functions."""
        optimizer = QuantumAnnealingOptimizer(
            n_qubits=self.n_qubits,
            max_iterations=15,
            random_state=42
        )
        
        for func_name, func in self.test_functions.items():
            try:
                results = optimizer.optimize(func)
                self.assertIn('best_energy', results)
                self.assertTrue(np.isfinite(results['best_energy']))
            except Exception as e:
                self.fail(f"Optimizer failed on {func_name}: {e}")


class TestQuantumOptimizationPerformance(unittest.TestCase):
    """Performance tests for quantum optimization algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_qubits = 4
        self.objective_function = lambda x: np.sum(x**2)
        
    def test_optimization_speed(self):
        """Test optimization speed."""
        optimizer = QuantumAnnealingOptimizer(
            n_qubits=self.n_qubits,
            max_iterations=50,
            random_state=42
        )
        
        start_time = time.time()
        results = optimizer.optimize(self.objective_function)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(execution_time, 30.0)  # 30 seconds max
        self.assertTrue(results['iterations'] > 0)
        
    def test_memory_usage(self):
        """Test memory usage patterns."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and run optimizer
        optimizer = QuantumAnnealingOptimizer(
            n_qubits=self.n_qubits,
            max_iterations=30,
            random_state=42
        )
        
        results = optimizer.optimize(self.objective_function)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024**2)  # MB
        
        # Memory increase should be reasonable
        self.assertLess(memory_increase, 100)  # Less than 100MB increase
        
    def test_scalability_with_qubits(self):
        """Test scalability with number of qubits."""
        qubit_counts = [2, 3, 4]
        execution_times = []
        
        for n_qubits in qubit_counts:
            optimizer = QuantumAnnealingOptimizer(
                n_qubits=n_qubits,
                max_iterations=20,
                random_state=42
            )
            
            start_time = time.time()
            results = optimizer.optimize(self.objective_function)
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
            
        # Time should increase with qubits (but not exponentially)
        self.assertLess(execution_times[-1], execution_times[0] * 10)


class TestQuantumOptimizationRobustness(unittest.TestCase):
    """Robustness tests for quantum optimization algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_qubits = 3
        
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        optimizer = QuantumAnnealingOptimizer(
            n_qubits=self.n_qubits,
            max_iterations=5,
            random_state=42
        )
        
        # Test with constant function
        constant_function = lambda x: 5.0
        results = optimizer.optimize(constant_function)
        
        self.assertEqual(results['best_energy'], 5.0)
        
        # Test with function that returns very large values
        large_function = lambda x: 1e10 * np.sum(x**2)
        results = optimizer.optimize(large_function)
        
        self.assertTrue(np.isfinite(results['best_energy']))
        
    def test_random_state_reproducibility(self):
        """Test reproducibility with random state."""
        objective_function = lambda x: np.sum(x**2)
        
        # Create two optimizers with same random state
        optimizer1 = QuantumAnnealingOptimizer(
            n_qubits=self.n_qubits,
            max_iterations=10,
            random_state=42
        )
        
        optimizer2 = QuantumAnnealingOptimizer(
            n_qubits=self.n_qubits,
            max_iterations=10,
            random_state=42
        )
        
        results1 = optimizer1.optimize(objective_function)
        results2 = optimizer2.optimize(objective_function)
        
        # Results should be identical
        self.assertEqual(results1['best_energy'], results2['best_energy'])
        
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        optimizer = QuantumAnnealingOptimizer(
            n_qubits=self.n_qubits,
            max_iterations=5,
            random_state=42
        )
        
        # Test with function that raises exception
        def bad_function(x):
            raise ValueError("Test error")
            
        # Should handle gracefully
        with self.assertRaises(Exception):
            optimizer.optimize(bad_function)


def run_quantum_optimization_tests():
    """Run comprehensive test suite for quantum optimization."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestQuantumAnnealingOptimizer,
        TestQAOAInspiredOptimizer,
        TestQuantumWalkOptimizer,
        TestQuantumNeuralOptimizer,
        TestQuantumEvolutionaryOptimizer,
        TestQuantumSwarmOptimizer,
        TestQuantumOptimizerFactory,
        TestQuantumOptimizationIntegration,
        TestQuantumOptimizationPerformance,
        TestQuantumOptimizationRobustness
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
    with open('quantum_optimization_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\n{'='*60}")
    print("QUANTUM OPTIMIZATION TEST REPORT")
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
    
    print("Quantum Optimization Algorithms - Comprehensive Test Suite")
    print("=" * 70)
    print("Testing all quantum optimization algorithms...")
    print()
    
    # Run comprehensive tests
    success = run_quantum_optimization_tests()
    
    if success:
        print("\n🎉 ALL QUANTUM OPTIMIZATION TESTS PASSED!")
    else:
        print("\n⚠️  Some tests failed. Please review the output above for details.")
        
    print("\nQuantum optimization testing completed.")
