"""
Ultra-Advanced Acquisition Functions for Bayesian Optimization

This module contains cutting-edge acquisition functions that go beyond
standard approaches, incorporating entropy-based methods, multi-fidelity
optimization, and adaptive strategies for complex optimization scenarios.

Key Features:
- Entropy-based acquisition functions
- Multi-fidelity acquisition strategies
- Adaptive acquisition selection
- Knowledge gradient extensions
- Thompson sampling variants
- Max-value entropy search implementations
- Batch acquisition functions
- Constrained optimization acquisition
"""

import numpy as np
import scipy
from scipy import stats, optimize
from scipy.special import erf, erfc, gamma, digamma
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mutual_info_score
import warnings
from typing import Optional, Dict, List, Tuple, Union, Callable
from abc import ABC, abstractmethod


class BaseUltraAcquisition(ABC):
    """Abstract base class for ultra-advanced acquisition functions."""
    
    def __init__(self, exploration_weight=1.0, random_state=None):
        """
        Initialize ultra acquisition function.
        
        Parameters
        ----------
        exploration_weight : float, default=1.0
            Weight for exploration vs exploitation
        random_state : int or RandomState, optional
            Random state for reproducibility
        """
        self.exploration_weight = exploration_weight
        self.random_state = random_state
        self.history = []
        
    @abstractmethod
    def evaluate(self, X, model, y_opt=None):
        """
        Evaluate acquisition function at points X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Points at which to evaluate acquisition function
        model : object
            Fitted surrogate model
        y_opt : float, optional
            Current best objective value
            
        Returns
        -------
        values : array-like, shape (n_samples,)
            Acquisition function values at X
        """
        pass


class EntropySearchAcquisition(BaseUltraAcquisition):
    """
    Entropy Search acquisition function that maximizes information gain
    about the global optimum location.
    """
    
    def __init__(self, n_samples=100, exploration_weight=1.0, random_state=None):
        """
        Initialize Entropy Search acquisition.
        
        Parameters
        ----------
        n_samples : int, default=100
            Number of samples for approximation
        exploration_weight : float, default=1.0
            Exploration weight
        random_state : int or RandomState, optional
            Random state
        """
        super().__init__(exploration_weight, random_state)
        self.n_samples = n_samples
        
    def evaluate(self, X, model, y_opt=None):
        """Evaluate entropy search acquisition."""
        # Get predictive statistics
        y_mean, y_std = model.predict(X, return_std=True)
        
        # Sample from posterior
        rng = np.random.RandomState(self.random_state)
        n_samples = min(self.n_samples, len(X))
        
        # Approximate entropy reduction
        entropy_values = np.zeros(len(X))
        
        for i in range(len(X)):
            # Local entropy estimation
            local_samples = rng.normal(y_mean[i], y_std[i], self.n_samples)
            
            # Estimate entropy of optimum location
            p_opt = stats.norm.cdf(local_samples, y_mean[i], y_std[i])
            entropy = -np.sum(p_opt * np.log(p_opt + 1e-10))
            
            entropy_values[i] = entropy * self.exploration_weight
            
        return entropy_values


class MultiFidelityAcquisition(BaseUltraAcquisition):
    """
    Multi-fidelity acquisition function for optimization with
    different levels of computational cost and accuracy.
    """
    
    def __init__(self, cost_ratio=0.1, fidelity_weight=0.5, 
                 exploration_weight=1.0, random_state=None):
        """
        Initialize multi-fidelity acquisition.
        
        Parameters
        ----------
        cost_ratio : float, default=0.1
            Ratio of low-fidelity to high-fidelity cost
        fidelity_weight : float, default=0.5
            Weight for fidelity in acquisition
        exploration_weight : float, default=1.0
            Exploration weight
        random_state : int or RandomState, optional
            Random state
        """
        super().__init__(exploration_weight, random_state)
        self.cost_ratio = cost_ratio
        self.fidelity_weight = fidelity_weight
        
    def evaluate(self, X, model, y_opt=None, fidelity=None):
        """
        Evaluate multi-fidelity acquisition.
        
        Parameters
        ----------
        X : array-like
            Points to evaluate
        model : object
            Surrogate model
        y_opt : float, optional
            Best objective value
        fidelity : array-like, optional
            Fidelity levels for points
            
        Returns
        -------
        values : array-like
            Acquisition values
        """
        y_mean, y_std = model.predict(X, return_std=True)
        
        if y_opt is None:
            y_opt = np.min(y_mean)
            
        # Standard expected improvement
        improvement = y_opt - y_mean
        z = improvement / (y_std + 1e-10)
        ei = improvement * stats.norm.cdf(z) + y_std * stats.norm.pdf(z)
        
        # Multi-fidelity adjustment
        if fidelity is not None:
            fidelity_bonus = self.fidelity_weight * fidelity
            cost_penalty = np.where(fidelity < 1.0, 
                                   self.cost_ratio * (1 - fidelity), 
                                   1.0)
            values = (ei + fidelity_bonus) / cost_penalty
        else:
            values = ei
            
        return values * self.exploration_weight


class KnowledgeGradientPlus(BaseUltraAcquisition):
    """
    Enhanced Knowledge Gradient acquisition that considers the value
    of information for future decision making.
    """
    
    def __init__(self, n_lookahead=1, discount_factor=0.9, 
                 exploration_weight=1.0, random_state=None):
        """
        Initialize Knowledge Gradient Plus.
        
        Parameters
        ----------
        n_lookahead : int, default=1
            Number of lookahead steps
        discount_factor : float, default=0.9
            Discount factor for future rewards
        exploration_weight : float, default=1.0
            Exploration weight
        random_state : int or RandomState, optional
            Random state
        """
        super().__init__(exploration_weight, random_state)
        self.n_lookahead = n_lookahead
        self.discount_factor = discount_factor
        
    def evaluate(self, X, model, y_opt=None):
        """Evaluate knowledge gradient plus acquisition."""
        y_mean, y_std = model.predict(X, return_std=True)
        
        if y_opt is None:
            y_opt = np.min(y_mean)
            
        kg_values = np.zeros(len(X))
        
        for i, x in enumerate(X):
            # Sample possible outcomes
            rng = np.random.RandomState(self.random_state)
            n_samples = 50
            
            # Monte Carlo estimation of knowledge gradient
            kg_sum = 0.0
            
            for _ in range(n_samples):
                # Sample observation
                y_sample = rng.normal(y_mean[i], y_std[i])
                
                # Simulate future optimization
                future_value = self._simulate_future_optimization(
                    x, y_sample, model, y_opt
                )
                
                kg_sum += max(0, future_value - y_opt)
                
            kg_values[i] = kg_sum / n_samples
            
        return kg_values * self.exploration_weight
    
    def _simulate_future_optimization(self, x, y, model, y_opt):
        """Simulate future optimization given an observation."""
        # Simplified simulation - in practice would update model
        # and re-optimize
        improvement = y_opt - y
        return y_opt - improvement * self.discount_factor


class ThompsonSamplingAdvanced(BaseUltraAcquisition):
    """
    Advanced Thompson Sampling with multiple posterior samples
    and uncertainty quantification.
    """
    
    def __init__(self, n_samples=10, exploration_weight=1.0, random_state=None):
        """
        Initialize advanced Thompson Sampling.
        
        Parameters
        ----------
        n_samples : int, default=10
            Number of posterior samples
        exploration_weight : float, default=1.0
            Exploration weight
        random_state : int or RandomState, optional
            Random state
        """
        super().__init__(exploration_weight, random_state)
        self.n_samples = n_samples
        
    def evaluate(self, X, model, y_opt=None):
        """Evaluate Thompson sampling acquisition."""
        y_mean, y_std = model.predict(X, return_std=True)
        
        rng = np.random.RandomState(self.random_state)
        
        # Sample multiple posterior functions
        samples = np.zeros((self.n_samples, len(X)))
        for i in range(self.n_samples):
            samples[i] = rng.normal(y_mean, y_std)
            
        # Thompson sampling: select min of each sample
        ts_values = np.min(samples, axis=0)
        
        # Add uncertainty quantification
        uncertainty = np.std(samples, axis=0)
        
        return (ts_values + self.exploration_weight * uncertainty)


class MaxValueEntropySearch(BaseUltraAcquisition):
    """
    Max-Value Entropy Search that focuses on information about
    the maximum function value.
    """
    
    def __init__(self, n_samples=100, exploration_weight=1.0, random_state=None):
        """
        Initialize Max-Value Entropy Search.
        
        Parameters
        ----------
        n_samples : int, default=100
            Number of samples for approximation
        exploration_weight : float, default=1.0
            Exploration weight
        random_state : int or RandomState, optional
            Random state
        """
        super().__init__(exploration_weight, random_state)
        self.n_samples = n_samples
        
    def evaluate(self, X, model, y_opt=None):
        """Evaluate max-value entropy search acquisition."""
        y_mean, y_std = model.predict(X, return_std=True)
        
        if y_opt is None:
            y_opt = np.min(y_mean)
            
        rng = np.random.RandomState(self.random_state)
        
        mves_values = np.zeros(len(X))
        
        for i in range(len(X)):
            # Sample from posterior at current point
            posterior_samples = rng.normal(y_mean[i], y_std[i], self.n_samples)
            
            # Estimate entropy of maximum value
            max_values = np.maximum(posterior_samples, y_opt)
            
            # Compute differential entropy
            entropy = self._compute_differential_entropy(max_values)
            
            mves_values[i] = entropy * self.exploration_weight
            
        return mves_values
    
    def _compute_differential_entropy(self, samples):
        """Compute differential entropy of samples."""
        # Kernel density estimation
        kde = stats.gaussian_kde(samples)
        
        # Approximate entropy
        x_range = np.linspace(np.min(samples), np.max(samples), 100)
        pdf_values = kde(x_range)
        
        # Numerical integration
        entropy = -np.trapz(pdf_values * np.log(pdf_values + 1e-10), x_range)
        
        return entropy


class BatchAcquisitionFunction(BaseUltraAcquisition):
    """
    Batch acquisition function for evaluating multiple points simultaneously.
    """
    
    def __init__(self, batch_size=5, diversity_weight=0.1, 
                 exploration_weight=1.0, random_state=None):
        """
        Initialize batch acquisition function.
        
        Parameters
        ----------
        batch_size : int, default=5
            Size of batches to evaluate
        diversity_weight : float, default=0.1
            Weight for encouraging diversity in batches
        exploration_weight : float, default=1.0
            Exploration weight
        random_state : int or RandomState, optional
            Random state
        """
        super().__init__(exploration_weight, random_state)
        self.batch_size = batch_size
        self.diversity_weight = diversity_weight
        
    def evaluate(self, X, model, y_opt=None):
        """Evaluate batch acquisition function."""
        y_mean, y_std = model.predict(X, return_std=True)
        
        if y_opt is None:
            y_opt = np.min(y_mean)
            
        # Standard acquisition (e.g., Expected Improvement)
        improvement = y_opt - y_mean
        z = improvement / (y_std + 1e-10)
        ei = improvement * stats.norm.cdf(z) + y_std * stats.norm.pdf(z)
        
        # Diversity penalty for batch selection
        diversity_penalty = self._compute_diversity_penalty(X)
        
        # Combined acquisition
        batch_values = ei - self.diversity_weight * diversity_penalty
        
        return batch_values * self.exploration_weight
    
    def _compute_diversity_penalty(self, X):
        """Compute diversity penalty for points in X."""
        n_points = len(X)
        diversity = np.zeros(n_points)
        
        for i in range(n_points):
            # Compute average distance to other points
            distances = np.linalg.norm(X - X[i], axis=1)
            diversity[i] = np.mean(distances[distances > 0])
            
        # Normalize
        if np.max(diversity) > 0:
            diversity = diversity / np.max(diversity)
            
        return diversity


class ConstrainedAcquisitionFunction(BaseUltraAcquisition):
    """
    Constrained acquisition function for optimization with constraints.
    """
    
    def __init__(self, constraint_weight=10.0, exploration_weight=1.0, random_state=None):
        """
        Initialize constrained acquisition function.
        
        Parameters
        ----------
        constraint_weight : float, default=10.0
            Weight for constraint violations
        exploration_weight : float, default=1.0
            Exploration weight
        random_state : int or RandomState, optional
            Random state
        """
        super().__init__(exploration_weight, random_state)
        self.constraint_weight = constraint_weight
        
    def evaluate(self, X, model, y_opt=None, constraint_models=None):
        """
        Evaluate constrained acquisition function.
        
        Parameters
        ----------
        X : array-like
            Points to evaluate
        model : object
            Objective surrogate model
        y_opt : float, optional
            Best objective value
        constraint_models : list, optional
            List of constraint surrogate models
            
        Returns
        -------
        values : array-like
            Constrained acquisition values
        """
        y_mean, y_std = model.predict(X, return_std=True)
        
        if y_opt is None:
            y_opt = np.min(y_mean)
            
        # Standard expected improvement
        improvement = y_opt - y_mean
        z = improvement / (y_std + 1e-10)
        ei = improvement * stats.norm.cdf(z) + y_std * stats.norm.pdf(z)
        
        # Constraint probability
        if constraint_models is not None:
            constraint_prob = np.ones(len(X))
            
            for constraint_model in constraint_models:
                c_mean, c_std = constraint_model.predict(X, return_std=True)
                # Probability of satisfying constraint (assuming constraint <= 0)
                prob_satisfy = stats.norm.cdf(-c_mean / (c_std + 1e-10))
                constraint_prob *= prob_satisfy
                
            # Penalize infeasible points
            constrained_ei = ei * constraint_prob
            
            # Add penalty for expected constraint violation
            expected_violation = 0.0
            for constraint_model in constraint_models:
                c_mean, c_std = constraint_model.predict(X, return_std=True)
                violation = np.maximum(0, c_mean)
                expected_violation += violation
                
            constrained_values = constrained_ei - self.constraint_weight * expected_violation
        else:
            constrained_values = ei
            
        return constrained_values * self.exploration_weight


class AdaptiveAcquisitionSelector:
    """
    Adaptive selector that chooses the best acquisition function
    based on optimization progress and problem characteristics.
    """
    
    def __init__(self, acquisition_pool=None, adaptation_strategy="dynamic",
                 random_state=None):
        """
        Initialize adaptive acquisition selector.
        
        Parameters
        ----------
        acquisition_pool : list, optional
            Pool of acquisition functions to choose from
        adaptation_strategy : str, default="dynamic"
            Strategy for adaptation: "dynamic", "performance_based", "ensemble"
        random_state : int or RandomState, optional
            Random state
        """
        self.adaptation_strategy = adaptation_strategy
        self.random_state = random_state
        
        if acquisition_pool is None:
            self.acquisition_pool = [
                EntropySearchAcquisition(random_state=random_state),
                MultiFidelityAcquisition(random_state=random_state),
                KnowledgeGradientPlus(random_state=random_state),
                ThompsonSamplingAdvanced(random_state=random_state),
                MaxValueEntropySearch(random_state=random_state)
            ]
        else:
            self.acquisition_pool = acquisition_pool
            
        self.performance_history = []
        self.current_acquisition = 0
        
    def select_acquisition(self, iteration, X, model, y_opt=None):
        """
        Select the best acquisition function for current iteration.
        
        Parameters
        ----------
        iteration : int
            Current optimization iteration
        X : array-like
            Points to evaluate
        model : object
            Surrogate model
        y_opt : float, optional
            Best objective value
            
        Returns
        -------
        acquisition : BaseUltraAcquisition
            Selected acquisition function
        """
        if self.adaptation_strategy == "dynamic":
            return self._dynamic_selection(iteration)
        elif self.adaptation_strategy == "performance_based":
            return self._performance_based_selection(iteration, X, model, y_opt)
        elif self.adaptation_strategy == "ensemble":
            return self._ensemble_selection(X, model, y_opt)
        else:
            return self.acquisition_pool[0]
    
    def _dynamic_selection(self, iteration):
        """Dynamic selection based on iteration count."""
        # Cycle through acquisition functions
        idx = iteration % len(self.acquisition_pool)
        return self.acquisition_pool[idx]
    
    def _performance_based_selection(self, iteration, X, model, y_opt):
        """Performance-based selection using historical performance."""
        if len(self.performance_history) < 5:
            # Use default selection initially
            return self._dynamic_selection(iteration)
            
        # Evaluate all acquisition functions
        performances = []
        for acq in self.acquisition_pool:
            values = acq.evaluate(X, model, y_opt)
            performance = np.mean(values)
            performances.append(performance)
            
        # Select best performing
        best_idx = np.argmax(performances)
        return self.acquisition_pool[best_idx]
    
    def _ensemble_selection(self, X, model, y_opt):
        """Ensemble selection combining multiple acquisition functions."""
        # Combine acquisition values from all functions
        combined_values = np.zeros(len(X))
        
        for acq in self.acquisition_pool:
            values = acq.evaluate(X, model, y_opt)
            combined_values += values
            
        combined_values /= len(self.acquisition_pool)
        
        # Return a wrapper that uses combined values
        class EnsembleAcquisition(BaseUltraAcquisition):
            def evaluate(self, X, model, y_opt=None):
                return combined_values
                
        return EnsembleAcquisition(random_state=self.random_state)


# Utility functions for acquisition function optimization
def optimize_acquisition(acquisition, model, bounds, n_restarts=10, 
                        random_state=None):
    """
    Optimize acquisition function to find next evaluation point.
    
    Parameters
    ----------
    acquisition : BaseUltraAcquisition
        Acquisition function to optimize
    model : object
        Surrogate model
    bounds : array-like, shape (n_features, 2)
        Bounds for optimization
    n_restarts : int, default=10
        Number of random restarts
    random_state : int or RandomState, optional
        Random state
        
    Returns
    -------
    x_opt : array-like, shape (n_features,)
        Optimal point
    f_opt : float
        Optimal acquisition value
    """
    rng = np.random.RandomState(random_state)
    n_features = bounds.shape[0]
    
    best_x = None
    best_f = -np.inf
    
    for _ in range(n_restarts):
        # Random starting point
        x0 = rng.uniform(bounds[:, 0], bounds[:, 1])
        
        # Optimize
        result = optimize.minimize(
            lambda x: -acquisition.evaluate(x.reshape(1, -1), model)[0],
            x0, bounds=bounds, method='L-BFGS-B'
        )
        
        if result.fun < best_f:
            best_f = result.fun
            best_x = result.x
            
    return best_x, -best_f


def evaluate_acquisition_robustness(acquisition, model, X_test, 
                                     noise_levels=[0.01, 0.05, 0.1]):
    """
    Evaluate robustness of acquisition function to noise.
    
    Parameters
    ----------
    acquisition : BaseUltraAcquisition
        Acquisition function to test
    model : object
        Surrogate model
    X_test : array-like
        Test points
    noise_levels : list, default=[0.01, 0.05, 0.1]
        Noise levels to test
        
    Returns
    -------
    robustness_scores : dict
        Robustness scores for each noise level
    """
    robustness_scores = {}
    
    # Clean evaluation
    clean_values = acquisition.evaluate(X_test, model)
    
    for noise in noise_levels:
        noisy_values = []
        
        for _ in range(10):  # Multiple trials
            # Add noise to model predictions
            y_mean, y_std = model.predict(X_test, return_std=True)
            noisy_mean = y_mean + np.random.normal(0, noise, y_mean.shape)
            
            # Create temporary noisy model
            class NoisyModel:
                def predict(self, X, return_std=False):
                    if return_std:
                        return noisy_mean, y_std
                    return noisy_mean
                    
            noisy_values.append(acquisition.evaluate(X_test, NoisyModel()))
            
        # Compute correlation with clean values
        correlations = []
        for vals in noisy_values:
            corr = np.corrcoef(clean_values, vals)[0, 1]
            correlations.append(corr)
            
        robustness_scores[noise] = np.mean(correlations)
        
    return robustness_scores


# Factory function for creating acquisition functions
def create_ultra_acquisition(name, **kwargs):
    """
    Factory function to create ultra-advanced acquisition functions.
    
    Parameters
    ----------
    name : str
        Name of acquisition function to create
    **kwargs : dict
        Additional parameters for acquisition function
        
    Returns
    -------
    acquisition : BaseUltraAcquisition
        Created acquisition function
    """
    acquisition_map = {
        'entropy_search': EntropySearchAcquisition,
        'multi_fidelity': MultiFidelityAcquisition,
        'knowledge_gradient_plus': KnowledgeGradientPlus,
        'thompson_sampling_advanced': ThompsonSamplingAdvanced,
        'max_value_entropy_search': MaxValueEntropySearch,
        'batch_acquisition': BatchAcquisitionFunction,
        'constrained_acquisition': ConstrainedAcquisitionFunction
    }
    
    if name not in acquisition_map:
        raise ValueError(f"Unknown acquisition function: {name}")
        
    return acquisition_map[name](**kwargs)


# Performance comparison utilities
class AcquisitionBenchmark:
    """
    Benchmark class for comparing acquisition functions.
    """
    
    def __init__(self, test_functions, n_runs=10, random_state=None):
        """
        Initialize acquisition benchmark.
        
        Parameters
        ----------
        test_functions : list
            List of test functions to evaluate
        n_runs : int, default=10
            Number of runs per function
        random_state : int or RandomState, optional
            Random state
        """
        self.test_functions = test_functions
        self.n_runs = n_runs
        self.random_state = random_state
        self.results = {}
        
    def benchmark_acquisitions(self, acquisitions):
        """
        Benchmark multiple acquisition functions.
        
        Parameters
        ----------
        acquisitions : list
            List of acquisition functions to benchmark
            
        Returns
        -------
        results : dict
            Benchmark results
        """
        results = {}
        
        for acq in acquisitions:
            acq_name = acq.__class__.__name__
            results[acq_name] = {}
            
            for test_func in self.test_functions:
                func_name = test_func.__name__
                results[acq_name][func_name] = self._benchmark_single(
                    acq, test_func
                )
                
        self.results = results
        return results
    
    def _benchmark_single(self, acquisition, test_function):
        """Benchmark single acquisition on single test function."""
        performances = []
        
        for run in range(self.n_runs):
            # Run optimization with this acquisition
            performance = self._run_optimization(acquisition, test_function)
            performances.append(performance)
            
        return {
            'mean': np.mean(performances),
            'std': np.std(performances),
            'min': np.min(performances),
            'max': np.max(performances)
        }
    
    def _run_optimization(self, acquisition, test_function, n_iterations=50):
        """Run single optimization trial."""
        # Simplified optimization loop
        rng = np.random.RandomState(self.random_state)
        
        # Initialize
        bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])  # 2D bounds
        X_init = rng.uniform(bounds[:, 0], bounds[:, 1], (5, 2))
        y_init = np.array([test_function(x) for x in X_init])
        
        # Fit initial model
        model = GaussianProcessRegressor()
        model.fit(X_init, y_init)
        
        # Optimization loop
        X = X_init.copy()
        y = y_init.copy()
        
        for _ in range(n_iterations - 5):
            # Select next point
            x_next, _ = optimize_acquisition(
                acquisition, model, bounds, random_state=rng
            )
            
            # Evaluate
            y_next = test_function(x_next)
            
            # Update data
            X = np.vstack([X, x_next.reshape(1, -1)])
            y = np.append(y, y_next)
            
            # Update model
            model.fit(X, y)
            
        # Return best found value
        return np.min(y)
    
    def generate_report(self):
        """Generate benchmark report."""
        if not self.results:
            return "No benchmark results available. Run benchmark first."
            
        report = "Acquisition Function Benchmark Report\n"
        report += "=" * 50 + "\n\n"
        
        for acq_name, acq_results in self.results.items():
            report += f"Acquisition: {acq_name}\n"
            report += "-" * 30 + "\n"
            
            for func_name, metrics in acq_results.items():
                report += f"  {func_name}:\n"
                report += f"    Mean: {metrics['mean']:.4f} ± {metrics['std']:.4f}\n"
                report += f"    Range: [{metrics['min']:.4f}, {metrics['max']:.4f}]\n"
                
            report += "\n"
            
        return report


# Example test functions for benchmarking
def sphere_function(x):
    """Simple sphere function for testing."""
    return np.sum(x**2)

def rastrigin_function(x):
    """Rastrigin function for testing."""
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock_function(x):
    """Rosenbrock function for testing."""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


if __name__ == "__main__":
    # Example usage
    print("Ultra-Advanced Acquisition Functions Module")
    print("=" * 50)
    
    # Create acquisition functions
    es = EntropySearchAcquisition(n_samples=50, random_state=42)
    mf = MultiFidelityAcquisition(random_state=42)
    kg = KnowledgeGradientPlus(random_state=42)
    ts = ThompsonSamplingAdvanced(random_state=42)
    mves = MaxValueEntropySearch(random_state=42)
    
    # Test on simple problem
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    
    # Create test data
    rng = np.random.RandomState(42)
    X_test = rng.uniform(-5, 5, (10, 2))
    y_test = np.sum(X_test**2, axis=1)
    
    # Fit model
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(X_test, y_test)
    
    # Evaluate acquisitions
    print("\nEvaluating acquisition functions:")
    acquisitions = [es, mf, kg, ts, mves]
    names = ["Entropy Search", "Multi-Fidelity", "Knowledge Gradient+", 
             "Thompson Sampling", "Max-Value Entropy Search"]
    
    for acq, name in zip(acquisitions, names):
        values = acq.evaluate(X_test, model)
        print(f"{name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
    
    # Benchmark example
    print("\nRunning benchmark...")
    benchmark = AcquisitionBenchmark(
        test_functions=[sphere_function, rastrigin_function],
        n_runs=3,
        random_state=42
    )
    
    results = benchmark.benchmark_acquisitions(acquisitions[:3])  # First 3 for speed
    print(benchmark.generate_report())
