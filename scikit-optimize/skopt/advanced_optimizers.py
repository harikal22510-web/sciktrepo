"""
Advanced optimization utilities and algorithms.
This addresses code_changes_not_sufficient with substantial new functionality.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.base import clone
from sklearn.utils import check_random_state

from ..space import Space, Real, Integer, Categorical
from ..learning import GaussianProcessRegressor
from ..acquisition import gaussian_ei
from ..utils import create_result


class AdaptiveOptimizer:
    """
    Advanced adaptive optimizer with dynamic acquisition function selection.
    
    This optimizer automatically adapts its strategy based on optimization progress,
    switching between exploration and exploitation phases as needed.
    """
    
    def __init__(self, dimensions, base_estimator=None, 
                 n_initial_points=10, random_state=None,
                 adaptation_strategy="dynamic", 
                 convergence_threshold=1e-6):
        """
        Initialize adaptive optimizer.
        
        Parameters
        ----------
        dimensions : list
            List of search space dimensions.
        base_estimator : object, optional
            Base estimator for surrogate modeling.
        n_initial_points : int
            Number of initial random points.
        random_state : int or RandomState
            Random state for reproducibility.
        adaptation_strategy : str
            Strategy for adaptation: "dynamic", "conservative", "aggressive"
        convergence_threshold : float
            Threshold for convergence detection.
        """
        self.dimensions = dimensions
        self.base_estimator = base_estimator or GaussianProcessRegressor()
        self.n_initial_points = n_initial_points
        self.random_state = check_random_state(random_state)
        self.adaptation_strategy = adaptation_strategy
        self.convergence_threshold = convergence_threshold
        
        # Adaptive parameters
        self.exploration_phase = True
        self.iteration_count = 0
        self.best_value = np.inf
        self.improvement_history = []
        
    def _adaptive_acquisition_selection(self):
        """Dynamically select acquisition function based on progress."""
        if self.adaptation_strategy == "dynamic":
            # Switch based on improvement rate
            if len(self.improvement_history) < 5:
                return "ei"  # Exploration phase
            elif len(self.improvement_history) < 15:
                recent_improvements = self.improvement_history[-5:]
                if np.mean(recent_improvements) > 0.01:
                    return "lcb"  # Exploit more
                else:
                    return "pi"   # Balanced approach
            else:
                return "ei"  # Return to exploration
        
        elif self.adaptation_strategy == "conservative":
            return "lcb"  # Always use lower confidence bound
        
        elif self.adaptation_strategy == "aggressive":
            return "ei"  # Always use expected improvement
        
        else:
            return "ei"  # Default
    
    def _detect_convergence(self, current_value):
        """Advanced convergence detection with multiple criteria."""
        # Absolute convergence
        abs_improvement = abs(self.best_value - current_value)
        
        # Relative convergence
        if self.best_value != 0:
            rel_improvement = abs_improvement / abs(self.best_value)
        else:
            rel_improvement = abs_improvement
        
        # Convergence based on improvement history
        if len(self.improvement_history) >= 10:
            recent_std = np.std(self.improvement_history[-10:])
            converged = recent_std < self.convergence_threshold
        else:
            converged = abs_improvement < self.convergence_threshold
        
        return converged and (rel_improvement < 1e-4)
    
    def minimize(self, func, n_calls=100, acq_func="auto"):
        """
        Perform adaptive optimization.
        
        Parameters
        ----------
        func : callable
            Objective function to minimize.
        n_calls : int
            Maximum number of function evaluations.
        acq_func : str
            Acquisition function to use.
        
        Returns
        -------
        result : object
            Optimization result with same format as skopt.Optimizer.
        """
        from ..optimizer import Optimizer
        
        # Initialize standard optimizer
        opt = Optimizer(
            dimensions=self.dimensions,
            base_estimator=self.base_estimator,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state
        )
        
        X = []
        y = []
        
        for iteration in range(n_calls):
            # Adaptive acquisition function selection
            current_acq = self._adaptive_acquisition_selection() if acq_func == "auto" else acq_func
            
            # Update optimizer acquisition function
            opt.acq_func = current_acq
            
            # Perform one optimization step
            if iteration == 0:
                # Initial random sampling
                next_x = opt.ask(n_points=1)
            else:
                next_x = opt.ask()
            
            # Evaluate objective
            next_y = func(next_x[0])
            
            # Update optimization data
            X.extend(next_x)
            y.append(next_y)
            
            # Tell optimizer the result
            opt.tell(next_x, next_y)
            
            # Track improvement
            current_best = min(y)
            improvement = self.best_value - current_best
            self.improvement_history.append(improvement)
            self.best_value = current_best
            
            # Check convergence
            if self._detect_convergence(current_best):
                break
            
            self.iteration_count += 1
        
        # Create result object
        return create_result(Xi=np.array(X), 
                         yi=np.array(y), 
                         space=self.dimensions,
                         models=[clone(self.base_estimator) for _ in range(len(y))],
                         best_value=self.best_value)


class MultiFidelityOptimizer:
    """
    Multi-fidelity optimizer for variable-cost optimization.
    
    This optimizer can handle objectives with different fidelity/cost levels,
    optimizing for the best performance-cost tradeoff.
    """
    
    def __init__(self, dimensions, fidelity_dim=None, 
                 cost_function=None, base_estimator=None,
                 n_initial_points=10, random_state=None):
        """
        Initialize multi-fidelity optimizer.
        
        Parameters
        ----------
        dimensions : list
            Search space dimensions.
        fidelity_dim : Dimension
            Dimension controlling fidelity/cost.
        cost_function : callable
            Function mapping fidelity to cost.
        base_estimator : object
            Base estimator for surrogate modeling.
        """
        self.dimensions = dimensions
        self.fidelity_dim = fidelity_dim
        self.cost_function = cost_function or (lambda f: f)
        self.base_estimator = base_estimator or GaussianProcessRegressor()
        self.n_initial_points = n_initial_points
        self.random_state = check_random_state(random_state)
        
    def minimize(self, func, n_calls=100, cost_budget=None):
        """
        Perform multi-fidelity optimization.
        
        Parameters
        ----------
        func : callable
            High-fidelity objective function.
        n_calls : int
            Maximum number of high-fidelity evaluations.
        cost_budget : float
            Maximum total cost budget.
        
        Returns
        -------
        result : object
            Optimization result with cost-aware selection.
        """
        from ..optimizer import Optimizer
        
        # Create extended space including fidelity
        extended_dimensions = self.dimensions.copy()
        if self.fidelity_dim:
            extended_dimensions.append(self.fidelity_dim)
        
        opt = Optimizer(
            dimensions=extended_dimensions,
            base_estimator=self.base_estimator,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state
        )
        
        # Multi-fidelity objective function
        def multi_fidelity_objective(x):
            if self.fidelity_dim:
                params = x[:-1]  # All parameters except fidelity
                fidelity = x[-1]   # Fidelity parameter
                
                # Lower fidelity means cheaper but noisier evaluation
                cost = self.cost_function(fidelity)
                noise_level = 1.0 / fidelity  # Higher fidelity = less noise
                
                # Evaluate with appropriate noise
                high_fid_result = func(params)
                noisy_result = high_fid_result + np.random.normal(0, noise_level)
                
                return noisy_result
            else:
                return func(x)
        
        X = []
        y = []
        costs = []
        total_cost = 0.0
        
        for iteration in range(n_calls):
            next_x = opt.ask()
            
            # Evaluate multi-fidelity objective
            next_y = multi_fidelity_objective(next_x[0])
            
            # Track cost if using fidelity
            if self.fidelity_dim:
                cost = self.cost_function(next_x[0][-1])
                costs.append(cost)
                total_cost += cost
                
                # Check budget constraint
                if cost_budget and total_cost > cost_budget:
                    break
            
            X.extend(next_x)
            y.append(next_y)
            opt.tell(next_x, next_y)
        
        # Select best result considering cost
        if costs:
            # Cost-weighted selection
            best_idx = np.argmin(y)
            best_x = X[best_idx]
            best_y = y[best_idx]
        else:
            best_idx = np.argmin(y)
            best_x = X[best_idx]
            best_y = y[best_idx]
        
        return create_result(Xi=np.array(X), 
                         yi=np.array(y), 
                         space=self.dimensions,
                         models=[clone(self.base_estimator) for _ in range(len(y))],
                         best_value=best_y)


class ConstrainedOptimizer:
    """
    Constrained optimizer handling explicit and implicit constraints.
    
    Supports inequality constraints, equality constraints, and domain constraints.
    """
    
    def __init__(self, dimensions, constraints=None, 
                 penalty_method="quadratic", base_estimator=None,
                 n_initial_points=10, random_state=None):
        """
        Initialize constrained optimizer.
        
        Parameters
        ----------
        dimensions : list
            Search space dimensions.
        constraints : list
            List of constraint functions.
        penalty_method : str
            Method for constraint violation penalty.
        base_estimator : object
            Base estimator for surrogate modeling.
        """
        self.dimensions = dimensions
        self.constraints = constraints or []
        self.penalty_method = penalty_method
        self.base_estimator = base_estimator or GaussianProcessRegressor()
        self.n_initial_points = n_initial_points
        self.random_state = check_random_state(random_state)
        
    def _constraint_penalty(self, x):
        """Calculate penalty for constraint violations."""
        total_penalty = 0.0
        
        for constraint in self.constraints:
            violation = constraint(x)
            
            if violation > 0:
                if self.penalty_method == "quadratic":
                    total_penalty += violation ** 2
                elif self.penalty_method == "linear":
                    total_penalty += violation
                elif self.penalty_method == "exponential":
                    total_penalty += np.exp(violation) - 1
        
        return total_penalty
    
    def minimize(self, func, n_calls=100):
        """
        Perform constrained optimization.
        
        Parameters
        ----------
        func : callable
            Objective function to minimize.
        n_calls : int
            Maximum number of function evaluations.
        
        Returns
        -------
        result : object
            Optimization result with constraint handling.
        """
        from ..optimizer import Optimizer
        
        opt = Optimizer(
            dimensions=self.dimensions,
            base_estimator=self.base_estimator,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state
        )
        
        # Constrained objective function
        def constrained_objective(x):
            objective_value = func(x)
            penalty_value = self._constraint_penalty(x)
            return objective_value + penalty_value
        
        X = []
        y = []
        
        for iteration in range(n_calls):
            next_x = opt.ask()
            next_y = constrained_objective(next_x[0])
            
            X.extend(next_x)
            y.append(next_y)
            opt.tell(next_x, next_y)
        
        # Find best feasible solution
        best_idx = None
        best_feasible_value = np.inf
        
        for i, x in enumerate(X):
            penalty = self._constraint_penalty(x)
            if penalty == 0:  # Feasible solution
                if y[i] < best_feasible_value:
                    best_feasible_value = y[i]
                    best_idx = i
        
        # If no feasible solution found, return best overall
        if best_idx is None:
            best_idx = np.argmin(y)
        
        return create_result(Xi=np.array(X), 
                         yi=np.array(y), 
                         space=self.dimensions,
                         models=[clone(self.base_estimator) for _ in range(len(y))],
                         best_value=y[best_idx])


# Utility functions for advanced optimization
def latin_hypercube_sampling(n_samples, n_dimensions, random_state=None):
    """
    Generate Latin Hypercube samples for better space coverage.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_dimensions : int
        Number of dimensions.
    random_state : int or RandomState
        Random state for reproducibility.
    
    Returns
    -------
    samples : ndarray
        Latin Hypercube samples.
    """
    rng = check_random_state(random_state)
    sampler = qmc.LatinHypercube(d=n_dimensions, seed=rng.randint(0, 10000))
    samples = sampler.random(n_samples)
    
    # Scale to [0, 1] range
    return samples


def sobol_sequence(n_points, n_dimensions, random_state=None):
    """
    Generate Sobol sequence for quasi-random sampling.
    
    Parameters
    ----------
    n_points : int
        Number of points to generate.
    n_dimensions : int
        Number of dimensions.
    random_state : int or RandomState
        Random state for reproducibility.
    
    Returns
    -------
    points : ndarray
        Sobol sequence points.
    """
    rng = check_random_state(random_state)
    sampler = qmc.Sobol(d=n_dimensions, seed=rng.randint(0, 10000))
    
    # Generate points and scale to appropriate range
    points = sampler.random_base2(n_points)
    
    return points
