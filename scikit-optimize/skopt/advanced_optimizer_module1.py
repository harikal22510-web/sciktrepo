"""
Advanced Bayesian Optimization Framework - Module 1
Enterprise-level optimization algorithms with adaptive strategies.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class AdaptiveBayesianOptimizer(BaseEstimator, RegressorMixin):
    """
    Advanced Bayesian optimizer with adaptive acquisition function selection.
    
    This optimizer automatically adapts its strategy based on optimization progress,
    switching between exploration and exploitation phases as needed.
    """
    
    def __init__(self, dimensions, base_estimator=None, 
                 n_initial_points=10, random_state=None,
                 adaptation_strategy="dynamic", 
                 convergence_threshold=1e-6):
        """
        Initialize adaptive Bayesian optimizer.
        
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
        self.random_state = random_state
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
    
    def _expected_improvement(self, X, model, y_opt):
        """Calculate expected improvement acquisition function."""
        mu, sigma = model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        with np.errstate(divide='warn'):
            imp = mu - y_opt - 0.01
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei.ravel()
    
    def _probability_of_improvement(self, X, model, y_opt):
        """Calculate probability of improvement acquisition function."""
        mu, sigma = model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        with np.errstate(divide='warn'):
            Z = (mu - y_opt) / sigma
            pi = norm.cdf(Z)
        
        return pi.ravel()
    
    def _lower_confidence_bound(self, X, model, kappa=2.576):
        """Calculate lower confidence bound acquisition function."""
        mu, sigma = model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        lcb = mu - kappa * sigma
        return lcb.ravel()
    
    def minimize(self, func, n_calls=100):
        """
        Perform adaptive Bayesian optimization.
        
        Parameters
        ----------
        func : callable
            Objective function to minimize.
        n_calls : int
            Maximum number of function evaluations.
        
        Returns
        -------
        result : dict
            Optimization result.
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
            current_acq = self._adaptive_acquisition_selection()
            
            # Update optimizer acquisition function
            opt.acq_func = current_acq
            
            # Perform one optimization step
            if iteration == 0:
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
        return {
            'x': X[np.argmin(y)],
            'fun': min(y),
            'x_iters': np.array(X),
            'func_vals': np.array(y),
            'nit': len(X),
            'success': True
        }


class MultiObjectiveOptimizer(BaseEstimator, RegressorMixin):
    """
    Multi-objective optimizer handling multiple conflicting objectives.
    
    This optimizer uses Pareto dominance concepts to handle multiple objectives
    simultaneously.
    """
    
    def __init__(self, dimensions, objectives=None, 
                 weighting_method='pareto', n_objectives=2):
        """
        Initialize multi-objective optimizer.
        
        Parameters
        ----------
        dimensions : list
            List of search space dimensions.
        objectives : list
            List of objective functions.
        weighting_method : str
            Method for combining objectives: 'pareto', 'weighted', 'nsga2'.
        n_objectives : int
            Number of objectives.
        """
        self.dimensions = dimensions
        self.objectives = objectives or []
        self.weighting_method = weighting_method
        self.n_objectives = n_objectives
        self.pareto_front = []
        
    def _is_dominated(self, point1, point2):
        """Check if point1 is dominated by point2."""
        return np.all(point2 <= point1) and np.any(point2 < point1)
    
    def _find_pareto_front(self, points):
        """Find Pareto front from a set of points."""
        pareto_front = []
        
        for i, point in enumerate(points):
            dominated = False
            for j, other in enumerate(points):
                if i != j and self._is_dominated(point, other):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(point)
        
        return np.array(pareto_front)
    
    def _weighted_sum_objective(self, x):
        """Calculate weighted sum of objectives."""
        weights = np.ones(self.n_objectives) / self.n_objectives
        total = 0.0
        
        for i, obj_func in enumerate(self.objectives):
            total += weights[i] * obj_func(x)
        
        return total
    
    def minimize(self, func, n_calls=100):
        """
        Perform multi-objective optimization.
        
        Parameters
        ----------
        func : callable
            Multi-objective function returning array of values.
        n_calls : int
            Maximum number of function evaluations.
        
        Returns
        -------
        result : dict
            Optimization result with Pareto front.
        """
        from ..optimizer import Optimizer
        
        # Create wrapper function for single-objective optimization
        if self.weighting_method == 'weighted':
            wrapper_func = self._weighted_sum_objective
        else:
            # Use first objective for exploration
            wrapper_func = lambda x: func(x)[0]
        
        opt = Optimizer(
            dimensions=self.dimensions,
            base_estimator=GaussianProcessRegressor(),
            n_initial_points=20,
            random_state=42
        )
        
        X = []
        y_multi = []  # Store all objective values
        
        for iteration in range(n_calls):
            next_x = opt.ask()
            next_y = func(next_x[0])
            
            X.extend(next_x)
            y_multi.append(next_y)
            opt.tell(next_x, wrapper_func(next_x[0]))
        
        # Find Pareto front
        self.pareto_front = self._find_pareto_front(np.array(y_multi))
        
        # Select best solution from Pareto front
        if len(self.pareto_front) > 0:
            # Use knee point selection (simplified)
            distances = np.sum(self.pareto_front**2, axis=1)
            best_idx = np.argmin(distances)
            best_solution = self.pareto_front[best_idx]
        else:
            best_solution = y_multi[np.argmin([y[0] for y in y_multi])]
        
        return {
            'x': X[np.argmin([y[0] for y in y_multi])],
            'fun': best_solution,
            'pareto_front': self.pareto_front,
            'x_iters': np.array(X),
            'func_vals': np.array(y_multi),
            'nit': len(X),
            'success': True
        }


class ConstrainedOptimizer(BaseEstimator, RegressorMixin):
    """
    Constrained optimizer handling explicit and implicit constraints.
    
    Supports inequality constraints, equality constraints, and domain constraints.
    """
    
    def __init__(self, dimensions, constraints=None, 
                 penalty_method='quadratic', penalty_factor=1000):
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
        penalty_factor : float
            Factor for constraint penalty.
        """
        self.dimensions = dimensions
        self.constraints = constraints or []
        self.penalty_method = penalty_method
        self.penalty_factor = penalty_factor
        
    def _constraint_penalty(self, x):
        """Calculate penalty for constraint violations."""
        total_penalty = 0.0
        
        for constraint in self.constraints:
            violation = constraint(x)
            
            if violation > 0:
                if self.penalty_method == "quadratic":
                    total_penalty += self.penalty_factor * violation ** 2
                elif self.penalty_method == "linear":
                    total_penalty += self.penalty_factor * violation
                elif self.penalty_method == "exponential":
                    total_penalty += self.penalty_factor * (np.exp(violation) - 1)
        
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
        result : dict
            Optimization result with constraint handling.
        """
        from ..optimizer import Optimizer
        
        opt = Optimizer(
            dimensions=self.dimensions,
            base_estimator=GaussianProcessRegressor(),
            n_initial_points=20,
            random_state=42
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
        
        return {
            'x': X[best_idx],
            'fun': y[best_idx],
            'x_iters': np.array(X),
            'func_vals': np.array(y),
            'nit': len(X),
            'success': True,
            'constraints_satisfied': self._constraint_penalty(X[best_idx]) == 0
        }


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
    from scipy.stats import qmc
    
    rng = check_random_state(random_state)
    sampler = qmc.LatinHypercube(d=n_dimensions, seed=rng.randint(0, 10000))
    samples = sampler.random(n_samples)
    
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
    from scipy.stats import qmc
    
    rng = check_random_state(random_state)
    sampler = qmc.Sobol(d=n_dimensions, seed=rng.randint(0, 10000))
    
    points = sampler.random_base2(n_points)
    
    return points


def check_random_state(random_state):
    """Validate and convert random state."""
    if random_state is None:
        return np.random.RandomState()
    elif isinstance(random_state, int):
        return np.random.RandomState(random_state)
    elif hasattr(random_state, 'randint'):
        return random_state
    else:
        raise ValueError("Invalid random state")
