"""
Enhanced acquisition functions with advanced optimization strategies.
This file provides sophisticated acquisition functions for complex optimization scenarios.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from ..learning.gaussian_process import gpr
from ..utils import check_random_state


class ExpectedImprovementPlus:
    """
    Enhanced Expected Improvement with adaptive exploration-exploitation balance.
    
    This acquisition function dynamically adjusts the exploration-exploitation tradeoff
    based on optimization progress and uncertainty estimates.
    """
    
    def __init__(self, exploration_factor=0.01, adaptive=True, random_state=None):
        """
        Initialize Enhanced Expected Improvement.
        
        Parameters
        ----------
        exploration_factor : float
            Base exploration factor for EI calculation.
        adaptive : bool
            Whether to use adaptive exploration factor.
        random_state : int or RandomState
            Random state for reproducibility.
        """
        self.exploration_factor = exploration_factor
        self.adaptive = adaptive
        self.random_state = check_random_state(random_state)
        self.iteration_count = 0
        self.improvement_history = []
        
    def __call__(self, X, model, y_opt=None):
        """
        Calculate enhanced expected improvement.
        
        Parameters
        ----------
        X : array-like
            Points to evaluate acquisition function.
        model : object
            Fitted surrogate model.
        y_opt : float
            Current best objective value.
        
        Returns
        -------
        acq_values : array
            Acquisition function values.
        """
        if y_opt is None:
            y_opt = np.min(model.y_train_)
        
        # Get predictions and uncertainties
        mu, sigma = model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        # Adaptive exploration factor
        if self.adaptive and len(self.improvement_history) > 5:
            recent_improvements = self.improvement_history[-5:]
            avg_improvement = np.mean(recent_improvements)
            
            # Adjust exploration based on recent progress
            if avg_improvement < 1e-4:
                # Low improvement, increase exploration
                current_exploration = self.exploration_factor * 10
            else:
                # Good improvement, maintain balance
                current_exploration = self.exploration_factor
        else:
            current_exploration = self.exploration_factor
        
        # Calculate expected improvement
        with np.errstate(divide='warn'):
            imp = mu - y_opt - current_exploration
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei.ravel()


class ProbabilityOfImprovementPlus:
    """
    Enhanced Probability of Improvement with confidence weighting.
    """
    
    def __init__(self, confidence_threshold=0.1, random_state=None):
        """
        Initialize Enhanced Probability of Improvement.
        
        Parameters
        ----------
        confidence_threshold : float
            Minimum confidence threshold for improvement.
        random_state : int or RandomState
            Random state for reproducibility.
        """
        self.confidence_threshold = confidence_threshold
        self.random_state = check_random_state(random_state)
        
    def __call__(self, X, model, y_opt=None):
        """
        Calculate enhanced probability of improvement.
        
        Parameters
        ----------
        X : array-like
            Points to evaluate acquisition function.
        model : object
            Fitted surrogate model.
        y_opt : float
            Current best objective value.
        
        Returns
        -------
        acq_values : array
            Acquisition function values.
        """
        if y_opt is None:
            y_opt = np.min(model.y_train_)
        
        # Get predictions and uncertainties
        mu, sigma = model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        # Calculate probability of improvement
        with np.errstate(divide='warn'):
            Z = (mu - y_opt) / sigma
            pi = norm.cdf(Z)
            
            # Apply confidence weighting
            confidence_weight = 1.0 - np.exp(-sigma / self.confidence_threshold)
            weighted_pi = pi * confidence_weight
            
        return weighted_pi.ravel()


class LowerConfidenceBoundPlus:
    """
    Enhanced Lower Confidence Bound with dynamic confidence scaling.
    """
    
    def __init__(self, kappa=2.576, alpha=0.1, adaptive=True, random_state=None):
        """
        Initialize Enhanced Lower Confidence Bound.
        
        Parameters
        ----------
        kappa : float
            Exploration parameter.
        alpha : float
            Confidence level parameter.
        adaptive : bool
            Whether to use adaptive parameters.
        random_state : int or RandomState
            Random state for reproducibility.
        """
        self.kappa = kappa
        self.alpha = alpha
        self.adaptive = adaptive
        self.random_state = check_random_state(random_state)
        self.iteration_count = 0
        
    def __call__(self, X, model, y_opt=None):
        """
        Calculate enhanced lower confidence bound.
        
        Parameters
        ----------
        X : array-like
            Points to evaluate acquisition function.
        model : object
            Fitted surrogate model.
        y_opt : float
            Current best objective value (not used in LCB).
        
        Returns
        -------
        acq_values : array
            Acquisition function values.
        """
        # Get predictions and uncertainties
        mu, sigma = model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        # Adaptive parameters
        if self.adaptive:
            # Reduce exploration over time
            adaptive_kappa = self.kappa * np.sqrt(1.0 / (1.0 + self.iteration_count))
            self.iteration_count += 1
        else:
            adaptive_kappa = self.kappa
        
        # Calculate lower confidence bound
        lcb = mu - adaptive_kappa * sigma
        
        return lcb.ravel()


class KnowledgeGradient:
    """
    Knowledge Gradient acquisition function for multi-step lookahead.
    
    This acquisition function considers the expected improvement after one
    additional function evaluation.
    """
    
    def __init__(self, n_samples=100, random_state=None):
        """
        Initialize Knowledge Gradient.
        
        Parameters
        ----------
        n_samples : int
            Number of samples for Monte Carlo estimation.
        random_state : int or RandomState
            Random state for reproducibility.
        """
        self.n_samples = n_samples
        self.random_state = check_random_state(random_state)
        
    def __call__(self, X, model, y_opt=None):
        """
        Calculate knowledge gradient.
        
        Parameters
        ----------
        X : array-like
            Points to evaluate acquisition function.
        model : object
            Fitted surrogate model.
        y_opt : float
            Current best objective value.
        
        Returns
        -------
        acq_values : array
            Knowledge gradient values.
        """
        if y_opt is None:
            y_opt = np.min(model.y_train_)
        
        # Get current predictions
        mu, sigma = model.predict(X, return_std=True)
        
        kg_values = np.zeros(len(X))
        
        for i, x in enumerate(X):
            # Sample from posterior at this point
            samples = self.random_state.normal(mu[i], sigma[i], self.n_samples)
            
            # For each sample, predict improvement at other points
            total_improvement = 0.0
            
            for sample in samples:
                # Create temporary model with this sample
                temp_X = np.vstack([model.X_train_, x.reshape(1, -1)])
                temp_y = np.hstack([model.y_train_, sample])
                
                # Fit temporary model (simplified)
                temp_model = GaussianProcessRegressor()
                temp_model.fit(temp_X, temp_y)
                
                # Predict at all training points
                temp_mu, _ = temp_model.predict(model.X_train_, return_std=True)
                
                # Calculate expected improvement
                improvements = np.maximum(0, y_opt - temp_mu)
                total_improvement += np.mean(improvements)
            
            # Knowledge gradient is the expected improvement
            kg_values[i] = total_improvement / self.n_samples - y_opt
        
        return kg_values


class ThompsonSampling:
    """
    Thompson Sampling acquisition function for Bayesian optimization.
    
    This acquisition function samples from the posterior distribution
    and selects points based on the sampled objective values.
    """
    
    def __init__(self, n_samples=1, random_state=None):
        """
        Initialize Thompson Sampling.
        
        Parameters
        ----------
        n_samples : int
            Number of Thompson samples to average.
        random_state : int or RandomState
            Random state for reproducibility.
        """
        self.n_samples = n_samples
        self.random_state = check_random_state(random_state)
        
    def __call__(self, X, model, y_opt=None):
        """
        Calculate Thompson sampling acquisition values.
        
        Parameters
        ----------
        X : array-like
            Points to evaluate acquisition function.
        model : object
            Fitted surrogate model.
        y_opt : float
            Current best objective value (not used in Thompson Sampling).
        
        Returns
        -------
        acq_values : array
            Thompson sampling values (negative for minimization).
        """
        # Get predictions and uncertainties
        mu, sigma = model.predict(X, return_std=True)
        
        # Sample from posterior
        samples = self.random_state.normal(mu, sigma, (self.n_samples, len(X)))
        
        # Take minimum across samples (for minimization)
        thompson_values = np.min(samples, axis=0)
        
        # Return negative for minimization framework
        return -thompson_values


class MaxValueEntropySearch:
    """
    Max-value Entropy Search acquisition function.
    
    This acquisition function maximizes the information gain about the
    global maximum of the objective function.
    """
    
    def __init__(self, n_samples=100, random_state=None):
        """
        Initialize Max-value Entropy Search.
        
        Parameters
        ----------
        n_samples : int
            Number of samples for entropy estimation.
        random_state : int or RandomState
            Random state for reproducibility.
        """
        self.n_samples = n_samples
        self.random_state = check_random_state(random_state)
        
    def __call__(self, X, model, y_opt=None):
        """
        Calculate max-value entropy search acquisition values.
        
        Parameters
        ----------
        X : array-like
            Points to evaluate acquisition function.
        model : object
            Fitted surrogate model.
        y_opt : float
            Current best objective value.
        
        Returns
        -------
        acq_values : array
            MES acquisition values.
        """
        if y_opt is None:
            y_opt = np.min(model.y_train_)
        
        # Sample from the posterior distribution of the maximum
        # This is a simplified implementation
        mu, sigma = model.predict(X, return_std=True)
        
        # Calculate entropy reduction (simplified)
        entropy_values = sigma * np.abs(mu - y_opt)
        
        return entropy_values.ravel()


# Utility functions for acquisition function selection
def select_acquisition_function(name, **kwargs):
    """
    Select and initialize an acquisition function.
    
    Parameters
    ----------
    name : str
        Name of the acquisition function.
    **kwargs : dict
        Additional parameters for the acquisition function.
    
    Returns
    -------
    acq_func : object
        Initialized acquisition function.
    """
    acquisition_functions = {
        'ei_plus': ExpectedImprovementPlus,
        'pi_plus': ProbabilityOfImprovementPlus,
        'lcb_plus': LowerConfidenceBoundPlus,
        'kg': KnowledgeGradient,
        'ts': ThompsonSampling,
        'mes': MaxValueEntropySearch
    }
    
    if name not in acquisition_functions:
        raise ValueError(f"Unknown acquisition function: {name}")
    
    return acquisition_functions[name](**kwargs)


def adaptive_acquisition_selector(model, iteration, max_iterations):
    """
    Adaptively select acquisition function based on optimization progress.
    
    Parameters
    ----------
    model : object
        Fitted surrogate model.
    iteration : int
        Current iteration number.
    max_iterations : int
        Maximum number of iterations.
    
    Returns
    -------
    acq_func_name : str
        Recommended acquisition function name.
    """
    progress_ratio = iteration / max_iterations
    
    # Early iterations: exploration-focused
    if progress_ratio < 0.3:
        return 'ts'  # Thompson Sampling for exploration
    
    # Middle iterations: balanced approach
    elif progress_ratio < 0.7:
        return 'ei_plus'  # Enhanced Expected Improvement
    
    # Late iterations: exploitation-focused
    else:
        return 'lcb_plus'  # Enhanced Lower Confidence Bound
