"""
Root-level optimization utilities for enterprise-level Bayesian optimization.
This file provides high-level interfaces for the advanced optimization framework.
"""

import numpy as np
from typing import List, Callable, Dict, Any, Optional, Union
from sklearn.base import BaseEstimator

# Import advanced modules
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
    AdaptiveRandomForest,
    select_best_model
)
from scikit_optimize.skopt.enhanced_acquisition_module3 import (
    ExpectedImprovementPlus,
    ProbabilityOfImprovementPlus,
    LowerConfidenceBoundPlus,
    KnowledgeGradient,
    ThompsonSampling,
    MaxValueEntropySearch,
    select_acquisition_function,
    adaptive_acquisition_selector
)
from scikit_optimize.skopt.advanced_transformations_module4 import (
    AdaptiveSpaceTransformer,
    HierarchicalSpaceTransformer,
    ConditionalSpaceTransformer,
    MultiObjectiveSpaceTransformer,
    create_hierarchical_space,
    analyze_space_complexity,
    latin_hypercube_sampling,
    sobol_sequence
)


class EnterpriseOptimizationSuite:
    """
    Enterprise-level optimization suite combining all advanced algorithms.
    
    This class provides a unified interface for accessing all advanced optimization
    capabilities including adaptive optimizers, advanced models, acquisition functions,
    and space transformations.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the enterprise optimization suite.
        
        Parameters
        ----------
        random_state : int, optional
            Random state for reproducibility.
        """
        self.random_state = random_state
        self._available_optimizers = {
            'adaptive_bayesian': AdaptiveBayesianOptimizer,
            'multi_objective': MultiObjectiveOptimizer,
            'constrained': ConstrainedOptimizer
        }
        self._available_models = {
            'heteroscedastic_gp': HeteroscedasticGaussianProcess,
            'multi_fidelity_gp': MultiFidelityGaussianProcess,
            'deep_ensemble': DeepEnsembleRegressor,
            'robust_gp': RobustGaussianProcess,
            'adaptive_rf': AdaptiveRandomForest
        }
        self._available_acquisitions = {
            'ei_plus': ExpectedImprovementPlus,
            'pi_plus': ProbabilityOfImprovementPlus,
            'lcb_plus': LowerConfidenceBoundPlus,
            'kg': KnowledgeGradient,
            'ts': ThompsonSampling,
            'mes': MaxValueEntropySearch
        }
    
    def create_optimizer(self, 
                        optimizer_type: str = 'adaptive_bayesian',
                        dimensions: List = None,
                        **kwargs) -> BaseEstimator:
        """
        Create an optimizer of the specified type.
        
        Parameters
        ----------
        optimizer_type : str
            Type of optimizer to create.
        dimensions : list, optional
            Search space dimensions.
        **kwargs : dict
            Additional parameters for the optimizer.
        
        Returns
        -------
        optimizer : BaseEstimator
            Configured optimizer instance.
        """
        if optimizer_type not in self._available_optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        optimizer_class = self._available_optimizers[optimizer_type]
        
        if dimensions is not None:
            kwargs['dimensions'] = dimensions
        
        if 'random_state' not in kwargs:
            kwargs['random_state'] = self.random_state
        
        return optimizer_class(**kwargs)
    
    def create_model(self,
                     model_type: str = 'heteroscedastic_gp',
                     **kwargs) -> BaseEstimator:
        """
        Create a surrogate model of the specified type.
        
        Parameters
        ----------
        model_type : str
            Type of model to create.
        **kwargs : dict
            Additional parameters for the model.
        
        Returns
        -------
        model : BaseEstimator
            Configured model instance.
        """
        if model_type not in self._available_models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self._available_models[model_type]
        
        if 'random_state' not in kwargs:
            kwargs['random_state'] = self.random_state
        
        return model_class(**kwargs)
    
    def create_acquisition(self,
                          acquisition_type: str = 'ei_plus',
                          **kwargs) -> BaseEstimator:
        """
        Create an acquisition function of the specified type.
        
        Parameters
        ----------
        acquisition_type : str
            Type of acquisition function to create.
        **kwargs : dict
            Additional parameters for the acquisition function.
        
        Returns
 -------
        acquisition : BaseEstimator
            Configured acquisition function instance.
        """
        if acquisition_type not in self._available_acquisitions:
            raise ValueError(f"Unknown acquisition type: {acquisition_type}")
        
        acquisition_class = self._available_acquisitions[acquisition_type]
        
        if 'random_state' not in kwargs:
            kwargs['random_state'] = self.random_state
        
        return acquisition_class(**kwargs)
    
    def optimize(self,
                  objective: Callable,
                  dimensions: List,
                  optimizer_type: str = 'adaptive_bayesian',
                  model_type: str = 'heteroscedastic_gp',
                  acquisition_type: str = 'ei_plus',
                  n_calls: int = 100,
                  **kwargs) -> Dict[str, Any]:
        """
        Perform optimization with specified components.
        
        Parameters
        ----------
        objective : callable
            Objective function to minimize.
        dimensions : list
            Search space dimensions.
        optimizer_type : str
            Type of optimizer to use.
        model_type : str
            Type of surrogate model to use.
        acquisition_type : str
            Type of acquisition function to use.
        n_calls : int
            Number of function evaluations.
        **kwargs : dict
            Additional parameters for optimization.
        
        Returns
        -------
        result : dict
            Optimization result.
        """
        # Create components
        model = self.create_model(model_type, **kwargs)
        optimizer = self.create_optimizer(
            optimizer_type=optimizer_type,
            dimensions=dimensions,
            base_estimator=model,
            **kwargs
        )
        
        # Perform optimization
        result = optimizer.minimize(func=objective, n_calls=n_calls)
        
        return result
    
    def benchmark_optimizers(self,
                           objective: Callable,
                           dimensions: List,
                           n_calls: int = 50,
                           **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark multiple optimizers on the same problem.
        
        Parameters
        ----------
        objective : callable
            Objective function to minimize.
        dimensions : list
            Search space dimensions.
        n_calls : int
            Number of function evaluations.
        **kwargs : dict
            Additional parameters for optimization.
        
        Returns
        -------
        results : dict
            Results for each optimizer.
        """
        results = {}
        
        for optimizer_type in self._available_optimizers:
            try:
                result = self.optimize(
                    objective=objective,
                    dimensions=dimensions,
                    optimizer_type=optimizer_type,
                    n_calls=n_calls,
                    **kwargs
                )
                results[optimizer_type] = {
                    'best_value': result['fun'],
                    'n_iterations': result['nit'],
                    'success': result['success']
                }
            except Exception as e:
                results[optimizer_type] = {
                    'best_value': np.inf,
                    'n_iterations': 0,
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def analyze_performance(self,
                           results: Dict[str, Dict[str, Any]],
                           metric: str = 'best_value') -> Dict[str, float]:
        """
        Analyze performance of optimization results.
        
        Parameters
        ----------
        results : dict
            Results from benchmark_optimizers.
        metric : str
            Performance metric to analyze.
        
        Returns
        -------
        analysis : dict
            Performance analysis.
        """
        analysis = {}
        
        for optimizer_type, result in results.items():
            if result['success'] and metric in result:
                analysis[optimizer_type] = result[metric]
            else:
                analysis[optimizer_type] = np.inf
        
        # Find best performer
        if analysis:
            best_optimizer = min(analysis, key=analysis.get)
            best_value = analysis[best_optimizer]
            
            analysis['best_optimizer'] = best_optimizer
            analysis['best_value'] = best_value
        
        return analysis


# Convenience functions for common use cases
def quick_optimize(objective: Callable,
                  dimensions: List,
                  n_calls: int = 100,
                  random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Quick optimization with sensible defaults.
    
    Parameters
    ----------
    objective : callable
        Objective function to minimize.
    dimensions : list
        Search space dimensions.
    n_calls : int
        Number of function evaluations.
    random_state : int, optional
        Random state for reproducibility.
    
    Returns
    -------
    result : dict
        Optimization result.
    """
    suite = EnterpriseOptimizationSuite(random_state=random_state)
    return suite.optimize(
        objective=objective,
        dimensions=dimensions,
        n_calls=n_calls
    )


def robust_optimize(objective: Callable,
                    dimensions: List,
                    n_calls: int = 100,
                    random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Robust optimization with outlier detection and noise handling.
    
    Parameters
    ----------
    objective : callable
        Objective function to minimize.
    dimensions : list
        Search space dimensions.
    n_calls : int
        Number of function evaluations.
    random_state : int, optional
        Random state for reproducibility.
    
    Returns
    -------
    result : dict
        Optimization result.
    """
    suite = EnterpriseOptimizationSuite(random_state=random_state)
    return suite.optimize(
        objective=objective,
        dimensions=dimensions,
        model_type='robust_gp',
        optimizer_type='adaptive_bayesian',
        n_calls=n_calls
    )


def multi_objective_optimize(objectives: List[Callable],
                             dimensions: List,
                             n_calls: int = 100,
                             random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Multi-objective optimization with Pareto front analysis.
    
    Parameters
    ----------
    objectives : list
        List of objective functions.
    dimensions : list
        Search space dimensions.
    n_calls : int
        Number of function evaluations.
    random_state : int, optional
        Random state for reproducibility.
    
    Returns
    -------
    result : dict
        Optimization result with Pareto front.
    """
    suite = EnterpriseOptimizationSuite(random_state=random_state)
    
    # Create multi-objective wrapper
    def multi_objective_wrapper(x):
        return np.array([obj(x) for obj in objectives])
    
    return suite.optimize(
        objective=multi_objective_wrapper,
        dimensions=dimensions,
        optimizer_type='multi_objective',
        n_calls=n_calls
    )


# Export main classes and functions
__all__ = [
    'EnterpriseOptimizationSuite',
    'quick_optimize',
    'robust_optimize',
    'multi_objective_optimize',
    # Re-export all advanced components
    'AdaptiveBayesianOptimizer',
    'MultiObjectiveOptimizer',
    'ConstrainedOptimizer',
    'HeteroscedasticGaussianProcess',
    'MultiFidelityGaussianProcess',
    'DeepEnsembleRegressor',
    'RobustGaussianProcess',
    'AdaptiveRandomForest',
    'ExpectedImprovementPlus',
    'ProbabilityOfImprovementPlus',
    'LowerConfidenceBoundPlus',
    'KnowledgeGradient',
    'ThompsonSampling',
    'MaxValueEntropySearch',
    'AdaptiveSpaceTransformer',
    'HierarchicalSpaceTransformer',
    'ConditionalSpaceTransformer',
    'MultiObjectiveSpaceTransformer'
]
