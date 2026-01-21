"""
Advanced space transformations and dimensionality reduction techniques.
This file provides sophisticated transformation methods for complex search spaces.
"""

import numpy as np
from scipy.stats import qmc
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

from ..space import Space, Real, Integer, Categorical


class AdaptiveSpaceTransformer(BaseEstimator, TransformerMixin):
    """
    Adaptive space transformer that automatically selects the best transformation
    based on the characteristics of the search space and optimization progress.
    """
    
    def __init__(self, method='auto', n_components=None, random_state=None):
        """
        Initialize adaptive space transformer.
        
        Parameters
        ----------
        method : str
            Transformation method: 'auto', 'pca', 'isomap', 'standard', 'minmax'
        n_components : int
            Number of components for dimensionality reduction.
        random_state : int
            Random state for reproducibility.
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.transformer_ = None
        self.scaler_ = None
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.
        
        Parameters
        ----------
        X : array-like
            Input data to transform.
        y : array-like
            Target values (optional).
        
        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = np.asarray(X)
        
        # Determine optimal transformation method
        if self.method == 'auto':
            self.method = self._select_optimal_method(X)
        
        # Initialize scaler
        if self.method in ['pca', 'isomap']:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X.copy()
        
        # Initialize transformer
        if self.method == 'pca':
            n_components = self.n_components or min(X.shape[1], X.shape[0] // 2)
            self.transformer_ = PCA(n_components=n_components, random_state=self.random_state)
        elif self.method == 'isomap':
            n_components = self.n_components or min(X.shape[1], 10)
            self.transformer_ = Isomap(n_components=n_components)
        elif self.method == 'standard':
            self.transformer_ = StandardScaler()
        elif self.method == 'minmax':
            self.transformer_ = MinMaxScaler()
        
        # Fit transformer
        if self.method in ['pca', 'isomap']:
            self.transformer_.fit(X_scaled)
        else:
            self.transformer_.fit(X)
        
        return self
    
    def transform(self, X):
        """
        Transform the data using the fitted transformer.
        
        Parameters
        ----------
        X : array-like
            Input data to transform.
        
        Returns
        -------
        X_transformed : array
            Transformed data.
        """
        X = np.asarray(X)
        
        # Apply scaling if needed
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X.copy()
        
        # Apply transformation
        return self.transformer_.transform(X_scaled)
    
    def inverse_transform(self, X):
        """
        Inverse transform the data back to original space.
        
        Parameters
        ----------
        X : array-like
            Transformed data to inverse transform.
        
        Returns
        -------
        X_original : array
            Data in original space.
        """
        X = np.asarray(X)
        
        # Inverse transform
        X_transformed = self.transformer_.inverse_transform(X)
        
        # Inverse scaling if needed
        if self.scaler_ is not None:
            X_original = self.scaler_.inverse_transform(X_transformed)
        else:
            X_original = X_transformed
        
        return X_original
    
    def _select_optimal_method(self, X):
        """
        Select the optimal transformation method based on data characteristics.
        
        Parameters
        ----------
        X : array-like
            Input data.
        
        Returns
        -------
        method : str
            Selected transformation method.
        """
        n_samples, n_features = X.shape
        
        # High dimensional data -> PCA
        if n_features > 50:
            return 'pca'
        
        # Non-linear structure -> Isomap
        if n_samples > n_features * 2:
            return 'isomap'
        
        # Low dimensional -> Standard scaling
        if n_features <= 10:
            return 'standard'
        
        # Default -> MinMax scaling
        return 'minmax'


class HierarchicalSpaceTransformer(BaseEstimator, TransformerMixin):
    """
    Hierarchical space transformer for complex multi-level search spaces.
    
    This transformer handles search spaces with hierarchical relationships
    between parameters, such as nested configurations.
    """
    
    def __init__(self, hierarchy_levels=None, aggregation_method='weighted'):
        """
        Initialize hierarchical space transformer.
        
        Parameters
        ----------
        hierarchy_levels : list
            List of hierarchy level specifications.
        aggregation_method : str
            Method for aggregating hierarchical features.
        """
        self.hierarchy_levels = hierarchy_levels or []
        self.aggregation_method = aggregation_method
        self.level_transformers_ = {}
        
    def fit(self, X, y=None):
        """
        Fit the hierarchical transformer.
        
        Parameters
        ----------
        X : array-like
            Input data with hierarchical structure.
        y : array-like
            Target values.
        
        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = np.asarray(X)
        
        # Create transformers for each hierarchy level
        for i, level_spec in enumerate(self.hierarchy_levels):
            start_idx, end_idx = level_spec
            level_data = X[:, start_idx:end_idx]
            
            # Fit transformer for this level
            transformer = AdaptiveSpaceTransformer(method='auto')
            transformer.fit(level_data)
            self.level_transformers_[i] = transformer
        
        return self
    
    def transform(self, X):
        """
        Transform hierarchical data.
        
        Parameters
        ----------
        X : array-like
            Input hierarchical data.
        
        Returns
        -------
        X_transformed : array
            Transformed hierarchical data.
        """
        X = np.asarray(X)
        transformed_levels = []
        
        # Transform each hierarchy level
        for i, level_spec in enumerate(self.hierarchy_levels):
            start_idx, end_idx = level_spec
            level_data = X[:, start_idx:end_idx]
            
            # Transform this level
            transformer = self.level_transformers_[i]
            transformed_level = transformer.transform(level_data)
            transformed_levels.append(transformed_level)
        
        # Aggregate transformed levels
        if self.aggregation_method == 'weighted':
            # Weight by level importance
            weights = [1.0 / (i + 1) for i in range(len(transformed_levels))]
            X_transformed = np.hstack([w * level for w, level in zip(weights, transformed_levels)])
        else:
            # Simple concatenation
            X_transformed = np.hstack(transformed_levels)
        
        return X_transformed
    
    def inverse_transform(self, X):
        """
        Inverse transform hierarchical data.
        
        Parameters
        ----------
        X : array-like
            Transformed hierarchical data.
        
        Returns
        -------
        X_original : array
            Data in original hierarchical space.
        """
        X = np.asarray(X)
        
        # Split transformed data back into levels
        if self.aggregation_method == 'weighted':
            # Need to unweight the data
            weights = [1.0 / (i + 1) for i in range(len(self.hierarchy_levels))]
            transformed_levels = []
            start_idx = 0
            
            for i, transformer in enumerate(self.level_transformers_.items():
                end_idx = start_idx + transformer[1].transformer_.n_components
                level_data = X[:, start_idx:end_idx] / weights[i]
                transformed_levels.append(level_data)
                start_idx = end_idx
        else:
            # Simple split based on transformer dimensions
            transformed_levels = []
            start_idx = 0
            
            for i, transformer in enumerate(self.level_transformers_.items():
                end_idx = start_idx + transformer[1].transformer_.n_components
                level_data = X[:, start_idx:end_idx]
                transformed_levels.append(level_data)
                start_idx = end_idx
        
        # Inverse transform each level
        original_levels = []
        for i, (level_idx, transformer) in enumerate(self.level_transformers_.items()):
            original_level = transformer.inverse_transform(transformed_levels[i])
            original_levels.append(original_level)
        
        # Reconstruct original hierarchical structure
        X_original = np.hstack(original_levels)
        
        return X_original


class ConditionalSpaceTransformer(BaseEstimator, TransformerMixin):
    """
    Conditional space transformer for handling conditional dependencies
    between parameters.
    """
    
    def __init__(self, conditions=None, default_value=0.0):
        """
        Initialize conditional space transformer.
        
        Parameters
        ----------
        conditions : list
            List of conditional dependency specifications.
        default_value : float
            Default value for inactive parameters.
        """
        self.conditions = conditions or []
        self.default_value = default_value
        self.active_masks_ = {}
        
    def fit(self, X, y=None):
        """
        Fit the conditional transformer.
        
        Parameters
        ----------
        X : array-like
            Input data with conditional structure.
        y : array-like
            Target values.
        
        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = np.asarray(X)
        
        # Learn activation patterns for each condition
        for i, condition in enumerate(self.conditions):
            trigger_idx, affected_indices = condition
            
            # Determine when trigger is active
            trigger_values = X[:, trigger_idx]
            
            # Create mask for active parameters
            active_mask = np.zeros(len(X), dtype=bool)
            
            # Different activation strategies based on parameter type
            if trigger_idx in self._get_categorical_indices():
                # Categorical trigger: active for specific values
                active_values = set(np.unique(trigger_values)[:len(affected_indices)])
                active_mask = np.isin(trigger_values, list(active_values))
            else:
                # Numerical trigger: active when above threshold
                threshold = np.percentile(trigger_values, 50)
                active_mask = trigger_values > threshold
            
            self.active_masks_[i] = active_mask
        
        return self
    
    def transform(self, X):
        """
        Transform conditional data.
        
        Parameters
        ----------
        X : array-like
            Input conditional data.
        
        Returns
        -------
        X_transformed : array
            Transformed conditional data.
        """
        X = np.asarray(X)
        X_transformed = X.copy()
        
        # Apply conditional transformations
        for i, condition in enumerate(self.conditions):
            trigger_idx, affected_indices = condition
            active_mask = self.active_masks_[i]
            
            # Set inactive parameters to default value
            for idx in affected_indices:
                X_transformed[~active_mask, idx] = self.default_value
        
        return X_transformed
    
    def inverse_transform(self, X):
        """
        Inverse transform conditional data.
        
        Parameters
        ----------
        X : array-like
            Transformed conditional data.
        
        Returns
        -------
        X_original : array
            Data in original conditional space.
        """
        # For conditional transformer, inverse transform is identity
        return X.copy()
    
    def _get_categorical_indices(self):
        """
        Get indices of categorical parameters.
        
        Returns
        -------
        cat_indices : list
            Indices of categorical parameters.
        """
        # This would need to be implemented based on space information
        # For now, return empty list
        return []


class MultiObjectiveSpaceTransformer(BaseEstimator, TransformerMixin):
    """
    Multi-objective space transformer for handling multiple conflicting objectives.
    """
    
    def __init__(self, objectives=None, weighting_method='pareto'):
        """
        Initialize multi-objective space transformer.
        
        Parameters
        ----------
        objectives : list
            List of objective functions.
        weighting_method : str
            Method for combining objectives: 'pareto', 'weighted', 'nsga2'.
        """
        self.objectives = objectives or []
        self.weighting_method = weighting_method
        self.pareto_front_ = None
        self.weights_ = None
        
    def fit(self, X, y=None):
        """
        Fit the multi-objective transformer.
        
        Parameters
        ----------
        X : array-like
            Input parameters.
        y : array-like
            Multi-objective values.
        
        Returns
        -------
        self : object
            Fitted transformer.
        """
        if y is None:
            raise ValueError("Multi-objective transformer requires target values")
        
        y = np.asarray(y)
        
        if self.weighting_method == 'pareto':
            # Find Pareto front
            self.pareto_front_ = self._find_pareto_front(y)
        elif self.weighting_method == 'weighted':
            # Learn optimal weights
            self.weights_ = self._learn_weights(X, y)
        elif self.weighting_method == 'nsga2':
            # NSGA-II style ranking
            self.pareto_front_ = self._nsga2_ranking(y)
        
        return self
    
    def transform(self, X):
        """
        Transform multi-objective space.
        
        Parameters
        ----------
        X : array-like
            Input parameters.
        
        Returns
        -------
        X_transformed : array
            Transformed multi-objective space.
        """
        X = np.asarray(X)
        
        # For multi-objective transformer, we mainly transform the objective space
        # The parameter space remains unchanged
        return X.copy()
    
    def transform_objectives(self, y):
        """
        Transform multi-objective values.
        
        Parameters
        ----------
        y : array-like
            Multi-objective values.
        
        Returns
        -------
        y_transformed : array
            Transformed objective values.
        """
        y = np.asarray(y)
        
        if self.weighting_method == 'pareto':
            # Return Pareto rank
            return self._get_pareto_rank(y)
        elif self.weighting_method == 'weighted':
            # Return weighted sum
            return np.dot(y, self.weights_)
        elif self.weighting_method == 'nsga2':
            # Return NSGA-II rank
            return self._get_nsga2_rank(y)
        
        return y
    
    def _find_pareto_front(self, y):
        """Find Pareto front from multi-objective values."""
        pareto_mask = np.ones(len(y), dtype=bool)
        
        for i in range(len(y)):
            for j in range(len(y)):
                if i != j and np.all(y[j] <= y[i]) and np.any(y[j] < y[i]):
                    pareto_mask[i] = False
                    break
        
        return y[pareto_mask]
    
    def _learn_weights(self, X, y):
        """Learn optimal weights for weighted sum approach."""
        # Simple variance-based weighting
        return np.var(y, axis=0) / np.sum(np.var(y, axis=0))
    
    def _nsga2_ranking(self, y):
        """NSGA-II style ranking."""
        # Simplified NSGA-II implementation
        fronts = [self._find_pareto_front(y)]
        remaining = np.array([i for i in range(len(y)) if i not in fronts[0]])
        
        while len(remaining) > 0:
            front = self._find_pareto_front(y[remaining])
            fronts.append(front)
            remaining = np.array([i for i in remaining if i not in front])
        
        return fronts
    
    def _get_pareto_rank(self, y):
        """Get Pareto rank for each point."""
        ranks = np.zeros(len(y))
        
        for i, point in enumerate(y):
            rank = 0
            for j, other in enumerate(y):
                if i != j and np.all(other <= point) and np.any(other < point):
                    rank += 1
            ranks[i] = rank
        
        return ranks
    
    def _get_nsga2_rank(self, y):
        """Get NSGA-II rank for each point."""
        # Simplified implementation
        return self._get_pareto_rank(y)


# Utility functions for space transformation
def create_hierarchical_space(space_spec):
    """
    Create a hierarchical search space from specification.
    
    Parameters
    ----------
    space_spec : dict
        Hierarchical space specification.
    
    Returns
    -------
    space : Space
        Hierarchical search space.
    """
    dimensions = []
    hierarchy_levels = []
    
    for level_name, level_params in space_spec.items():
        start_idx = len(dimensions)
        
        for param_spec in level_params:
            if param_spec['type'] == 'real':
                dimensions.append(Real(param_spec['low'], param_spec['high']))
            elif param_spec['type'] == 'integer':
                dimensions.append(Integer(param_spec['low'], param_spec['high']))
            elif param_spec['type'] == 'categorical':
                dimensions.append(Categorical(param_spec['categories']))
        
        end_idx = len(dimensions)
        hierarchy_levels.append((start_idx, end_idx))
    
    return Space(dimensions), hierarchy_levels


def analyze_space_complexity(space):
    """
    Analyze the complexity of a search space.
    
    Parameters
    ----------
    space : Space
        Search space to analyze.
    
    Returns
    -------
    complexity : dict
        Complexity metrics.
    """
    dimensions = space.dimensions
    
    complexity = {
        'n_dimensions': len(dimensions),
        'n_real': sum(1 for d in dimensions if isinstance(d, Real)),
        'n_integer': sum(1 for d in dimensions if isinstance(d, Integer)),
        'n_categorical': sum(1 for d in dimensions if isinstance(d, Categorical)),
        'avg_categorical_size': np.mean([len(d.categories) for d in dimensions 
                                       if isinstance(d, Categorical)]),
        'has_constraints': hasattr(space, 'constraints'),
        'estimated_volume': 1.0
    }
    
    # Estimate search space volume
    for d in dimensions:
        if isinstance(d, Real):
            complexity['estimated_volume'] *= (d.high - d.low)
        elif isinstance(d, Integer):
            complexity['estimated_volume'] *= (d.high - d.low + 1)
        elif isinstance(d, Categorical):
            complexity['estimated_volume'] *= len(d.categories)
    
    return complexity
