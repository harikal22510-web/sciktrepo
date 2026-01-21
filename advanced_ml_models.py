"""
Advanced Machine Learning Models - File 2
Sophisticated surrogate models for Bayesian optimization with robustness features.
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class HeteroscedasticGaussianProcess(BaseEstimator, RegressorMixin):
    """
    Heteroscedastic Gaussian Process that models input-dependent noise.
    
    This GP can handle varying noise levels across the input space,
    which is common in real-world optimization problems.
    """
    
    def __init__(self, kernel=None, alpha=1e-10, n_restarts_optimizer=9, 
                 normalize_y=False, random_state=None):
        """
        Initialize heteroscedastic Gaussian Process.
        
        Parameters
        ----------
        kernel : kernel object
            Kernel for the GP.
        alpha : float
            Value added to diagonal of kernel matrix.
        n_restarts_optimizer : int
            Number of restarts for optimization.
        normalize_y : bool
            Whether to normalize target values.
        random_state : int
            Random state for reproducibility.
        """
        self.kernel = kernel
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Fit the heteroscedastic GP.
        
        Parameters
        ----------
        X : array-like
            Training inputs.
        y : array-like
            Training targets.
        
        Returns
        -------
        self : object
            Fitted model.
        """
        X, y = check_X_y(X, y, multi_output=False)
        
        # Set default kernel if not provided
        if self.kernel is None:
            self.kernel_ = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        else:
            self.kernel_ = self.kernel
        
        # Normalize targets if requested
        if self.normalize_y:
            self._y_mean = np.mean(y)
            self._y_std = np.std(y)
            y_normalized = (y - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            y_normalized = y
        
        # Fit the GP
        self.gp_ = GaussianProcessRegressor(
            kernel=self.kernel_,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=False,
            random_state=self.random_state
        )
        
        self.gp_.fit(X, y_normalized)
        
        # Estimate heteroscedastic noise
        self._estimate_noise_levels(X, y_normalized)
        
        self.X_train_ = X
        self.y_train_ = y
        
        return self
    
    def predict(self, X, return_std=False):
        """
        Predict using the heteroscedastic GP.
        
        Parameters
        ----------
        X : array-like
            Input points for prediction.
        return_std : bool
            Whether to return standard deviation.
        
        Returns
        -------
        y_mean : array
            Predicted mean.
        y_std : array
            Predicted standard deviation (if return_std=True).
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Predict with base GP
        y_mean_normalized, y_std_normalized = self.gp_.predict(X, return_std=True)
        
        # Add heteroscedastic noise
        heteroscedastic_noise = self._predict_noise_levels(X)
        total_std = np.sqrt(y_std_normalized**2 + heteroscedastic_noise**2)
        
        # Denormalize
        y_mean = y_mean_normalized * self._y_std + self._y_mean
        total_std = total_std * self._y_std
        
        if return_std:
            return y_mean, total_std
        else:
            return y_mean
    
    def _estimate_noise_levels(self, X, y):
        """Estimate input-dependent noise levels."""
        from sklearn.neighbors import NearestNeighbors
        
        n_neighbors = min(5, len(X))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(X)
        
        noise_levels = []
        for i, x in enumerate(X):
            # Find nearest neighbors
            distances, indices = nbrs.kneighbors([x])
            neighbor_y = y[indices[0]]
            
            # Estimate local variance
            local_var = np.var(neighbor_y)
            noise_levels.append(local_var)
        
        self.noise_levels_ = np.array(noise_levels)
        self.noise_gp_ = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0),
            alpha=1e-6,
            normalize_y=True
        )
        self.noise_gp_.fit(X, self.noise_levels_)
    
    def _predict_noise_levels(self, X):
        """Predict noise levels at new points."""
        noise_pred = self.noise_gp_.predict(X)
        return np.maximum(noise_pred, 1e-6)  # Ensure positive noise


class MultiFidelityGaussianProcess(BaseEstimator, RegressorMixin):
    """
    Multi-fidelity Gaussian Process for variable-cost optimization.
    
    This GP can handle data from different fidelity levels and learn
    the relationship between fidelities.
    """
    
    def __init__(self, kernel=None, fidelity_kernel=None, alpha=1e-10, 
                 n_restarts_optimizer=9, random_state=None):
        """
        Initialize multi-fidelity GP.
        
        Parameters
        ----------
        kernel : kernel object
            Kernel for input space.
        fidelity_kernel : kernel object
            Kernel for fidelity dimension.
        alpha : float
            Value added to diagonal of kernel matrix.
        n_restarts_optimizer : int
            Number of restarts for optimization.
        random_state : int
            Random state for reproducibility.
        """
        self.kernel = kernel
        self.fidelity_kernel = fidelity_kernel
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        
    def fit(self, X, y, fidelity=None):
        """
        Fit the multi-fidelity GP.
        
        Parameters
        ----------
        X : array-like
            Training inputs.
        y : array-like
            Training targets.
        fidelity : array-like
            Fidelity levels for each training point.
        
        Returns
        -------
        self : object
            Fitted model.
        """
        X, y = check_X_y(X, y, multi_output=False)
        
        if fidelity is None:
            fidelity = np.ones(len(X))
        
        fidelity = np.asarray(fidelity).reshape(-1, 1)
        
        # Combine input and fidelity
        X_extended = np.hstack([X, fidelity])
        
        # Set default kernels if not provided
        if self.kernel is None:
            input_kernel = RBF(length_scale=np.ones(X.shape[1]))
        else:
            input_kernel = self.kernel
        
        if self.fidelity_kernel is None:
            fidelity_kernel = RBF(length_scale=1.0)
        else:
            fidelity_kernel = self.fidelity_kernel
        
        # Combined kernel
        self.kernel_ = input_kernel * fidelity_kernel + WhiteKernel(noise_level=1.0)
        
        # Fit GP
        self.gp_ = GaussianProcessRegressor(
            kernel=self.kernel_,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True,
            random_state=self.random_state
        )
        
        self.gp_.fit(X_extended, y)
        
        self.X_train_ = X
        self.fidelity_train_ = fidelity
        self.y_train_ = y
        
        return self
    
    def predict(self, X, fidelity=1.0, return_std=False):
        """
        Predict using the multi-fidelity GP.
        
        Parameters
        ----------
        X : array-like
            Input points for prediction.
        fidelity : float or array-like
            Fidelity level(s) for prediction.
        return_std : bool
            Whether to return standard deviation.
        
        Returns
        -------
        y_mean : array
            Predicted mean.
        y_std : array
            Predicted standard deviation (if return_std=True).
        """
        check_is_fitted(self)
        X = check_array(X)
        
        fidelity = np.asarray(fidelity).reshape(-1, 1)
        
        # Combine input and fidelity
        X_extended = np.hstack([X, fidelity])
        
        # Predict
        return self.gp_.predict(X_extended, return_std=return_std)


class DeepEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Deep Ensemble Regressor combining multiple neural networks.
    
    This ensemble provides robust predictions with uncertainty estimates.
    """
    
    def __init__(self, n_estimators=5, hidden_layer_sizes=(100, 50), 
                 activation='relu', solver='adam', alpha=0.0001, 
                 random_state=None):
        """
        Initialize Deep Ensemble Regressor.
        
        Parameters
        ----------
        n_estimators : int
            Number of neural networks in ensemble.
        hidden_layer_sizes : tuple
            Hidden layer sizes for neural networks.
        activation : str
            Activation function.
        solver : str
            Solver for optimization.
        alpha : float
            Regularization parameter.
        random_state : int
            Random state for reproducibility.
        """
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Fit the deep ensemble.
        
        Parameters
        ----------
        X : array-like
            Training inputs.
        y : array-like
            Training targets.
        
        Returns
        -------
        self : object
            Fitted model.
        """
        X, y = check_X_y(X, y, multi_output=False)
        
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Create neural network with different random seed
            nn = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                max_iter=1000,
                random_state=self.random_state + i if self.random_state else None
            )
            
            # Bootstrap sample
            if i > 0:
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_boot, y_boot = X[indices], y[indices]
            else:
                X_boot, y_boot = X, y
            
            nn.fit(X_boot, y_boot)
            self.estimators_.append(nn)
        
        self.X_train_ = X
        self.y_train_ = y
        
        return self
    
    def predict(self, X, return_std=False):
        """
        Predict using the deep ensemble.
        
        Parameters
        ----------
        X : array-like
            Input points for prediction.
        return_std : bool
            Whether to return standard deviation.
        
        Returns
        -------
        y_mean : array
            Predicted mean.
        y_std : array
            Predicted standard deviation (if return_std=True).
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Get predictions from all estimators
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        
        # Calculate mean and standard deviation
        y_mean = np.mean(predictions, axis=0)
        
        if return_std:
            y_std = np.std(predictions, axis=0)
            return y_mean, y_std
        else:
            return y_mean


class RobustGaussianProcess(BaseEstimator, RegressorMixin):
    """
    Robust Gaussian Process with outlier detection and handling.
    
    This GP can handle outliers and noisy data robustly.
    """
    
    def __init__(self, kernel=None, alpha=1e-10, outlier_threshold=3.0,
                 n_restarts_optimizer=9, random_state=None):
        """
        Initialize Robust Gaussian Process.
        
        Parameters
        ----------
        kernel : kernel object
            Kernel for the GP.
        alpha : float
            Value added to diagonal of kernel matrix.
        outlier_threshold : float
            Threshold for outlier detection (in standard deviations).
        n_restarts_optimizer : int
            Number of restarts for optimization.
        random_state : int
            Random state for reproducibility.
        """
        self.kernel = kernel
        self.alpha = alpha
        self.outlier_threshold = outlier_threshold
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Fit the robust GP.
        
        Parameters
        ----------
        X : array-like
            Training inputs.
        y : array-like
            Training targets.
        
        Returns
        -------
        self : object
            Fitted model.
        """
        X, y = check_X_y(X, y, multi_output=False)
        
        # Detect and remove outliers
        inlier_mask = self._detect_outliers(X, y)
        X_clean, y_clean = X[inlier_mask], y[inlier_mask]
        
        # Set default kernel if not provided
        if self.kernel is None:
            self.kernel_ = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        else:
            self.kernel_ = self.kernel
        
        # Fit GP on clean data
        self.gp_ = GaussianProcessRegressor(
            kernel=self.kernel_,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True,
            random_state=self.random_state
        )
        
        self.gp_.fit(X_clean, y_clean)
        
        self.X_train_ = X
        self.y_train_ = y
        self.inlier_mask_ = inlier_mask
        
        return self
    
    def predict(self, X, return_std=False):
        """
        Predict using the robust GP.
        
        Parameters
        ----------
        X : array-like
            Input points for prediction.
        return_std : bool
            Whether to return standard deviation.
        
        Returns
        -------
        y_mean : array
            Predicted mean.
        y_std : array
            Predicted standard deviation (if return_std=True).
        """
        check_is_fitted(self)
        X = check_array(X)
        
        return self.gp_.predict(X, return_std=return_std)
    
    def _detect_outliers(self, X, y):
        """Detect outliers using robust statistics."""
        from sklearn.ensemble import IsolationForest
        
        # Use Isolation Forest for outlier detection
        iso_forest = IsolationForest(contamination=0.1, random_state=self.random_state)
        outlier_labels = iso_forest.fit_predict(X)
        
        # Also check for extreme y values
        y_mean, y_std = np.mean(y), np.std(y)
        y_outliers = np.abs(y - y_mean) > self.outlier_threshold * y_std
        
        # Combine outlier detection methods
        inlier_mask = (outlier_labels == 1) & (~y_outliers)
        
        return inlier_mask


class AdaptiveRandomForest(BaseEstimator, RegressorMixin):
    """
    Adaptive Random Forest that adjusts to data characteristics.
    
    This forest automatically adapts its structure based on the
    complexity and noise level of the data.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, adaptive=True, random_state=None):
        """
        Initialize Adaptive Random Forest.
        
        Parameters
        ----------
        n_estimators : int
            Number of trees in forest.
        max_depth : int
            Maximum depth of trees.
        min_samples_split : int
            Minimum samples required to split.
        min_samples_leaf : int
            Minimum samples required at leaf nodes.
        adaptive : bool
            Whether to adapt parameters to data.
        random_state : int
            Random state for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.adaptive = adaptive
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Fit the adaptive random forest.
        
        Parameters
        ----------
        X : array-like
            Training inputs.
        y : array-like
            Training targets.
        
        Returns
        -------
        self : object
            Fitted model.
        """
        X, y = check_X_y(X, y, multi_output=False)
        
        # Analyze data characteristics
        if self.adaptive:
            self._adapt_parameters(X, y)
        
        # Create forest with adapted parameters
        self.forest_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        
        self.forest_.fit(X, y)
        
        self.X_train_ = X
        self.y_train_ = y
        
        return self
    
    def predict(self, X, return_std=False):
        """
        Predict using the adaptive random forest.
        
        Parameters
        ----------
        X : array-like
            Input points for prediction.
        return_std : bool
            Whether to return standard deviation.
        
        Returns
        -------
        y_mean : array
            Predicted mean.
        y_std : array
            Predicted standard deviation (if return_std=True).
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.forest_.estimators_])
        
        # Calculate mean and standard deviation
        y_mean = np.mean(tree_predictions, axis=0)
        
        if return_std:
            y_std = np.std(tree_predictions, axis=0)
            return y_mean, y_std
        else:
            return y_mean
    
    def _adapt_parameters(self, X, y):
        """Adapt forest parameters based on data characteristics."""
        n_samples, n_features = X.shape
        
        # Adapt max depth based on data complexity
        if self.max_depth is None:
            if n_samples > 1000:
                self.max_depth = 15
            elif n_samples > 100:
                self.max_depth = 20
            else:
                self.max_depth = None
        
        # Adapt min samples based on noise level
        y_std = np.std(y)
        if y_std > np.std(y) * 0.5:  # High noise
            self.min_samples_leaf = max(5, self.min_samples_leaf)
            self.min_samples_split = max(10, self.min_samples_split)


# Utility function for model selection
def select_best_model(X, y, models=None, cv=5):
    """
    Select the best model from a list of candidates.
    
    Parameters
    ----------
    X : array-like
        Training inputs.
    y : array-like
        Training targets.
    models : list
        List of model candidates.
    cv : int
        Number of cross-validation folds.
    
    Returns
    -------
    best_model : object
        Best performing model.
    """
    from sklearn.model_selection import cross_val_score
    
    if models is None:
        models = [
            HeteroscedasticGaussianProcess(),
            MultiFidelityGaussianProcess(),
            DeepEnsembleRegressor(n_estimators=3),
            AdaptiveRandomForest(n_estimators=50),
            RobustGaussianProcess()
        ]
    
    best_score = -np.inf
    best_model = None
    
    for model in models:
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
        except:
            continue
    
    # Fit best model on all data
    if best_model is not None:
        best_model.fit(X, y)
    
    return best_model
