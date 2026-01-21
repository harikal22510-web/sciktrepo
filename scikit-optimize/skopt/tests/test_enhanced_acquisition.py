"""
Comprehensive tests for advanced acquisition functions.
This provides thorough testing of all enhanced acquisition methods.
"""

import pytest
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from skopt.space import Real, Space
from skopt.acquisition.enhanced_acquisition import (
    ExpectedImprovementPlus,
    ProbabilityOfImprovementPlus,
    LowerConfidenceBoundPlus,
    KnowledgeGradient,
    ThompsonSampling,
    MaxValueEntropySearch,
    select_acquisition_function,
    adaptive_acquisition_selector
)


class TestExpectedImprovementPlus:
    """Comprehensive tests for Enhanced Expected Improvement."""
    
    def test_initialization(self):
        """Test EI+ initialization with different parameters."""
        # Test basic initialization
        ei = ExpectedImprovementPlus()
        assert ei.exploration_factor == 0.01
        assert ei.adaptive is True
        
        # Test custom parameters
        ei_custom = ExpectedImprovementPlus(
            exploration_factor=0.05,
            adaptive=False,
            random_state=42
        )
        assert ei_custom.exploration_factor == 0.05
        assert ei_custom.adaptive is False
    
    def test_acquisition_calculation(self):
        """Test EI+ acquisition function calculation."""
        # Create test data
        X = np.random.random(10, 2)
        y = np.random.random(10)
        
        # Fit GP model
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        # Test EI+ calculation
        ei = ExpectedImprovementPlus(random_state=42)
        acq_values = ei(X, model, y_opt=np.min(y))
        
        # Validate results
        assert len(acq_values) == 10
        assert np.all(acq_values >= 0)  # EI should be non-negative
        assert np.all(np.isfinite(acq_values))
    
    def test_adaptive_exploration(self):
        """Test adaptive exploration factor adjustment."""
        ei = ExpectedImprovementPlus(adaptive=True, random_state=42)
        
        # Simulate improvement history
        ei.improvement_history = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
        
        # Create test data
        X = np.random.random(5, 2)
        y = np.random.random(5)
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        # Test with low improvement (should increase exploration)
        acq_values_low = ei(X, model, y_opt=np.min(y))
        
        # Test with good improvement
        ei.improvement_history = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        acq_values_good = ei(X, model, y_opt=np.min(y))
        
        # Both should produce valid results
        assert len(acq_values_low) == 5
        assert len(acq_values_good) == 5
        assert np.all(np.isfinite(acq_values_low))
        assert np.all(np.isfinite(acq_values_good))


class TestProbabilityOfImprovementPlus:
    """Tests for Enhanced Probability of Improvement."""
    
    def test_initialization(self):
        """Test PI+ initialization."""
        pi = ProbabilityOfImprovementPlus(confidence_threshold=0.2)
        assert pi.confidence_threshold == 0.2
    
    def test_acquisition_calculation(self):
        """Test PI+ acquisition function calculation."""
        X = np.random.random(8, 3)
        y = np.random.random(8)
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        pi = ProbabilityOfImprovementPlus(random_state=42)
        acq_values = pi(X, model, y_opt=np.min(y))
        
        assert len(acq_values) == 8
        assert np.all(acq_values >= 0)  # PI should be non-negative
        assert np.all(acq_values <= 1)  # PI should be probability
    
    def test_confidence_weighting(self):
        """Test confidence weighting in PI+."""
        X = np.random.random(6, 2)
        y = np.random.random(6)
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        # Test with different confidence thresholds
        pi_low = ProbabilityOfImprovementPlus(confidence_threshold=0.05)
        pi_high = ProbabilityOfImprovementPlus(confidence_threshold=0.5)
        
        acq_low = pi_low(X, model, y_opt=np.min(y))
        acq_high = pi_high(X, model, y_opt=np.min(y))
        
        # Both should produce valid results
        assert len(acq_low) == 6
        assert len(acq_high) == 6
        assert np.all(np.isfinite(acq_low))
        assert np.all(np.isfinite(acq_high))


class TestLowerConfidenceBoundPlus:
    """Tests for Enhanced Lower Confidence Bound."""
    
    def test_initialization(self):
        """Test LCB+ initialization."""
        lcb = LowerConfidenceBoundPlus(kappa=3.0, alpha=0.2, adaptive=True)
        assert lcb.kappa == 3.0
        assert lcb.alpha == 0.2
        assert lcb.adaptive is True
    
    def test_acquisition_calculation(self):
        """Test LCB+ acquisition function calculation."""
        X = np.random.random(7, 2)
        y = np.random.random(7)
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        lcb = LowerConfidenceBoundPlus(random_state=42)
        acq_values = lcb(X, model)
        
        assert len(acq_values) == 7
        assert np.all(np.isfinite(acq_values))
    
    def test_adaptive_kappa(self):
        """Test adaptive kappa adjustment."""
        lcb = LowerConfidenceBoundPlus(adaptive=True, random_state=42)
        
        X = np.random.random(5, 2)
        y = np.random.random(5)
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        # Test multiple iterations
        acq_values_1 = lcb(X, model)
        acq_values_2 = lcb(X, model)  # Second call should have lower kappa
        
        assert len(acq_values_1) == 5
        assert len(acq_values_2) == 5
        assert np.all(np.isfinite(acq_values_1))
        assert np.all(np.isfinite(acq_values_2))


class TestKnowledgeGradient:
    """Tests for Knowledge Gradient acquisition function."""
    
    def test_initialization(self):
        """Test KG initialization."""
        kg = KnowledgeGradient(n_samples=50, random_state=42)
        assert kg.n_samples == 50
    
    def test_acquisition_calculation(self):
        """Test KG acquisition function calculation."""
        X = np.random.random(6, 2)
        y = np.random.random(6)
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        kg = KnowledgeGradient(n_samples=20, random_state=42)
        acq_values = kg(X, model, y_opt=np.min(y))
        
        assert len(acq_values) == 6
        assert np.all(np.isfinite(acq_values))
    
    def test_sample_size_impact(self):
        """Test impact of sample size on KG calculation."""
        X = np.random.random(4, 2)
        y = np.random.random(4)
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        # Test with different sample sizes
        kg_small = KnowledgeGradient(n_samples=10, random_state=42)
        kg_large = KnowledgeGradient(n_samples=100, random_state=42)
        
        acq_small = kg_small(X, model, y_opt=np.min(y))
        acq_large = kg_large(X, model, y_opt=np.min(y))
        
        assert len(acq_small) == 4
        assert len(acq_large) == 4
        assert np.all(np.isfinite(acq_small))
        assert np.all(np.isfinite(acq_large))


class TestThompsonSampling:
    """Tests for Thompson Sampling acquisition function."""
    
    def test_initialization(self):
        """Test Thompson Sampling initialization."""
        ts = ThompsonSampling(n_samples=5, random_state=42)
        assert ts.n_samples == 5
    
    def test_acquisition_calculation(self):
        """Test Thompson Sampling acquisition function calculation."""
        X = np.random.random(8, 3)
        y = np.random.random(8)
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        ts = ThompsonSampling(n_samples=10, random_state=42)
        acq_values = ts(X, model)
        
        assert len(acq_values) == 8
        assert np.all(np.isfinite(acq_values))
    
    def test_sampling_reproducibility(self):
        """Test reproducibility of Thompson Sampling."""
        X = np.random.random(5, 2)
        y = np.random.random(5)
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        ts1 = ThompsonSampling(n_samples=5, random_state=42)
        ts2 = ThompsonSampling(n_samples=5, random_state=42)
        
        acq1 = ts1(X, model)
        acq2 = ts2(X, model)
        
        # Should be identical with same random state
        np.testing.assert_array_equal(acq1, acq2)


class TestMaxValueEntropySearch:
    """Tests for Max-value Entropy Search acquisition function."""
    
    def test_initialization(self):
        """Test MES initialization."""
        mes = MaxValueEntropySearch(n_samples=100, random_state=42)
        assert mes.n_samples == 100
    
    def test_acquisition_calculation(self):
        """Test MES acquisition function calculation."""
        X = np.random.random(6, 2)
        y = np.random.random(6)
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        mes = MaxValueEntropySearch(n_samples=50, random_state=42)
        acq_values = mes(X, model, y_opt=np.min(y))
        
        assert len(acq_values) == 6
        assert np.all(np.isfinite(acq_values))
        assert np.all(acq_values >= 0)  # Entropy should be non-negative


class TestAcquisitionUtilities:
    """Tests for acquisition function utilities."""
    
    def test_select_acquisition_function(self):
        """Test acquisition function selection."""
        # Test valid selections
        ei = select_acquisition_function('ei_plus', exploration_factor=0.02)
        assert isinstance(ei, ExpectedImprovementPlus)
        assert ei.exploration_factor == 0.02
        
        pi = select_acquisition_function('pi_plus', confidence_threshold=0.15)
        assert isinstance(pi, ProbabilityOfImprovementPlus)
        assert pi.confidence_threshold == 0.15
        
        lcb = select_acquisition_function('lcb_plus', kappa=3.0)
        assert isinstance(lcb, LowerConfidenceBoundPlus)
        assert lcb.kappa == 3.0
        
        kg = select_acquisition_function('kg', n_samples=50)
        assert isinstance(kg, KnowledgeGradient)
        assert kg.n_samples == 50
        
        ts = select_acquisition_function('ts', n_samples=5)
        assert isinstance(ts, ThompsonSampling)
        assert ts.n_samples == 5
        
        mes = select_acquisition_function('mes', n_samples=100)
        assert isinstance(mes, MaxValueEntropySearch)
        assert mes.n_samples == 100
    
    def test_invalid_acquisition_function(self):
        """Test invalid acquisition function selection."""
        with pytest.raises(ValueError):
            select_acquisition_function('invalid_function')
    
    def test_adaptive_acquisition_selector(self):
        """Test adaptive acquisition function selection."""
        X = np.random.random(10, 3)
        y = np.random.random(10)
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        # Test early iterations
        early_func = adaptive_acquisition_selector(model, 5, 100)
        assert early_func == 'ts'
        
        # Test middle iterations
        middle_func = adaptive_acquisition_selector(model, 50, 100)
        assert middle_func == 'ei_plus'
        
        # Test late iterations
        late_func = adaptive_acquisition_selector(model, 90, 100)
        assert late_func == 'lcb_plus'


class TestAcquisitionIntegration:
    """Integration tests for acquisition functions."""
    
    def test_acquisition_comparison(self):
        """Test comparison of different acquisition functions."""
        # Create test problem
        X = np.random.random(15, 3)
        y = np.random.random(15)
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        
        # Test different acquisition functions
        acquisition_functions = [
            ('ei_plus', ExpectedImprovementPlus(random_state=42)),
            ('pi_plus', ProbabilityOfImprovementPlus(random_state=42)),
            ('lcb_plus', LowerConfidenceBoundPlus(random_state=42)),
            ('ts', ThompsonSampling(n_samples=10, random_state=42))
        ]
        
        results = {}
        
        for name, acq_func in acquisition_functions:
            acq_values = acq_func(X, model, y_opt=np.min(y))
            results[name] = {
                'values': acq_values,
                'mean': np.mean(acq_values),
                'std': np.std(acq_values),
                'max': np.max(acq_values),
                'argmax': np.argmax(acq_values)
            }
        
        # Validate all results
        for name, result in results.items():
            assert len(result['values']) == 15
            assert np.all(np.isfinite(result['values']))
            assert result['mean'] >= 0
            assert result['std'] >= 0
            assert result['max'] >= 0
            assert 0 <= result['argmax'] < 15
    
    def test_acquisition_optimization_consistency(self):
        """Test that acquisition functions lead to consistent optimization."""
        # Simple 2D optimization problem
        def test_objective(x):
            return x[0]**2 + x[1]**2
        
        # Generate some data
        X_train = np.random.uniform(-2, 2, (20, 2))
        y_train = np.array([test_objective(x) for x in X_train])
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X_train, y_train)
        
        # Test points
        X_test = np.random.uniform(-2, 2, (10, 2))
        
        # Test different acquisition functions
        acq_functions = [
            ExpectedImprovementPlus(random_state=42),
            ProbabilityOfImprovementPlus(random_state=42),
            LowerConfidenceBoundPlus(random_state=42)
        ]
        
        best_points = []
        
        for acq_func in acq_functions:
            acq_values = acq_func(X_test, model, y_opt=np.min(y_train))
            best_idx = np.argmax(acq_values)
            best_points.append(X_test[best_idx])
        
        # All should suggest points near origin (optimal solution)
        for point in best_points:
            distance_from_origin = np.sqrt(point[0]**2 + point[1]**2)
            assert distance_from_origin < 2.0  # Should be reasonably close to origin


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
