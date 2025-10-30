"""
Unit Tests for Refactored Model

Demonstrates how to properly test the refactored components
in isolation using the SOLID principles.
"""

import unittest
import tempfile
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from models.model import (
    PCAConfig, VAEConfig, GMMConfig, ThresholdConfig, ModelConfig,
    PCAReducer, VAEReducer, GMMScorer, ReconstructionScorer,
    ThresholdCalibrator, AnomalyDetectionModel,
    setup_logger,
)


# ============================================================================
# Configuration Tests
# ============================================================================

class TestPCAConfig(unittest.TestCase):
    """Test PCA configuration validation."""
    
    def test_valid_config(self):
        """Valid configuration should not raise."""
        config = PCAConfig(n_components=20, whiten=True, random_state=42)
        config.validate()  # Should not raise
    
    def test_invalid_n_components_zero(self):
        """n_components < 1 should raise ValueError."""
        config = PCAConfig(n_components=0)
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_invalid_n_components_negative(self):
        """Negative n_components should raise ValueError."""
        config = PCAConfig(n_components=-5)
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_invalid_random_state(self):
        """Negative random_state should raise ValueError."""
        config = PCAConfig(random_state=-1)
        with self.assertRaises(ValueError):
            config.validate()


class TestVAEConfig(unittest.TestCase):
    """Test VAE configuration validation."""
    
    def test_valid_config(self):
        """Valid VAE config should not raise."""
        config = VAEConfig(latent_dim=32, epochs=100, lr=1e-3)
        config.validate()
    
    def test_invalid_hidden_dim(self):
        """Invalid hidden_dim should raise."""
        config = VAEConfig(hidden_dim=0)
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_invalid_dropout(self):
        """Invalid dropout (not in [0,1)) should raise."""
        config = VAEConfig(dropout=1.5)
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_invalid_lr(self):
        """Learning rate <= 0 should raise."""
        config = VAEConfig(lr=0)
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_invalid_score_mode(self):
        """Invalid score_mode should raise."""
        config = VAEConfig(score_mode="invalid_mode")
        with self.assertRaises(ValueError):
            config.validate()


class TestGMMConfig(unittest.TestCase):
    """Test GMM configuration validation."""
    
    def test_valid_config(self):
        """Valid GMM config should not raise."""
        config = GMMConfig(Ks=(1, 2, 3, 4, 5))
        config.validate()
    
    def test_invalid_ks_empty(self):
        """Empty Ks should raise."""
        config = GMMConfig(Ks=())
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_invalid_ks_zero(self):
        """Ks containing zero should raise."""
        config = GMMConfig(Ks=(0, 1, 2))
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_invalid_covariance_type(self):
        """Invalid covariance_type should raise."""
        config = GMMConfig(covariance_type="invalid")
        with self.assertRaises(ValueError):
            config.validate()


class TestThresholdConfig(unittest.TestCase):
    """Test threshold configuration validation."""
    
    def test_valid_config(self):
        """Valid threshold config should not raise."""
        config = ThresholdConfig(train_val_split=0.9, quantile=0.05)
        config.validate()
    
    def test_invalid_split_zero(self):
        """train_val_split = 0 should raise."""
        config = ThresholdConfig(train_val_split=0.0)
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_invalid_split_one(self):
        """train_val_split = 1.0 should raise."""
        config = ThresholdConfig(train_val_split=1.0)
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_invalid_quantile_negative(self):
        """Negative quantile should raise."""
        config = ThresholdConfig(quantile=-0.1)
        with self.assertRaises(ValueError):
            config.validate()


class TestModelConfig(unittest.TestCase):
    """Test full model configuration validation."""
    
    def test_valid_config(self):
        """Valid model config should not raise."""
        config = ModelConfig(ft_reduction="PCA")
        config.validate()
    
    def test_invalid_ft_reduction(self):
        """Invalid ft_reduction should raise."""
        config = ModelConfig(ft_reduction="INVALID")
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_invalid_test_grid_cols(self):
        """Invalid test_grid_cols should raise."""
        config = ModelConfig(test_grid_cols=0)
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_cascading_validation(self):
        """Invalid sub-config should raise."""
        config = ModelConfig(
            pca=PCAConfig(n_components=-1)
        )
        with self.assertRaises(ValueError):
            config.validate()


# ============================================================================
# Feature Reducer Tests
# ============================================================================

class TestPCAReducer(unittest.TestCase):
    """Test PCA feature reducer."""
    
    def setUp(self):
        self.logger = setup_logger("test_pca_reducer")
        self.config = PCAConfig(n_components=10, random_state=42)
        self.reducer = PCAReducer(self.config, self.logger)
        self.X_train = np.random.randn(100, 64)
        self.X_test = np.random.randn(20, 64)
    
    def test_fit_reduces_dimensions(self):
        """Fit should reduce features to n_components."""
        Z = self.reducer.fit(self.X_train)
        self.assertEqual(Z.shape, (100, 10))
    
    def test_transform_after_fit(self):
        """Transform should work after fit."""
        self.reducer.fit(self.X_train)
        Z = self.reducer.transform(self.X_test)
        self.assertEqual(Z.shape, (20, 10))
    
    def test_transform_before_fit_raises(self):
        """Transform before fit should raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            self.reducer.transform(self.X_test)
    
    def test_fit_with_small_n_samples(self):
        """Fit should handle n_samples < n_components."""
        X_small = np.random.randn(5, 64)
        Z = self.reducer.fit(X_small)
        # Should cap to n_samples - 1
        self.assertLessEqual(Z.shape[1], 5)
    
    def test_state_serialization(self):
        """Get/load state should preserve reducer state."""
        self.reducer.fit(self.X_train)
        state = self.reducer.get_state()
        
        # Create new reducer and load state
        reducer2 = PCAReducer(PCAConfig(), self.logger)
        reducer2.load_state(state)
        
        # Should produce same transformation
        Z1 = self.reducer.transform(self.X_test)
        Z2 = reducer2.transform(self.X_test)
        np.testing.assert_array_almost_equal(Z1, Z2)


class TestVAEReducer(unittest.TestCase):
    """Test VAE feature reducer."""
    
    def setUp(self):
        self.logger = setup_logger("test_vae_reducer")
        self.device = "cpu"
        self.config = VAEConfig(
            latent_dim=16,
            hidden_dim=128,
            epochs=2,  # Quick for testing
            batch_size=32,
            seed=42,
        )
        self.reducer = VAEReducer(self.config, self.device, self.logger)
        self.X_train = np.random.randn(100, 64).astype(np.float32)
        self.X_val = np.random.randn(20, 64).astype(np.float32)
    
    def test_fit_trains_vae(self):
        """Fit should train VAE and return reduced features."""
        Z = self.reducer.fit(self.X_train, self.X_val)
        
        self.assertIsNotNone(self.reducer.vae)
        self.assertEqual(Z.shape, (100, 16))
        self.assertIn("loss", self.reducer.history)
    
    def test_transform_after_fit(self):
        """Transform should work after VAE training."""
        self.reducer.fit(self.X_train, self.X_val)
        Z = self.reducer.transform(self.X_val)
        self.assertEqual(Z.shape, (20, 16))
    
    def test_transform_before_fit_raises(self):
        """Transform before fit should raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            self.reducer.transform(self.X_val)


# ============================================================================
# Anomaly Scorer Tests
# ============================================================================

class TestGMMScorer(unittest.TestCase):
    """Test GMM-based scorer."""
    
    def setUp(self):
        self.logger = setup_logger("test_gmm_scorer")
        self.config = GMMConfig(Ks=(1, 2, 3), random_state=42)
        self.scorer = GMMScorer(self.config, self.logger)
        self.Z_train = np.random.randn(100, 16)
        self.Z_test = np.random.randn(20, 16)
    
    def test_fit_selects_best_gmm(self):
        """Fit should select GMM with best BIC."""
        self.scorer.fit(self.Z_train)
        
        self.assertIsNotNone(self.scorer.gmm)
        self.assertIn(self.scorer.gmm.n_components, (1, 2, 3))
    
    def test_score_after_fit(self):
        """Score should return per-sample scores."""
        self.scorer.fit(self.Z_train)
        scores = self.scorer.score(self.Z_test)
        
        self.assertEqual(len(scores), 20)
        self.assertTrue(np.all(np.isfinite(scores)))
    
    def test_score_before_fit_raises(self):
        """Score before fit should raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            self.scorer.score(self.Z_test)
    
    def test_state_serialization(self):
        """Get/load state should preserve scorer state."""
        self.scorer.fit(self.Z_train)
        state = self.scorer.get_state()
        
        # Create new scorer and load state
        scorer2 = GMMScorer(GMMConfig(), self.logger)
        scorer2.load_state(state)
        
        # Should produce same scores
        s1 = self.scorer.score(self.Z_test)
        s2 = scorer2.score(self.Z_test)
        np.testing.assert_array_almost_equal(s1, s2)


# ============================================================================
# Threshold Calibrator Tests
# ============================================================================

class TestThresholdCalibrator(unittest.TestCase):
    """Test threshold calibration."""
    
    def setUp(self):
        self.logger = setup_logger("test_calibrator")
        self.config = ThresholdConfig(quantile=0.05)
        self.calibrator = ThresholdCalibrator(self.config, self.logger)
    
    def test_quantile_calibration(self):
        """Quantile calibration should return valid threshold."""
        scores = np.random.randn(1000)
        tau = self.calibrator.calibrate(scores)
        
        self.assertIsInstance(tau, float)
        self.assertGreater(tau, np.min(scores))
        self.assertLess(tau, np.max(scores))
    
    def test_very_small_validation_set(self):
        """Small validation set should not crash."""
        scores = np.array([1.0, 2.0])
        tau = self.calibrator.calibrate(scores)
        
        self.assertIsInstance(tau, float)
        self.assertTrue(np.isfinite(tau))
    
    def test_f1_calibration_with_labels(self):
        """F1-based calibration should use labels if provided."""
        # Mix of normal (0) and anomalous (1) samples
        scores = np.concatenate([
            np.random.randn(50) + 1,      # Normal: higher scores
            np.random.randn(50) - 1,      # Anomalous: lower scores
        ])
        labels = np.concatenate([np.zeros(50), np.ones(50)])
        
        tau = self.calibrator.calibrate(scores, labels)
        
        # Threshold should be between the two distributions
        self.assertGreater(tau, np.mean(scores[labels == 1]))
        self.assertLess(tau, np.mean(scores[labels == 0]))


# ============================================================================
# Integration Tests
# ============================================================================

class TestAnomalyDetectionModelIntegration(unittest.TestCase):
    """Test complete model workflow."""
    
    def setUp(self):
        self.logger = setup_logger("test_integration")
        self.device = "cpu"
        
        # Mock encoder and preprocessor
        self.encoder = Mock()
        self.preprocess = Mock()
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_model_initialization(self):
        """Model should initialize without errors."""
        config = ModelConfig(ft_reduction="PCA")
        model = AnomalyDetectionModel(
            self.encoder, self.preprocess, self.device, config, self.logger
        )
        
        self.assertIsNotNone(model.reducer)
        self.assertIsNotNone(model.scorer)
        self.assertIsNotNone(model.calibrator)
    
    def test_invalid_config_raises(self):
        """Invalid config should raise during initialization."""
        config = ModelConfig(ft_reduction="INVALID")
        
        with self.assertRaises(ValueError):
            AnomalyDetectionModel(
                self.encoder, self.preprocess, self.device, config, self.logger
            )
    
    def test_reducer_scorer_compatibility(self):
        """Different reducer/scorer combinations should work."""
        combinations = [
            ("PCA", GMMScorer),
            ("VAE", GMMScorer),
        ]
        
        for ft_reduction, scorer_class in combinations:
            config = ModelConfig(ft_reduction=ft_reduction)
            model = AnomalyDetectionModel(
                self.encoder, self.preprocess, self.device, config, self.logger
            )
            
            # Model should be properly initialized
            self.assertEqual(model.config.ft_reduction, ft_reduction)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def setUp(self):
        self.logger = setup_logger("test_performance")
    
    def test_pca_scales_to_large_datasets(self):
        """PCA should handle large datasets efficiently."""
        config = PCAConfig(n_components=50, random_state=42)
        reducer = PCAReducer(config, self.logger)
        
        # Large dataset
        X = np.random.randn(10000, 512)
        Z = reducer.fit(X)
        
        self.assertEqual(Z.shape, (10000, 50))
    
    def test_gmm_fast_for_reasonable_sizes(self):
        """GMM should complete in reasonable time for typical data."""
        config = GMMConfig(Ks=(2, 3, 4, 5), random_state=42)
        scorer = GMMScorer(config, self.logger)
        
        # Typical dataset
        Z = np.random.randn(1000, 32)
        
        scorer.fit(Z)
        scores = scorer.score(Z)
        
        self.assertEqual(len(scores), 1000)


# ============================================================================
# Test Runner
# ============================================================================

def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPCAConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestVAEConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestGMMConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestThresholdConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestModelConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestPCAReducer))
    suite.addTests(loader.loadTestsFromTestCase(TestVAEReducer))
    suite.addTests(loader.loadTestsFromTestCase(TestGMMScorer))
    suite.addTests(loader.loadTestsFromTestCase(TestThresholdCalibrator))
    suite.addTests(loader.loadTestsFromTestCase(TestAnomalyDetectionModelIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
