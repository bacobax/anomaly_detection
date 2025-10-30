"""
Models package for anomaly detection.

Provides a modular, SOLID-principle-based architecture for anomaly detection models.

Modules:
- configs: Configuration dataclasses for model parameters
- reducers: Feature reduction strategies (PCA, VAE)
- scorers: Anomaly scoring strategies (GMM, Reconstruction, Hybrid)
- threshold: Threshold calibration utilities
- utils: Logging and visualization utilities
- model_refactored: Main AnomalyDetectionModel class
- VAE: Variational Autoencoder implementation
"""

from .configs import (
    PCAConfig,
    VAEConfig,
    GMMConfig,
    ThresholdConfig,
    ModelConfig,
)

from .reducers import (
    FeatureReducer,
    PCAReducer,
    VAEReducer,
)

from .scorers import (
    AnomalyScorer,
    GMMScorer,
    ReconstructionScorer,
    HybridScorer,
)

from .threshold import ThresholdCalibrator

from .utils import (
    setup_logger,
    save_test_visualizations,
    save_metrics_report,
    plot_model_2d_visualization,
)

from .model import AnomalyDetectionModel, Model

from .VAE import VAEonCLIP, train_vae, vae_reduce_mu, vae_loss_per_sample

__all__ = [
    # Configurations
    "PCAConfig",
    "VAEConfig",
    "GMMConfig",
    "ThresholdConfig",
    "ModelConfig",
    # Reducers
    "FeatureReducer",
    "PCAReducer",
    "VAEReducer",
    # Scorers
    "AnomalyScorer",
    "GMMScorer",
    "ReconstructionScorer",
    "HybridScorer",
    # Threshold
    "ThresholdCalibrator",
    # Utils
    "setup_logger",
    "save_test_visualizations",
    "save_metrics_report",
    "plot_model_2d_visualization",
    # Main Model
    "AnomalyDetectionModel",
    "Model",
    # VAE
    "VAEonCLIP",
    "train_vae",
    "vae_reduce_mu",
    "vae_loss_per_sample",
]
