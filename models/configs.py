"""
Configuration classes for anomaly detection model.

Provides dataclass-based configurations following ML best practices:
- Centralized parameter management
- Type-safe configuration objects
- Built-in validation
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class PCAConfig:
    """Configuration for PCA feature reduction."""
    n_components: int = 20
    whiten: bool = True
    random_state: int = 42
    
    def validate(self) -> None:
        """Validate PCA configuration parameters."""
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}")
        if self.random_state < 0:
            raise ValueError(f"random_state must be >= 0, got {self.random_state}")


@dataclass
class VAEConfig:
    """Configuration for VAE feature reduction and training."""
    latent_dim: Optional[int] = None
    hidden_dim: int = 512
    dropout: float = 0.2
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    beta: float = 2.0
    alpha: float = 0.5
    early_stopping: bool = True
    patience: int = 5
    seed: int = 42
    score_mode: str = "gmm_mu"  # 'gmm_mu' | 'recon_kl' | 'hybrid'
    hybrid_w_gmm: float = 0.5
    hybrid_w_recon: float = 0.5
    finetune_encoder_layers: int = 0
    encoder_lr: float = 3e-5
    
    def validate(self) -> None:
        """Validate VAE configuration parameters."""
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.score_mode not in ("gmm_mu", "recon_kl", "hybrid"):
            raise ValueError(f"score_mode must be one of 'gmm_mu', 'recon_kl', 'hybrid', got {self.score_mode}")
        if self.hybrid_w_gmm < 0 or self.hybrid_w_recon < 0:
            raise ValueError(f"Hybrid weights must be non-negative")


@dataclass
class GMMConfig:
    """Configuration for Gaussian Mixture Model."""
    Ks: Tuple[int, ...] = field(default_factory=lambda: (1, 2, 3, 4, 5))
    covariance_type: str = "full"
    reg_covar: float = 1e-5
    max_iter: int = 100
    random_state: int = 42
    
    def validate(self) -> None:
        """Validate GMM configuration parameters."""
        if not self.Ks or min(self.Ks) < 1:
            raise ValueError(f"Ks must contain positive integers, got {self.Ks}")
        if self.covariance_type not in ("full", "tied", "diag", "spherical"):
            raise ValueError(f"covariance_type must be one of 'full', 'tied', 'diag', 'spherical', got {self.covariance_type}")
        if self.reg_covar < 0:
            raise ValueError(f"reg_covar must be >= 0, got {self.reg_covar}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")


@dataclass
class ThresholdConfig:
    """Configuration for threshold calibration."""
    train_val_split: float = 0.95
    quantile: float = 0.05
    
    def validate(self) -> None:
        """Validate threshold configuration."""
        if not 0 < self.train_val_split < 1:
            raise ValueError(f"train_val_split must be in (0, 1), got {self.train_val_split}")
        if not 0 <= self.quantile <= 1:
            raise ValueError(f"quantile must be in [0, 1], got {self.quantile}")


@dataclass
class ModelConfig:
    """Complete model configuration combining all sub-configs."""
    ft_reduction: str = "PCA"  # "PCA" or "VAE"
    pca: PCAConfig = field(default_factory=PCAConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    gmm: GMMConfig = field(default_factory=GMMConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    file_extensions: Tuple[str, ...] = field(default_factory=lambda: ("*.png", "*.jpg", "*.jpeg"))
    test_grid_cols: int = 5
    run_timestamp: Optional[str] = None
    runs_root: str = "training_runs"
    save_vae_training: bool = True
    
    def validate(self) -> None:
        """Validate entire configuration."""
        if self.ft_reduction not in ("PCA", "VAE"):
            raise ValueError(f"ft_reduction must be 'PCA' or 'VAE', got {self.ft_reduction}")
        if self.test_grid_cols < 1:
            raise ValueError(f"test_grid_cols must be >= 1, got {self.test_grid_cols}")
        
        self.pca.validate()
        self.vae.validate()
        self.gmm.validate()
        self.threshold.validate()
