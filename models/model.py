"""
Refactored Model Class with SOLID Principles and ML Development Best Practices

SOLID Principles Applied:
- Single Responsibility: Separate concerns into specific classes for feature reduction,
  scoring, and model persistence.
- Open/Closed: Use abstract base classes for extensibility without modifying existing code.
- Liskov Substitution: Abstract classes ensure interchangeable implementations.
- Interface Segregation: Specific interfaces for different components.
- Dependency Inversion: Depend on abstractions (feature reducer, scorer) not concrete classes.

ML Best Practices:
- Separation of concerns: Feature extraction, reduction, scoring, and threshold calibration
- Type hints and validation for better error detection
- Configuration objects instead of long parameter lists
- Comprehensive logging and warnings
- Reproducibility through explicit seed management
- Model versioning and serialization
"""

import numpy as np
import torch
import pickle
import json
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Tuple, Union, List, Any
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
import seaborn as sns
import logging
from tqdm import tqdm

from .VAE import VAEonCLIP, train_vae, vae_reduce_mu, vae_loss_per_sample


# ============================================================================
# Configuration Classes (Replaces long parameter lists)
# ============================================================================

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


# ============================================================================
# Abstract Base Classes (Strategy Pattern)
# ============================================================================

class FeatureReducer(ABC):
    """Abstract base class for feature reduction strategies."""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit reducer on training data and return reduced features."""
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce features."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get reducer state for serialization."""
        pass
    
    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load reducer state from serialization."""
        pass


class AnomalyScorer(ABC):
    """Abstract base class for anomaly scoring strategies."""
    
    @abstractmethod
    def fit(self, Z: np.ndarray, X_original: Optional[np.ndarray] = None) -> None:
        """Fit scorer on reduced features."""
        pass
    
    @abstractmethod
    def score(self, Z: np.ndarray, X_original: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute anomaly scores (higher = more normal)."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get scorer state for serialization."""
        pass
    
    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load scorer state from serialization."""
        pass


# ============================================================================
# Concrete Feature Reducers
# ============================================================================

class PCAReducer(FeatureReducer):
    """PCA-based feature reduction."""
    
    def __init__(self, config: PCAConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.pca: Optional[PCA] = None
    
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and transform training data."""
        n_samples, n_features = X.shape
        n_components = max(1, min(n_samples - 1, n_features, self.config.n_components))
        
        self.logger.info(f"Fitting PCA with {n_components} components")
        self.pca = PCA(
            n_components=n_components,
            whiten=self.config.whiten,
            random_state=self.config.random_state,
        )
        return self.pca.fit_transform(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted PCA."""
        if self.pca is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")
        return self.pca.transform(X)
    
    def get_state(self) -> Dict[str, Any]:
        """Get PCA state."""
        return {
            "pca": self.pca,
            "config": asdict(self.config),
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load PCA state."""
        self.pca = state["pca"]
        self.config = PCAConfig(**state["config"])


class VAEReducer(FeatureReducer):
    """VAE-based feature reduction."""
    
    def __init__(self, config: VAEConfig, device: str, logger: logging.Logger):
        self.config = config
        self.device = device
        self.logger = logger
        self.vae: Optional[VAEonCLIP] = None
        self.history: Dict[str, List[float]] = {}
    
    def fit(self, X: np.ndarray, X_val: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit VAE and transform training data."""
        d_in = X.shape[1]
        d_lat = self.config.latent_dim or self.config.hidden_dim
        
        self.logger.info(f"Training VAE: d_in={d_in}, d_lat={d_lat}, d_hid={self.config.hidden_dim}")
        
        self.vae, self.history, _ = train_vae(
            X, X_val,
            d_in=d_in,
            d_lat=int(d_lat),
            d_hid=int(self.config.hidden_dim),
            dropout=float(self.config.dropout),
            epochs=int(self.config.epochs),
            batch_size=int(self.config.batch_size),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
            beta=float(self.config.beta),
            alpha=float(self.config.alpha),
            early_stopping=bool(self.config.early_stopping),
            patience=int(self.config.patience),
            device=self.device,
            seed=int(self.config.seed),
            verbose=True,
        )
        
        return vae_reduce_mu(self.vae, X).numpy()
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted VAE."""
        if self.vae is None:
            raise RuntimeError("VAE not fitted. Call fit() first.")
        return vae_reduce_mu(self.vae, X).numpy()
    
    def get_state(self) -> Dict[str, Any]:
        """Get VAE state."""
        state = {
            "config": asdict(self.config),
            "history": self.history,
            "vae": None,
        }
        if self.vae is not None:
            state["vae"] = {
                "state_dict": {k: v.cpu() for k, v in self.vae.state_dict().items()},
                "d_in": self.vae.enc[0].in_features,
                "d_lat": self.vae.mu.out_features,
                "d_hid": self.vae.enc[0].out_features,
                "dropout": self.config.dropout,
            }
        return state
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load VAE state."""
        self.config = VAEConfig(**state["config"])
        self.history = state.get("history", {})
        if state.get("vae"):
            vae_info = state["vae"]
            self.vae = VAEonCLIP(
                d_in=vae_info["d_in"],
                d_lat=vae_info["d_lat"],
                d_hid=vae_info["d_hid"],
                dropout=vae_info["dropout"],
            )
            self.vae.load_state_dict(vae_info["state_dict"])


# ============================================================================
# Concrete Anomaly Scorers
# ============================================================================

class GMMScorer(AnomalyScorer):
    """Gaussian Mixture Model-based anomaly scoring."""
    
    def __init__(self, config: GMMConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.gmm: Optional[GaussianMixture] = None
    
    def fit(self, Z: np.ndarray, X_original: Optional[np.ndarray] = None) -> None:
        """Fit GMM on reduced features."""
        best_gmm = None
        best_bic = np.inf
        
        for k in self.config.Ks:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.config.covariance_type,
                reg_covar=self.config.reg_covar,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
            )
            gmm.fit(Z)
            bic = gmm.bic(Z)
            self.logger.debug(f"k={k}: BIC={bic:.4f}")
            
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        
        self.gmm = best_gmm
        self.logger.info(f"Selected GMM with k={best_gmm.n_components}")
    
    def score(self, Z: np.ndarray, X_original: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute log-likelihood scores."""
        if self.gmm is None:
            raise RuntimeError("GMM not fitted. Call fit() first.")
        return self.gmm.score_samples(Z)
    
    def get_state(self) -> Dict[str, Any]:
        """Get GMM state."""
        return {
            "gmm": self.gmm,
            "config": asdict(self.config),
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load GMM state."""
        self.gmm = state["gmm"]
        self.config = GMMConfig(**state["config"])


class ReconstructionScorer(AnomalyScorer):
    """Reconstruction error-based anomaly scoring for VAE."""
    
    def __init__(self, config: VAEConfig, vae: Optional["VAEonCLIP"], logger: logging.Logger):
        self.config = config
        self.vae = vae
        self.logger = logger
        self._recon_inv_sigma2: Optional[np.ndarray] = None
        self._recon_const: float = 0.0
    
    def fit(self, Z: np.ndarray, X_original: np.ndarray) -> None:
        """Fit reconstruction variance on training data reconstructions."""
        if X_original is None or self.vae is None:
            self.logger.warning("X_original or VAE not provided; skipping reconstruction variance fitting")
            return
        
        # Compute reconstructions and residuals
        try:
            self.vae.eval()
            with torch.no_grad():
                X_torch = torch.from_numpy(X_original).float()
                # Move to same device as VAE
                device = next(self.vae.parameters()).device
                X_torch = X_torch.to(device)
                X_hat, _, _ = self.vae(X_torch)
                X_hat_np = X_hat.cpu().numpy()
            
            # Compute variance-aware log-likelihood parameters
            resid = X_hat_np - X_original
            sigma2 = resid.var(axis=0) + 1e-6
            self._recon_inv_sigma2 = 1.0 / sigma2
            self._recon_const = 0.5 * np.sum(np.log(2 * np.pi * sigma2))
            self.logger.info("Reconstruction variance fitted from training data")
        except Exception as e:
            self.logger.warning(f"Failed to compute reconstruction variance: {e}")
            self._recon_inv_sigma2 = None
            self._recon_const = 0.0
    
    def score(self, Z: np.ndarray, X_original: np.ndarray) -> np.ndarray:
        """Compute reconstruction log-likelihood scores."""
        if X_original is None or self.vae is None:
            self.logger.warning("Cannot compute reconstruction scores without original features and VAE")
            return np.zeros(len(Z))
        
        if self._recon_inv_sigma2 is None:
            self.logger.warning("Reconstruction variance not fitted; using simple MSE")
            return self._compute_simple_recon_score(X_original)
        
        try:
            self.vae.eval()
            with torch.no_grad():
                X_torch = torch.from_numpy(X_original).float()
                # Move to same device as VAE
                device = next(self.vae.parameters()).device
                X_torch = X_torch.to(device)
                X_hat, _, _ = self.vae(X_torch)
                X_hat_np = X_hat.cpu().numpy()
            
            # Variance-aware log-likelihood
            resid = X_hat_np - X_original
            scores = -(0.5 * (resid * resid * self._recon_inv_sigma2).sum(axis=1) + self._recon_const)
            return scores
        except Exception as e:
            self.logger.error(f"Error computing reconstruction scores: {e}")
            return self._compute_simple_recon_score(X_original)
    
    def _compute_simple_recon_score(self, X_original: np.ndarray) -> np.ndarray:
        """Fallback: simple MSE-based scoring."""
        try:
            self.vae.eval()
            with torch.no_grad():
                X_torch = torch.from_numpy(X_original).float()
                # Move to same device as VAE
                device = next(self.vae.parameters()).device
                X_torch = X_torch.to(device)
                X_hat, _, _ = self.vae(X_torch)
                X_hat_np = X_hat.cpu().numpy()
            return -np.mean((X_hat_np - X_original) ** 2, axis=1)
        except Exception:
            return np.zeros(len(X_original))
    
    def get_state(self) -> Dict[str, Any]:
        """Get reconstruction scorer state."""
        return {
            "recon_inv_sigma2": self._recon_inv_sigma2,
            "recon_const": self._recon_const,
            "config": asdict(self.config),
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load reconstruction scorer state."""
        self._recon_inv_sigma2 = state.get("recon_inv_sigma2")
        self._recon_const = state.get("recon_const", 0.0)
        self.config = VAEConfig(**state["config"])


class HybridScorer(AnomalyScorer):
    """Hybrid scoring combining GMM and reconstruction scores."""
    
    def __init__(
        self, 
        config: VAEConfig, 
        gmm_scorer: GMMScorer,
        recon_scorer: ReconstructionScorer,
        logger: logging.Logger
    ):
        self.config = config
        self.gmm_scorer = gmm_scorer
        self.recon_scorer = recon_scorer
        self.logger = logger
        
        # Normalization parameters (computed on validation set)
        self._gmm_mean: float = 0.0
        self._gmm_std: float = 1.0
        self._recon_mean: float = 0.0
        self._recon_std: float = 1.0
        
        # Learned classifier for hybrid scoring
        self._scaler: Optional[StandardScaler] = None
        self._classifier: Optional[LogisticRegression] = None
    
    def fit(self, Z: np.ndarray, X_original: np.ndarray) -> None:
        """
        Fit hybrid scorer: normalize scores and optionally train logistic regression.
        
        Args:
            Z: Reduced features
            X_original: Original features for reconstruction scoring
        """
        # Get GMM scores
        if self.gmm_scorer.gmm is None:
            self.logger.warning("GMM not fitted; cannot use hybrid scoring")
            return
        
        gmm_scores = self.gmm_scorer.score(Z, X_original)
        
        # Get reconstruction scores
        recon_scores = self.recon_scorer.score(Z, X_original)
        
        # Store normalization statistics
        eps = 1e-8
        self._gmm_mean = float(np.mean(gmm_scores))
        self._gmm_std = float(np.std(gmm_scores) + eps)
        self._recon_mean = float(np.mean(recon_scores))
        self._recon_std = float(np.std(recon_scores) + eps)
        
        self.logger.info(
            f"Hybrid scorer fitted: GMM μ={self._gmm_mean:.4f}±{self._gmm_std:.4f}, "
            f"Recon μ={self._recon_mean:.4f}±{self._recon_std:.4f}"
        )
        
        # Try to train learned hybrid classifier
        try:
            gmm_z = (gmm_scores - self._gmm_mean) / self._gmm_std
            recon_z = (recon_scores - self._recon_mean) / self._recon_std
            
            # Stack features (assume validation data is all normal, label=0)
            S = np.c_[gmm_z, recon_z]
            y = np.zeros(len(S), dtype=int)
            
            # Train logistic regression
            self._scaler = StandardScaler().fit(S)
            S_scaled = self._scaler.transform(S)
            
            self._classifier = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.config.seed
            )
            self._classifier.fit(S_scaled, y)
            
            self.logger.info("Hybrid logistic regression classifier trained successfully")
        except Exception as e:
            self.logger.warning(f"Failed to train hybrid classifier: {e}. Using weighted average fallback.")
            self._scaler = None
            self._classifier = None
    
    def score(self, Z: np.ndarray, X_original: np.ndarray) -> np.ndarray:
        """
        Compute hybrid anomaly scores.
        
        Returns:
            Normalized hybrid scores (higher = more normal)
        """
        # Get component scores
        gmm_scores = self.gmm_scorer.score(Z, X_original)
        recon_scores = self.recon_scorer.score(Z, X_original)
        
        # Normalize
        eps = 1e-8
        gmm_z = (gmm_scores - self._gmm_mean) / (self._gmm_std if self._gmm_std != 0 else 1.0)
        recon_z = (recon_scores - self._recon_mean) / (self._recon_std if self._recon_std != 0 else 1.0)
        
        # Use learned classifier if available
        if self._scaler is not None and self._classifier is not None:
            try:
                S = np.c_[gmm_z, recon_z]
                S_scaled = self._scaler.transform(S)
                proba = self._classifier.predict_proba(S_scaled)
                # Return probability of normal class (class 0)
                return proba[:, 0]
            except Exception as e:
                self.logger.warning(f"Error using hybrid classifier: {e}. Using weighted sum.")
        
        # Fallback: weighted sum
        w_gmm = self.config.hybrid_w_gmm
        w_recon = self.config.hybrid_w_recon
        total_weight = w_gmm + w_recon
        
        if total_weight > 0:
            scores = (w_gmm * gmm_z + w_recon * recon_z) / total_weight
        else:
            scores = 0.5 * gmm_z + 0.5 * recon_z
        
        return scores
    
    def get_state(self) -> Dict[str, Any]:
        """Get hybrid scorer state."""
        state = {
            "gmm_mean": self._gmm_mean,
            "gmm_std": self._gmm_std,
            "recon_mean": self._recon_mean,
            "recon_std": self._recon_std,
            "config": asdict(self.config),
            "scaler": self._scaler,
            "classifier": self._classifier,
        }
        return state
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load hybrid scorer state."""
        self._gmm_mean = state.get("gmm_mean", 0.0)
        self._gmm_std = state.get("gmm_std", 1.0)
        self._recon_mean = state.get("recon_mean", 0.0)
        self._recon_std = state.get("recon_std", 1.0)
        self._scaler = state.get("scaler")
        self._classifier = state.get("classifier")
        self.config = VAEConfig(**state["config"])


# ============================================================================
# Threshold Calibration
# ============================================================================

class ThresholdCalibrator:
    """Calibrate decision threshold for anomaly detection."""
    
    def __init__(self, config: ThresholdConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.tau: float = 0.0
    
    def calibrate(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> float:
        """
        Calibrate threshold using validation scores.
        
        Args:
            scores: Anomaly scores (higher = more normal)
            labels: Optional labels for F1-based calibration (0=normal, 1=anomalous)
        
        Returns:
            Optimal threshold tau
        """
        if len(scores) < 3:
            self.logger.warning("Very small validation set for threshold calibration")
            self.tau = float(np.min(scores)) if len(scores) > 0 else -1e9
            return self.tau
        
        if labels is not None and len(np.unique(labels)) > 1:
            # F1-based calibration
            self.tau = self._calibrate_by_f1(scores, labels)
        else:
            # Quantile-based calibration
            self.tau = float(np.quantile(scores, self.config.quantile))
        
        self.logger.info(f"Threshold calibrated: tau={self.tau:.6f}")
        return self.tau
    
    @staticmethod
    def _calibrate_by_f1(scores: np.ndarray, y: np.ndarray) -> float:
        """Find threshold maximizing F1 score."""
        qs = np.linspace(0.01, 0.99, 99)
        grid = np.quantile(scores, qs)
        best_f1 = -1
        best_tau = float(np.quantile(scores, 0.05))
        
        for t in grid:
            y_pred = (scores < t).astype(int)
            tp = np.sum((y == 1) & (y_pred == 1))
            fp = np.sum((y == 0) & (y_pred == 1))
            fn = np.sum((y == 1) & (y_pred == 0))
            
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            
            if f1 > best_f1:
                best_f1 = f1
                best_tau = t
        
        return float(best_tau)


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
    
    return logger


# ============================================================================
# Main Model Class (Simplified with Dependency Inversion)
# ============================================================================

class AnomalyDetectionModel:
    """
    Refactored anomaly detection model using SOLID principles.
    
    Responsibilities:
    - Orchestrate feature extraction, reduction, and scoring
    - Manage model lifecycle (fit, predict, save/load)
    - Provide clean inference API
    """
    
    def __init__(
        self,
        vision_encoder,
        preprocess,
        device: str,
        config: Optional[ModelConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize anomaly detection model.
        
        Args:
            vision_encoder: Pre-trained vision encoder for feature extraction
            preprocess: Preprocessing function for images
            device: Device for computation ('cpu', 'cuda', 'mps')
            config: ModelConfig instance (uses defaults if None)
            logger: logging.Logger instance (creates new if None)
        """
        self.vision_encoder = vision_encoder
        self.preprocess = preprocess
        self.device = device
        self.config = config or ModelConfig()
        self.logger = logger or setup_logger(__name__)
        
        # Validate configuration
        self.config.validate()
        
        # Initialize components
        self._init_components()
        
        self.logger.info("Model initialized successfully")
    
    def _init_components(self) -> None:
        """Initialize feature reducer, scorers, and calibrator."""
        if self.config.ft_reduction == "PCA":
            self.reducer: FeatureReducer = PCAReducer(self.config.pca, self.logger)
            self.scorer: AnomalyScorer = GMMScorer(self.config.gmm, self.logger)
        elif self.config.ft_reduction == "VAE":
            self.reducer = VAEReducer(self.config.vae, self.device, self.logger)
            
            # VAE reducer sets self.vae after fitting, so we handle scorer setup in fit()
            self.scorer = None  # Will be set based on score_mode in fit()
        else:
            raise ValueError(f"Unknown ft_reduction: {self.config.ft_reduction}")
        
        self.calibrator = ThresholdCalibrator(self.config.threshold, self.logger)
        self.threshold: float = 0.0
        
        # Store validation metrics for reproducibility
        self.val_metrics: Dict[str, float] = {}
    
    def fit(self, train_folder: Union[str, Path]) -> None:
        """
        Fit anomaly detection model on training data.
        
        Args:
            train_folder: Path to folder containing training images
        """
        self.logger.info(f"Starting model training on {train_folder}")
        train_folder = Path(train_folder)
        
        # Extract features
        X = self._embed_folder(train_folder)
        self.logger.info(f"Extracted {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        n = X.shape[0]
        perm = np.random.RandomState(self.config.pca.random_state).permutation(n)
        X = X[perm]
        
        split_idx = max(1, int(self.config.threshold.train_val_split * n))
        X_train, X_val = X[:split_idx], X[split_idx:]
        
        # Reduce features
        Z_train = self.reducer.fit(X_train)
        Z_val = self.reducer.transform(X_val) if len(X_val) > 0 else np.empty((0, Z_train.shape[1]))
        self.logger.info(f"Features reduced to dimension: {Z_train.shape[1]}")
        
        # Initialize scorer based on reduction method and score mode
        if self.config.ft_reduction == "PCA":
            # PCA path: use only GMM scoring
            self.scorer = GMMScorer(self.config.gmm, self.logger)
            self.scorer.fit(Z_train, X_train)
        
        elif self.config.ft_reduction == "VAE":
            # VAE path: support gmm_mu, recon_kl, and hybrid modes
            score_mode = self.config.vae.score_mode.lower()
            
            if score_mode == "gmm_mu":
                # GMM on latent space
                self.scorer = GMMScorer(self.config.gmm, self.logger)
                self.scorer.fit(Z_train, X_train)
            
            elif score_mode == "recon_kl":
                # Reconstruction error
                vae = self.reducer.vae  # Get the trained VAE from reducer
                self.scorer = ReconstructionScorer(self.config.vae, vae, self.logger)
                self.scorer.fit(Z_train, X_train)
            
            elif score_mode == "hybrid":
                # Combine GMM and reconstruction
                vae = self.reducer.vae
                gmm_scorer = GMMScorer(self.config.gmm, self.logger)
                gmm_scorer.fit(Z_train, X_train)
                
                recon_scorer = ReconstructionScorer(self.config.vae, vae, self.logger)
                recon_scorer.fit(Z_train, X_train)
                
                self.scorer = HybridScorer(self.config.vae, gmm_scorer, recon_scorer, self.logger)
                self.scorer.fit(Z_val, X_val)  # Fit on validation data for normalization
            
            else: 
                raise ValueError(f"Unknown score_mode: {score_mode}")
        
        # Compute validation scores
        if len(Z_val) > 0:
            val_scores = self.scorer.score(Z_val, X_val)
            self.threshold = self.calibrator.calibrate(val_scores)
            
            # Store validation metrics for reproducibility
            self.val_metrics = {
                "threshold": float(self.threshold),
                "val_scores_min": float(np.min(val_scores)),
                "val_scores_max": float(np.max(val_scores)),
                "val_scores_mean": float(np.mean(val_scores)),
                "val_scores_std": float(np.std(val_scores)),
                "val_count": len(val_scores),
            }
            self.logger.info(f"Validation metrics: {self.val_metrics}")
        else:
            self.logger.warning("No validation data for threshold calibration")
            self.threshold = float(np.min(self.scorer.score(Z_train, X_train)))
            self.val_metrics = {"threshold": float(self.threshold)}
        
        self.logger.info("Model training completed")
    
    def predict(self, image_path: Union[str, Path]) -> Tuple[bool, float]:
        """
        Predict if image is anomalous.
        
        Args:
            image_path: Path to image file
        
        Returns:
            (is_normal, anomaly_score): is_normal=True if authorized (normal),
                                       anomaly_score higher means more normal
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Ensure no gradients are computed
        with torch.no_grad():
            # Load and embed
            img = Image.open(image_path).convert("RGB")
            X = self._embed_image(img)
            
            # Reduce and score
            Z = self.reducer.transform(X)
            score = self.scorer.score(Z, X)[0]
        
        is_normal = score >= self.threshold
        return is_normal, score
    
    def test_folder(
        self,
        data_parent: Union[str, Path],
        output_folder: Union[str, Path] = "test_results",
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set with confusion matrix and metrics.
        
        Args:
            data_parent: Parent folder containing test/ subdirectory
                        (data_parent/test/normal/ and data_parent/test/anomalous/)
            output_folder: Output directory for visualizations
        
        Returns:
            Dictionary with comprehensive test results and metrics
        """
        data_parent = Path(data_parent)
        test_folder = data_parent / "test"
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Collect test data
        normal_folder = test_folder / "normal"
        anomalous_folder = test_folder / "anomalous"
        
        if not (normal_folder.exists() and anomalous_folder.exists()):
            raise FileNotFoundError(f"Expected /normal and /anomalous in {test_folder}")
        
        image_data = []
        for ext in self.config.file_extensions:
            for img_path in sorted(normal_folder.glob(ext)):
                image_data.append((img_path, 0))
            for img_path in sorted(anomalous_folder.glob(ext)):
                image_data.append((img_path, 1))
        
        if not image_data:
            raise FileNotFoundError(f"No images found in {test_folder}")
        
        # Process images
        results = {}
        images, verdicts, scores, labels = [], [], [], []
        tp = fp = fn = tn = 0
        true_labels_all, pred_labels_all = [], []
        
        for img_path, true_label in tqdm(image_data, desc="Testing images", unit="img"):
            try:
                is_normal, score = self.predict(img_path)
                pred_label = 0 if is_normal else 1
                
                is_correct = (is_normal == (true_label == 0))
                
                # Update confusion matrix
                if true_label == 1:
                    tp += (1 if not is_normal else 0)
                    fn += (1 if is_normal else 0)
                else:
                    tn += (1 if is_normal else 0)
                    fp += (1 if not is_normal else 0)
                
                true_labels_all.append(true_label)
                pred_labels_all.append(pred_label)
                
                # Store results
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                verdicts.append("✓ CORRECT" if is_correct else "✗ WRONG")
                scores.append(score)
                labels.append("NORMAL" if true_label == 0 else "ANOMALOUS")
                
                results[str(img_path)] = {
                    "label": true_label,
                    "label_str": "NORMAL" if true_label == 0 else "ANOMALOUS",
                    "authorized": is_normal,
                    "score": score,
                    "threshold": self.threshold,
                }
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {e}")
        
        # Compute metrics
        total = len(image_data)
        correct = tp + tn
        accuracy = 100 * correct / total if total > 0 else 0
        precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Visualizations
        self._save_test_visualizations(images, verdicts, scores, labels, output_folder)
        self._save_metrics_report(
            tp, fp, fn, tn, accuracy, precision, recall, f1, output_folder
        )
        
        # Summary
        results["_summary"] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "metrics": {"precision": precision, "recall": recall, "f1": f1},
            "threshold": self.threshold,
        }
        
        self.logger.info(f"Test completed: Accuracy={accuracy:.2f}%, F1={f1:.4f}")
        return results
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": asdict(self.config),
            "reducer": self.reducer.get_state(),
            "scorer": self.scorer.get_state(),
            "threshold": self.threshold,
            "val_metrics": self.val_metrics,  # ← Save validation metrics
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load(
        path: Union[str, Path],
        vision_encoder,
        preprocess,
        device: str,
        logger: Optional[logging.Logger] = None,
    ) -> "AnomalyDetectionModel":
        """Load model from disk."""
        logger = logger or setup_logger(__name__)
        path = Path(path)
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Reconstruct nested dataclass objects from dicts
        config_dict = state["config"]
        pca_config = PCAConfig(**config_dict.get("pca", {}))
        vae_config = VAEConfig(**config_dict.get("vae", {}))
        gmm_config = GMMConfig(**config_dict.get("gmm", {}))
        threshold_config = ThresholdConfig(**config_dict.get("threshold", {}))
        
        # Create ModelConfig with reconstructed sub-configs
        config = ModelConfig(
            ft_reduction=config_dict.get("ft_reduction", "PCA"),
            pca=pca_config,
            vae=vae_config,
            gmm=gmm_config,
            threshold=threshold_config,
            file_extensions=tuple(config_dict.get("file_extensions", ("*.png", "*.jpg", "*.jpeg"))),
            test_grid_cols=config_dict.get("test_grid_cols", 5),
            run_timestamp=config_dict.get("run_timestamp"),
            runs_root=config_dict.get("runs_root", "training_runs"),
            save_vae_training=config_dict.get("save_vae_training", True),
        )
        
        model = AnomalyDetectionModel(vision_encoder, preprocess, device, config, logger)
        
        # Load reducer first
        model.reducer.load_state(state["reducer"])
        
        # Reconstruct scorer based on config
        scorer_state = state["scorer"]
        if config.ft_reduction == "PCA":
            model.scorer = GMMScorer(config.gmm, logger)
            model.scorer.load_state(scorer_state)
        
        elif config.ft_reduction == "VAE":
            score_mode = config.vae.score_mode.lower()
            vae = model.reducer.vae  # Get loaded VAE
            
            if score_mode == "gmm_mu":
                model.scorer = GMMScorer(config.gmm, logger)
                model.scorer.load_state(scorer_state)
            
            elif score_mode == "recon_kl":
                model.scorer = ReconstructionScorer(config.vae, vae, logger)
                model.scorer.load_state(scorer_state)
            
            elif score_mode == "hybrid":
                # Reconstruct GMM and reconstruction scorers for hybrid
                gmm_scorer = GMMScorer(config.gmm, logger)
                recon_scorer = ReconstructionScorer(config.vae, vae, logger)
                model.scorer = HybridScorer(config.vae, gmm_scorer, recon_scorer, logger)
                model.scorer.load_state(scorer_state)
        
        model.threshold = state["threshold"]
        model.val_metrics = state.get("val_metrics", {})  # ← Restore validation metrics
        
        logger.info(f"Model loaded from {path}")
        return model
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _embed_folder(self, folder: Path) -> np.ndarray:
        """Extract features from all images in folder."""
        features = []
        for ext in self.config.file_extensions:
            for img_path in sorted(folder.glob(ext)):
                try:
                    img = Image.open(img_path).convert("RGB")
                    feat = self._embed_image(img)
                    features.append(feat)
                except Exception as e:
                    self.logger.warning(f"Failed to embed {img_path}: {e}")
        
        if not features:
            return np.empty((0, 512), dtype=np.float32)  # Default to CLIP dim
        return np.vstack(features)
    
    def _embed_image(self, img: Image.Image) -> np.ndarray:
        """Extract features from single image."""
        self.vision_encoder.eval()  # ← Put CLIP in eval mode (disable dropout/batch norm)
        with torch.no_grad():
            x = self.preprocess(img).unsqueeze(0).to(self.device)
            # Ensure vision_encoder is on the right device and use deterministic inference
            feat = self.vision_encoder.to(self.device)(x).float().cpu().detach().numpy()
        return feat
    
    def _save_test_visualizations(
        self,
        images: List[Image.Image],
        verdicts: List[str],
        scores: List[float],
        labels: List[str],
        output_folder: Path,
    ) -> None:
        """Save test result visualizations."""
        if not images:
            return
        
        n = len(images)
        cols = self.config.test_grid_cols
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (ax, img, verdict, score, label) in enumerate(
            zip(axes, images, verdicts, scores, labels)
        ):
            ax.imshow(img)
            ax.axis("off")
            color = "green" if "✓" in verdict else "red"
            text = f"{verdict}\n{label}\nScore: {score:.4f}\nτ: {self.threshold:.4f}"
            ax.text(
                0.5, -0.05, text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="center",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
            )
        
        for idx in range(n, len(axes)):
            axes[idx].axis("off")
        
        plt.tight_layout()
        viz_path = output_folder / "test_results_images.png"
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Visualizations saved to {viz_path}")
    
    def _save_metrics_report(
        self,
        tp: int, fp: int, fn: int, tn: int,
        accuracy: float, precision: float, recall: float, f1: float,
        output_folder: Path,
    ) -> None:
        """Save metrics report with confusion matrix."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'],
            cbar_kws={'label': 'Count'},
        )
        axes[0].set_xlabel('Predicted', fontweight='bold')
        axes[0].set_ylabel('True', fontweight='bold')
        axes[0].set_title('Confusion Matrix', fontweight='bold')
        
        # Metrics
        metrics_text = f"""
METRICS

Accuracy: {accuracy:.2f}%
Precision: {precision:.2f}%
Recall: {recall:.2f}%
F1 Score: {f1:.4f}

TP: {tp}  FP: {fp}
FN: {fn}  TN: {tn}

Threshold: {self.threshold:.6f}
        """
        axes[1].text(
            0.1, 0.5, metrics_text,
            fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )
        axes[1].axis("off")
        
        plt.tight_layout()
        report_path = output_folder / "metrics_report.png"
        plt.savefig(report_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Metrics report saved to {report_path}")


# Backward compatibility alias
Model = AnomalyDetectionModel
