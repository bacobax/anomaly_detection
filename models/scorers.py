"""
Anomaly scoring strategies for anomaly detection.

Implements abstract base class and concrete scoring methods (GMM, Reconstruction, Hybrid).
Follows Strategy pattern for extensibility.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Optional, Dict, Any
import logging
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .configs import GMMConfig, VAEConfig
from .VAE import VAEonCLIP


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
    
    def __init__(self, config: VAEConfig, vae: Optional[VAEonCLIP], logger: logging.Logger):
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
