"""
Feature reduction strategies for anomaly detection.

Implements abstract base class and concrete feature reduction methods (PCA, VAE).
Follows Strategy pattern for extensibility.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Optional, Dict, Any
import logging
from sklearn.decomposition import PCA

from .configs import PCAConfig, VAEConfig
from .VAE import VAEonCLIP, train_vae, vae_reduce_mu


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
        self.history: Dict[str, list] = {}
    
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
