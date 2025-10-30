"""
Threshold calibration for anomaly detection.

Provides utilities for calibrating decision thresholds using validation data.
"""

import numpy as np
import logging
from typing import Optional

from .configs import ThresholdConfig


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
