"""
Main Anomaly Detection Model Class

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
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, List, Any
from PIL import Image
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from .configs import ModelConfig, PCAConfig, VAEConfig, GMMConfig, ThresholdConfig
from .reducers import FeatureReducer, PCAReducer, VAEReducer
from .scorers import AnomalyScorer, GMMScorer, ReconstructionScorer, HybridScorer
from .threshold import ThresholdCalibrator
from .utils import setup_logger, save_test_visualizations, save_metrics_report, plot_model_2d_visualization


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
        save_test_visualizations(
            images, verdicts, scores, labels, output_folder, 
            self.threshold, self.config.test_grid_cols, self.logger
        )
        save_metrics_report(
            tp, fp, fn, tn, accuracy, precision, recall, f1, 
            self.threshold, output_folder, self.logger
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
        
        from dataclasses import asdict
        
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
    
    # ========================================================================
    # Visualization Methods
    # ========================================================================
    
    def plot_with_model_2d(
        self,
        normal_paths: Union[str, Path, List[Union[str, Path]]],
        anomaly_folders: Optional[List[Union[str, Path]]] = None,
        batch_size: int = 64,
        reduction_method: str = "pca",
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Visualize model predictions in 2D space with decision boundaries.
        
        Works with both PCA and VAE feature reduction:
        - If ft_reduction="PCA": uses fitted PCA to project CLIP features directly
        - If ft_reduction="VAE": uses VAE to reduce to latent space, then projects to 2D
        
        Args:
            normal_paths: Path(s) to normal images (string, Path, or list)
            anomaly_folders: List of paths to anomaly folders
            batch_size: Batch size for encoding (default 64)
            reduction_method: '2d' reduction method ('pca' or 'umap'; default 'pca')
            figsize: Figure size (default (14, 10))
        
        Returns:
            matplotlib Figure object
        
        Example:
            >>> model.plot_with_model_2d(
            ...     normal_paths="path/to/normal",
            ...     anomaly_folders=["path/to/anomaly1", "path/to/anomaly2"],
            ...     reduction_method="pca"
            ... )
            >>> plt.show()
        """
        def collect(paths_or_root):
            """Collect image paths from input."""
            if isinstance(paths_or_root, (list, tuple)):
                return [Path(p) for p in paths_or_root]
            root = Path(paths_or_root)
            exts = ("*.jpg", "*.jpeg", "*.png")
            return [p for ext in exts for p in root.rglob(ext)]
        
        def encode(paths: List[Path]) -> Tuple[np.ndarray, List[Path]]:
            """Encode images to CLIP features."""
            feats, keep = [], []
            with torch.no_grad():
                batch, stash = [], []
                for p in paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        batch.append(self.preprocess(img))
                        stash.append(p)
                    except Exception:
                        continue
                    if len(batch) == batch_size:
                        x = torch.stack(batch).to(self.device)
                        f = self.vision_encoder(x).float()
                        feats.append(f.cpu())
                        keep += stash
                        batch, stash = [], []
                if batch:
                    x = torch.stack(batch).to(self.device)
                    f = self.vision_encoder(x).float()
                    feats.append(f.cpu())
                    keep += stash
            
            X = np.concatenate([t.numpy() for t in feats], axis=0) if feats else np.empty((0, 512))
            return X, keep
        
        # Encode all image sets
        self.logger.info("Encoding normal images...")
        Xn, paths_n = encode(collect(normal_paths))
        self.logger.info(f"Encoded {len(paths_n)} normal images")
        
        an_sets = []
        anomaly_folders = anomaly_folders or []
        for folder in anomaly_folders:
            self.logger.info(f"Encoding anomaly folder: {folder}")
            Xa, pa = encode(collect(folder))
            an_sets.append((str(folder), Xa, pa))
            self.logger.info(f"Encoded {len(pa)} anomaly images from {folder}")
        
        # Prepare data for visualization
        normal_data = (Xn, paths_n)
        anomaly_data = [(name, Xa, pa) for name, Xa, pa in an_sets]
        
        # Use utility function to create visualization
        fig = plot_model_2d_visualization(
            reducer=self.reducer,
            scorer=self.scorer,
            threshold=self.threshold,
            normal_data=normal_data,
            anomaly_data=anomaly_data,
            ft_reduction=self.config.ft_reduction,
            reduction_method=reduction_method,
            figsize=figsize,
            logger=self.logger,
        )
        
        return fig
    
    def plot_with_model_pca_2d(
        self,
        normal_paths: Union[str, Path, List[Union[str, Path]]],
        anomaly_folders: Optional[List[Union[str, Path]]] = None,
        batch_size: int = 64,
    ) -> plt.Figure:
        """
        Backward compatibility wrapper for plot_with_model_2d() using PCA reduction.
        
        Use plot_with_model_2d() with reduction_method="pca" instead.
        
        Args:
            normal_paths: Path(s) to normal images
            anomaly_folders: List of paths to anomaly folders
            batch_size: Batch size for encoding (default 64)
        
        Returns:
            matplotlib Figure object
        """
        return self.plot_with_model_2d(
            normal_paths=normal_paths,
            anomaly_folders=anomaly_folders,
            batch_size=batch_size,
            reduction_method="pca"
        )


# Backward compatibility alias
Model = AnomalyDetectionModel
