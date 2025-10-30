"""
Refactored Training Script with SOLID Principles

SOLID Principles Applied:
- Single Responsibility: Separate concerns into configuration builders, model trainers, and results handlers
- Open/Closed: Use abstract base classes for extensibility
- Liskov Substitution: Configuration objects are interchangeable
- Interface Segregation: Specific interfaces for different responsibilities
- Dependency Inversion: Depend on abstractions (TrainingStrategy) not concrete implementations

ML Best Practices:
- Configuration management through builder pattern
- Reproducibility through timestamp-based organization
- Metrics tracking and comparison
- Comprehensive logging and error handling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import csv
import yaml
import logging
from models.model import (
    AnomalyDetectionModel,
    ModelConfig,
    PCAConfig,
    VAEConfig,
    GMMConfig,
    ThresholdConfig,
)


# ============================================================================
# Setup
# ============================================================================

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# ============================================================================
# Vision Encoder (Wrapper for CLIP)
# ============================================================================

class VisualEncoder(nn.Module):
    """CLIP visual encoder wrapper with optional normalization."""
    
    def __init__(self, visual, normalize: bool = True):
        super().__init__()
        self.visual = visual
        self.normalize = normalize
    
    def forward(self, x):
        x = self.visual(x)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x


# ============================================================================
# Configuration Builders (Builder Pattern for SOLID)
# ============================================================================

@dataclass
class ModelConfigBuilder:
    """
    Builder for creating ModelConfig instances from dict parameters.
    
    Converts flat hyperparameter dictionaries to structured ModelConfig objects.
    """
    
    @staticmethod
    def from_dict(params: Dict[str, Any]) -> ModelConfig:
        """
        Build ModelConfig from flat hyperparameter dictionary.
        
        Args:
            params: Dictionary containing model hyperparameters
        
        Returns:
            Structured ModelConfig instance
        """
        # Extract feature reduction method
        ft_reduction = params.get("ft_reduction", "PCA").upper()
        
        # Build PCA config
        pca_config = PCAConfig(
            n_components=params.get("pca_n_components", 20),
            whiten=params.get("pca_whiten", True),
            random_state=params.get("pca_random_state", 42),
        )
        
        # Build VAE config
        vae_config = VAEConfig(
            latent_dim=params.get("vae_latent_dim"),
            hidden_dim=params.get("vae_hidden_dim", 512),
            dropout=params.get("vae_dropout", 0.2),
            epochs=params.get("vae_epochs", 30),
            batch_size=params.get("vae_batch_size", 128),
            lr=params.get("vae_lr", 1e-3),
            weight_decay=params.get("vae_weight_decay", 0.0),
            beta=params.get("vae_beta", 2.0),
            alpha=params.get("vae_alpha", 0.5),
            early_stopping=params.get("vae_early_stopping", True),
            patience=params.get("vae_patience", 5),
            seed=params.get("vae_seed", 42),
            score_mode=params.get("vae_score_mode", "gmm_mu"),
            hybrid_w_gmm=params.get("vae_hybrid_w_gmm", 0.5),
            hybrid_w_recon=params.get("vae_hybrid_w_recon", 0.5),
            finetune_encoder_layers=params.get("vae_finetune_encoder_layers", 0),
            encoder_lr=params.get("vae_encoder_lr", 3e-5),
        )
        
        # Build GMM config
        gmm_Ks = params.get("gmm_Ks", (1, 2, 3, 4, 5))
        if isinstance(gmm_Ks, list):
            gmm_Ks = tuple(gmm_Ks)
        
        gmm_config = GMMConfig(
            Ks=gmm_Ks,
            covariance_type=params.get("gmm_covariance_type", "full"),
            reg_covar=params.get("gmm_reg_covar", 1e-5),
            max_iter=params.get("gmm_max_iter", 100),
            random_state=params.get("gmm_random_state", 42),
        )
        
        # Build threshold config
        threshold_config = ThresholdConfig(
            train_val_split=params.get("train_val_split", 0.95),
            quantile=params.get("threshold_quantile", 0.05),
        )
        
        # Build complete model config
        model_config = ModelConfig(
            ft_reduction=ft_reduction,
            pca=pca_config,
            vae=vae_config,
            gmm=gmm_config,
            threshold=threshold_config,
            file_extensions=tuple(params.get("file_extensions", ("*.png", "*.jpg", "*.jpeg"))),
            test_grid_cols=params.get("test_grid_cols", 5),
            run_timestamp=params.get("run_timestamp"),
            runs_root=params.get("runs_root", "training_runs"),
            save_vae_training=params.get("save_vae_training", True),
        )
        
        return model_config


# ============================================================================
# Model Trainer (Strategy Pattern)
# ============================================================================

@dataclass
class TrainingResult:
    """Result of a single model training run."""
    config_name: str
    model: AnomalyDetectionModel
    model_path: Path
    test_results: Dict[str, Any]
    metrics: Dict[str, float]
    training_time: float


class ModelTrainer:
    """Orchestrates model training workflow."""
    
    def __init__(self, encoder, preprocess, device: str, logger: logging.Logger):
        self.encoder = encoder
        self.preprocess = preprocess
        self.device = device
        self.logger = logger
    
    def train(
        self,
        config_name: str,
        model_config: ModelConfig,
        data_parent: str,
        model_output_path: Path,
        test_output_folder: Path,
    ) -> TrainingResult:
        """
        Train a single model.
        
        Args:
            config_name: Name of configuration
            model_config: ModelConfig instance
            train_folder: Path to training data
            data_parent: Parent folder for test data (contains test/normal and test/anomalous)
            model_output_path: Where to save trained model
            test_output_folder: Where to save test results
        
        Returns:
            TrainingResult with metrics and model
        """
        self.logger.info(f"Starting training for config: {config_name}")
        start_time = datetime.now()
        train_folder = f"{data_parent}/train"
        
        try:
            # Initialize model
            model = AnomalyDetectionModel(
                vision_encoder=self.encoder,
                preprocess=self.preprocess,
                device=self.device,
                config=model_config,
                logger=self.logger,
            )
            
            # Train
            self.logger.info(f"Fitting model on {train_folder}...")
            model.fit(train_folder)
            
            # Save
            model_output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(model_output_path)
            self.logger.info(f"Model saved to {model_output_path}")
            
            # Test
            self.logger.info(f"Testing model on {data_parent}...")
            test_output_folder.mkdir(parents=True, exist_ok=True)
            test_results = model.test_folder(data_parent, test_output_folder)
            
            # Extract metrics
            metrics = self._extract_metrics(test_results)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"✓ Training completed for {config_name} in {elapsed:.1f}s")
            
            return TrainingResult(
                config_name=config_name,
                model=model,
                model_path=model_output_path,
                test_results=test_results,
                metrics=metrics,
                training_time=elapsed,
            )
        
        except Exception as e:
            self.logger.error(f"Failed to train {config_name}: {e}")
            raise
    
    @staticmethod
    def _extract_metrics(test_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics from test results."""
        if "_summary" not in test_results:
            return {}
        
        summary = test_results["_summary"]
        return {
            "accuracy": summary.get("accuracy", 0.0),
            "precision": summary.get("metrics", {}).get("precision", 0.0),
            "recall": summary.get("metrics", {}).get("recall", 0.0),
            "f1": summary.get("metrics", {}).get("f1", 0.0),
        }


# ============================================================================
# Results Handler (Reporting and Persistence)
# ============================================================================

class ResultsHandler:
    """Handles results aggregation, reporting, and persistence."""
    
    def __init__(self, output_folder: Path, logger: logging.Logger):
        self.output_folder = Path(output_folder)
        self.logger = logger
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results: List[TrainingResult], configs: List[Dict]) -> None:
        """
        Save training results to multiple formats.
        
        Args:
            results: List of TrainingResult objects
            configs: List of original config dictionaries
        """
        # Save metrics CSV
        self._save_metrics_csv(results)
        
        # Save config YAML files
        self._save_config_yamls(configs)
        
        # Print summary
        self._print_summary(results)
        
        self.logger.info(f"Results saved to {self.output_folder}")
    
    def _save_metrics_csv(self, results: List[TrainingResult]) -> None:
        """Save metrics summary to CSV."""
        csv_path = self.output_folder / "models_metrics_summary.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['config_name', 'accuracy', 'precision', 'recall', 'f1', 'training_time']
            )
            writer.writeheader()
            
            for result in results:
                row = {
                    'config_name': result.config_name,
                    'accuracy': result.metrics.get('accuracy', 0.0),
                    'precision': result.metrics.get('precision', 0.0),
                    'recall': result.metrics.get('recall', 0.0),
                    'f1': result.metrics.get('f1', 0.0),
                    'training_time': result.training_time,
                }
                writer.writerow(row)
        
        self.logger.info(f"Metrics saved to {csv_path}")
    
    def _save_config_yamls(self, configs: List[Dict]) -> None:
        """Save configuration YAML files."""
        for config in configs:
            yaml_path = self.output_folder / f"{config['name']}_config.yaml"
            
            config_data = {
                'name': config['name'],
                'filename': config['filename'],
                'parameters': config['params'],
            }
            
            with open(yaml_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"Configuration saved to {yaml_path}")
    
    def _print_summary(self, results: List[TrainingResult]) -> None:
        """Print results summary to stdout."""
        print(f"\n{'='*90}")
        print("TRAINING SUMMARY - MODELS PERFORMANCE COMPARISON")
        print(f"{'='*90}")
        
        header = f"{'Config Name':<25} {'Accuracy (%)':<15} {'Precision (%)':<15} {'Recall (%)':<15} {'F1 Score':<12} {'Time (s)':<10}"
        print(header)
        print("-" * 90)
        
        for result in results:
            row = (
                f"{result.config_name:<25} "
                f"{result.metrics.get('accuracy', 0.0):<15.2f} "
                f"{result.metrics.get('precision', 0.0):<15.2f} "
                f"{result.metrics.get('recall', 0.0):<15.2f} "
                f"{result.metrics.get('f1', 0.0):<12.4f} "
                f"{result.training_time:<10.1f}"
            )
            print(row)
        
        print("-" * 90)
        print(f"{'='*90}\n")


# ============================================================================
# Configuration Definitions (Predefined Model Configurations)
# ============================================================================

class ConfigurationFactory:
    """Factory for creating model configurations."""
    
    @staticmethod
    def create_high_sensitivity_config() -> Dict[str, Any]:
        """
        High-sensitivity model: Sensitive to anomalies, lower false negatives.
        
        Best for: Applications where missing anomalies is costly (safety-critical).
        """
        return {
            "name": "high_sensitivity",
            "filename": "./weights/model_high_sensitivity.pth",
            "params": {
                "ft_reduction": "PCA",
                "pca_n_components": 128,
                "pca_whiten": True,
                "pca_random_state": 42,
                "gmm_Ks": (2, 3, 4, 5, 6),
                "gmm_covariance_type": "full",
                "gmm_reg_covar": 1e-5,
                "gmm_max_iter": 150,
                "gmm_random_state": 42,
                "train_val_split": 0.90,
                "threshold_quantile": 0.08,
            }
        }
    
    @staticmethod
    def create_balanced_pca_config() -> Dict[str, Any]:
        """
        Balanced PCA model: Trade-off between sensitivity and specificity.
        
        Best for: General-purpose anomaly detection.
        """
        return {
            "name": "balanced",
            "filename": "./weights/model_balanced.pth",
            "params": {
                "ft_reduction": "PCA",
                "pca_n_components": 64,
                "pca_whiten": True,
                "pca_random_state": 42,
                "gmm_Ks": (2, 3, 4, 5),
                "gmm_covariance_type": "full",
                "gmm_reg_covar": 1e-5,
                "gmm_max_iter": 100,
                "gmm_random_state": 42,
                "train_val_split": 0.90,
                "threshold_quantile": 0.05,
            }
        }
    
    @staticmethod
    def create_fast_robust_config() -> Dict[str, Any]:
        """
        Fast/Robust model: Faster training, lower false positives.
        
        Best for: Real-time applications with resource constraints.
        """
        return {
            "name": "fast_robust",
            "filename": "./weights/model_fast_robust.pth",
            "params": {
                "ft_reduction": "PCA",
                "pca_n_components": 32,
                "pca_whiten": True,
                "pca_random_state": 42,
                "gmm_Ks": (2, 3, 4),
                "gmm_covariance_type": "diag",
                "gmm_reg_covar": 1e-4,
                "gmm_max_iter": 80,
                "gmm_random_state": 42,
                "train_val_split": 0.90,
                "threshold_quantile": 0.08,
            }
        }
    
    @staticmethod
    def create_vae_gmm_mu_config() -> Dict[str, Any]:
        """
        VAE with GMM-Mu scoring: Structure-based anomaly detection.
        
        Best for: Detecting structural/layout anomalies.
        """
        return {
            "name": "VAE_GMM_mu_score",
            "filename": "./weights/model_VAE_GMM_mu_score.pth",
            "params": {
                "ft_reduction": "VAE",
                "vae_score_mode": "gmm_mu",
                "vae_latent_dim": 64,
                "vae_hidden_dim": 512,
                "vae_dropout": 0.2,
                "vae_epochs": 100,
                "vae_batch_size": 128,
                "vae_lr": 1e-3,
                "vae_weight_decay": 0.0,
                "vae_beta": 4.0,
                "vae_alpha": 0.5,
                "vae_early_stopping": True,
                "vae_patience": 5,
                "vae_seed": 42,
                "pca_whiten": True,
                "pca_random_state": 42,
                "gmm_Ks": (2, 3, 4, 5),
                "gmm_covariance_type": "full",
                "gmm_reg_covar": 1e-5,
                "gmm_max_iter": 100,
                "gmm_random_state": 42,
                "train_val_split": 0.90,
                "threshold_quantile": 0.25,
            }
        }
    
    @staticmethod
    def create_vae_recon_kl_config() -> Dict[str, Any]:
        """
        VAE with Recon-KL scoring: Texture/detail-based anomaly detection.
        
        Best for: Detecting pixel-level/texture anomalies.
        """
        return {
            "name": "VAE_recon_kl_score",
            "filename": "./weights/model_VAE_recon_kl_score.pth",
            "params": {
                "ft_reduction": "VAE",
                "vae_score_mode": "recon_kl",
                "vae_latent_dim": 64,
                "vae_hidden_dim": 512,
                "vae_dropout": 0.2,
                "vae_epochs": 100,
                "vae_batch_size": 128,
                "vae_lr": 1e-3,
                "vae_weight_decay": 0.0,
                "vae_beta": 4.0,
                "vae_alpha": 0.5,
                "vae_early_stopping": True,
                "vae_patience": 5,
                "vae_seed": 42,
                "pca_whiten": True,
                "pca_random_state": 42,
                "gmm_Ks": (2, 3, 4, 5),
                "gmm_covariance_type": "full",
                "gmm_reg_covar": 1e-5,
                "gmm_max_iter": 100,
                "gmm_random_state": 42,
                "train_val_split": 0.9,
                "threshold_quantile": 0.25,
            }
        }
    
    @staticmethod
    def create_vae_hybrid_config() -> Dict[str, Any]:
        """
        VAE with Hybrid scoring: Best overall balance (structure + texture).
        
        Best for: Comprehensive anomaly detection requiring both structure and texture analysis.
        """
        return {
            "name": "VAE_hybrid_score",
            "filename": "./weights/model_VAE_hybrid_score.pth",
            "params": {
                "ft_reduction": "VAE",
                "vae_score_mode": "hybrid",
                "vae_latent_dim": 64,
                "vae_hidden_dim": 512,
                "vae_dropout": 0.2,
                "vae_epochs": 100,
                "vae_batch_size": 128,
                "vae_lr": 1e-3,
                "vae_weight_decay": 0.0,
                "vae_beta": 4.0,
                "vae_alpha": 0.5,
                "vae_early_stopping": True,
                "vae_patience": 5,
                "vae_seed": 42,
                "vae_hybrid_w_gmm": 0.1,
                "vae_hybrid_w_recon": 0.9,
                "pca_whiten": True,
                "pca_random_state": 42,
                "gmm_Ks": (1, 2, 3, 4),
                "gmm_covariance_type": "diag",
                "gmm_reg_covar": 1e-5,
                "gmm_max_iter": 100,
                "gmm_random_state": 42,
                "train_val_split": 0.90,
                "threshold_quantile": 0.18,
            }
        }
    
    @staticmethod
    def create_vae_recon_kl_finetuned_config() -> Dict[str, Any]:
        """
        VAE with Recon-KL scoring and encoder finetuning: Advanced texture detection.
        
        Best for: Fine-grained texture anomaly detection with encoder adaptation.
        """
        return {
            "name": "VAE_recon_kl_score_enc_finetuned",
            "filename": "./weights/model_VAE_recon_kl_score_enc_finetuned.pth",
            "params": {
                "ft_reduction": "VAE",
                "vae_score_mode": "recon_kl",
                "vae_latent_dim": 64,
                "vae_hidden_dim": 512,
                "vae_dropout": 0.2,
                "vae_epochs": 100,
                "vae_batch_size": 128,
                "vae_lr": 1e-3,
                "vae_weight_decay": 0.0,
                "vae_beta": 6.0,
                "vae_alpha": 0.5,
                "vae_early_stopping": True,
                "vae_patience": 5,
                "vae_seed": 42,
                "vae_finetune_encoder_layers": 2,
                "vae_encoder_lr": 3e-5,
                "pca_whiten": True,
                "pca_random_state": 42,
                "gmm_Ks": (2, 3, 4, 5),
                "gmm_covariance_type": "full",
                "gmm_reg_covar": 1e-5,
                "gmm_max_iter": 100,
                "gmm_random_state": 42,
                "train_val_split": 0.90,
                "threshold_quantile": 0.2,
            }
        }


# ============================================================================
# Main Training Orchestrator
# ============================================================================

class TrainingOrchestrator:
    """Orchestrates complete training pipeline."""
    
    def __init__(self, device: str, logger: logging.Logger):
        self.device = device
        self.logger = logger
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run(
        self,
        configs: List[Dict[str, Any]],
        data_parent: str = "trento_house_data",
        weights_folder: str = "./weights",
    ) -> List[TrainingResult]:
        """
        Run complete training pipeline for multiple configurations.
        
        Args:
            configs: List of configuration dictionaries
            data_parent: Parent folder containing train/ and test/ subdirectories
                        (data_parent/train/ for training data)
                        (data_parent/test/normal/ and data_parent/test/anomalous/ for test data)
            weights_folder: Where to save trained models
        
        Returns:
            List of TrainingResult objects
        """
        # Construct internal paths from data_parent
        data_parent_path = Path(data_parent)
        train_folder = str(data_parent_path / "train")
        
        self.logger.info(f"Starting training pipeline with {len(configs)} configurations")
        self.logger.info(f"Timestamp: {self.timestamp}")
        self.logger.info(f"Data parent: {data_parent}")
        self.logger.info(f"Training data: {train_folder}")
        self.logger.info(f"Test data: {data_parent}/{{'test/normal', 'test/anomalous'}}")
        
        # Initialize encoder
        self.logger.info("Initializing CLIP encoder...")
        clip_model, preprocess, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        clip_model = clip_model.to(self.device).eval()
        encoder = VisualEncoder(clip_model.visual, normalize=False).eval()
        
        # Initialize trainer
        trainer = ModelTrainer(encoder, preprocess, self.device, self.logger)
        
        # Train models
        results = []
        for config in configs:
            config['params']['run_timestamp'] = self.timestamp
            
            model_config = ModelConfigBuilder.from_dict(config['params'])
            
            model_path = Path(config['filename'])
            test_output = Path("runs") / f"test_results_{config['name']}"
            
            result = trainer.train(
                config_name=config['name'],
                model_config=model_config,
                data_parent=data_parent,
                model_output_path=model_path,
                test_output_folder=test_output,
            )
            
            results.append(result)
        
        # Save results
        output_folder = Path("training_runs") / self.timestamp
        results_handler = ResultsHandler(output_folder, self.logger)
        results_handler.save_results(results, configs)
        
        return results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Create training orchestrator
    orchestrator = TrainingOrchestrator(device, logger)
    
    # Define configurations to train
    # ==================== Configuration Selection ====================
    # You can choose which models to train by uncommenting the desired configs
    
    configs = [
        # PCA-based models (fast)
        # ConfigurationFactory.create_high_sensitivity_config(),
        # ConfigurationFactory.create_balanced_pca_config(),
        # ConfigurationFactory.create_fast_robust_config(),
        
        # VAE-based models (comprehensive)
        # ConfigurationFactory.create_vae_gmm_mu_config(),
        # ConfigurationFactory.create_vae_recon_kl_config(),
        ConfigurationFactory.create_vae_hybrid_config(),
        # ConfigurationFactory.create_vae_recon_kl_finetuned_config(),
    ]
    
    # ==================== Run Training ====================
    try:
        results = orchestrator.run(
            configs=configs,
            data_parent="trento_house_data",
            weights_folder="./weights",
        )
        
        print(f"\n{'='*90}")
        print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
        print(f"{'='*90}\n")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
