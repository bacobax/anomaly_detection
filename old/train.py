import numpy as np
import torch, open_clip
from models.model import Model
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import csv
import yaml
from pathlib import Path
from datetime import datetime
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class VisualEncoder(nn.Module):
    def __init__(self, visual, normalize=True):
        super().__init__()
        self.visual = visual
        self.normalize = normalize
    
    def forward(self, x,):
        x = self.visual(x)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x

if __name__ == "__main__":
    clip_model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    enc = VisualEncoder(clip_model.visual, normalize=False).eval()

    # Create a single timestamp for this training session and reuse everywhere
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_runs_folder = Path("training_runs") / timestamp

    # ==================== Multiple Model Configurations ====================
    # Configuration 1: High-sensitivity model (sensitive to anomalies, lower false negatives)
    config_1_high_sensitivity = {
        "name": "high_sensitivity",
        "filename": "./weights/model_high_sensitivity.pth",
        "params": {
            "pca_n_components": 128,           # More components capture fine details
            "pca_whiten": True,
            "pca_random_state": 42,
            "gmm_Ks": (2, 3, 4, 5, 6),         # Test more components for finer separation
            "gmm_covariance_type": "full",     # Full covariance captures correlations
            "gmm_reg_covar": 1e-5,             # Lower regularization for sharper boundaries
            "gmm_max_iter": 150,
            "gmm_random_state": 42,
            "train_val_split": 0.90,           # More validation data for careful threshold calibration
            "threshold_quantile": 0.08,        # Conservative threshold: only ~8% normal data flagged as anomalies

        }
    }

    # Configuration 2: Balanced model (balanced trade-off between sensitivity and specificity)
    config_VAE_GMM_mu_score = {
        "name": "VAE GMM_mu_score",
        "filename": "./weights/model_VAE_GMM_mu_score.pth",
        "params": {
            "pca_n_components": 64,            # Medium components balance detail and generalization

            "ft_reduction" : "VAE",
            "vae_latent_dim": 64,        # default: pca_n_components
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
            "vae_finetune_encoder_layers" : 2,

            "pca_whiten": True,
            "pca_random_state": 42,
            "gmm_Ks": (2, 3, 4, 5),            # Moderate component testing
            "gmm_covariance_type": "full",
            "gmm_reg_covar": 1e-5,
            "gmm_max_iter": 100,
            "gmm_random_state": 42,
            "train_val_split": 0.90,           # Standard split
            "threshold_quantile": 0.08,        # Standard threshold: ~8% normal data flagged

        }
    }

    config_VAE_recon_kl_score = {
        "name": "VAE recon_kl_score",
        "filename": "./weights/model_VAE_recon_kl_score.pth",
        "params": {
            "pca_n_components": 64,            # Medium components balance detail and generalization

            "ft_reduction" : "VAE",
            "vae_score_mode": "recon_kl",
            "vae_latent_dim": 64,        # default: pca_n_components
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
            "vae_finetune_encoder_layers" : 0,

            "pca_whiten": True,
            "pca_random_state": 42,
            "gmm_Ks": (2, 3, 4, 5),            # Moderate component testing
            "gmm_covariance_type": "full",
            "gmm_reg_covar": 1e-5,
            "gmm_max_iter": 100,
            "gmm_random_state": 42,
            "train_val_split": 0.9,           # Standard split
            "threshold_quantile": 0.03,        # Standard threshold: ~3% normal data flagged

        }
    }
    config_VAE_hybrid_score = {
        "name": "VAE_hybrid_score",
        "filename": "./weights/model_VAE_hybrid_score.pth",
        "params": {
            "pca_n_components": 64,            # Medium components balance detail and generalization

            "ft_reduction" : "VAE",
            "vae_score_mode": "hybrid",
            "vae_latent_dim": 64,        # default: pca_n_components
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
            "vae_hybrid_w_gmm": 0.3,
            "vae_hybrid_w_recon":  0.7,
            # "vae_finetune_encoder_layers" : 1,

            "pca_whiten": True,
            "pca_random_state": 42,
            "gmm_Ks": [1,2,3,4],            # Moderate component testing
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1e-5,
            "gmm_max_iter": 100,
            "gmm_random_state": 42,
            "train_val_split": 0.90,           # Standard split
            "threshold_quantile": 0.03,        # Standard threshold: ~3% normal data flagged

        }
    }
    config_VAE_recon_kl_score_finetune_clip = {
        "name": "VAE recon_kl_score enc finetuned",
        "filename": "./weights/model_VAE_recon_kl_score_enc_finetuned.pth",
        "params": {
            "pca_n_components": 64,            # Medium components balance detail and generalization

            "ft_reduction" : "VAE",
            "vae_score_mode": "recon_kl",
            "vae_latent_dim": 64,        # default: pca_n_components
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
            "vae_finetune_encoder_layers" : 2,

            "pca_whiten": True,
            "pca_random_state": 42,
            "gmm_Ks": (2, 3, 4, 5),            # Moderate component testing
            "gmm_covariance_type": "full",
            "gmm_reg_covar": 1e-5,
            "gmm_max_iter": 100,
            "gmm_random_state": 42,
            "train_val_split": 0.9,           # Standard split
            "threshold_quantile": 0.03,        # Standard threshold: ~3% normal data flagged

        }
    }

    # Configuration 3: Fast/Robust model (faster training, lower false positives)
    config_3_fast_robust = {
        "name": "fast_robust",
        "filename": "./weights/model_fast_robust.pth",
        "params": {
            "pca_n_components": 32,            # Fewer components reduce noise, faster computation
            "pca_whiten": True,
            "pca_random_state": 42,
            "gmm_Ks": (2, 3, 4),               # Test fewer components for faster training
            "gmm_covariance_type": "diag",     # Diagonal covariance: simpler, faster, more robust
            "gmm_reg_covar": 1e-4,             # Stronger regularization for stability
            "gmm_max_iter": 80,                # Fewer iterations for faster convergence
            "gmm_random_state": 42,
            "train_val_split": 0.90,
            "threshold_quantile": 0.08,        # More lenient threshold: ~8% normal data flagged

        }
    }

    # ==================== Train Multiple Models ====================
    # configs = [config_1_high_sensitivity, config_2_balanced, config_3_fast_robust]
    configs = [config_VAE_hybrid_score, config_VAE_recon_kl_score]
    #configs = [config_VAE_recon_kl_score_finetune_clip]
    
    # List to collect metrics from all models
    metrics_list = []

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Training model: {config['name']}")
        print(f"{'='*60}")
        
        my_model = Model(
            vision_encoder=enc,
            preprocess=preprocess,
            device=device,
            run_timestamp=timestamp,
            **config["params"]
        )
        
        print(f"Fitting model on training_data...")
        my_model.fit(folder_path="training_data")
        
        print(f"Saving model to {config['filename']}...")
        my_model.save_model(config["filename"])
        
        print(f"Testing model on testing_data...")
        test_results = my_model.test_folder("testing_data", output_folder=f"runs/test_results_{config['name']}")
        
        # Extract metrics from test results
        if "_summary" in test_results:
            summary = test_results["_summary"]
            metrics = {
                "config_name": config["name"],
                "precision": summary["anomaly_metrics"]["precision"],
                "recall": summary["anomaly_metrics"]["recall"],
                "f1_score": summary["anomaly_metrics"]["f1_score"]
            }
            metrics_list.append(metrics)
            print(f"✓ Metrics collected for '{config['name']}'")
        
        print(f"✓ Model '{config['name']}' successfully trained and saved!")

    print(f"\n{'='*60}")
    print("All models trained and saved successfully!")
    print(f"{'='*60}")
    
    # ==================== Print and Save Metrics Summary ====================
    if metrics_list:
        # Ensure timestamped folder exists (also used by VAE training artifacts)
        training_runs_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("MODELS PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        # Print table to stdout
        print(f"\n{'Config Name':<20} {'Precision (%)':<18} {'Recall (%)':<18} {'F1 Score':<15}")
        print("-" * 80)
        for metric in metrics_list:
            print(f"{metric['config_name']:<20} {metric['precision']:<18.2f} {metric['recall']:<18.2f} {metric['f1_score']:<15.4f}")
        print("-" * 80)
        
        # Save metrics to CSV
        csv_path = training_runs_folder / "models_metrics_summary.csv"
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['config_name', 'precision', 'recall', 'f1_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metric in metrics_list:
                writer.writerow(metric)
        
        print(f"\n✓ Metrics summary saved to {csv_path}")
        
        # Save YAML configuration files for each model
        for config in configs:
            yaml_path = training_runs_folder / f"{config['name']}_config.yaml"
            
            # Prepare config data for YAML
            config_data = {
                "name": config["name"],
                "filename": config["filename"],
                "parameters": config["params"]
            }
            
            with open(yaml_path, 'w') as yamlfile:
                yaml.dump(config_data, yamlfile, default_flow_style=False, sort_keys=False)
            
            print(f"✓ Configuration saved to {yaml_path}")
        
        print(f"{'='*80}")
        print(f"Training run saved to: {training_runs_folder}")
        print(f"{'='*80}\n")
