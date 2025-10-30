"""
Utility functions for logging and visualization.

Provides centralized configuration for logging and helper functions.
"""

import logging
import warnings
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA


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


def save_test_visualizations(
    images: List[Image.Image],
    verdicts: List[str],
    scores: List[float],
    labels: List[str],
    output_folder: Path,
    threshold: float,
    test_grid_cols: int = 5,
    logger: logging.Logger = None,
) -> None:
    """Save test result visualizations."""
    if not images:
        return
    
    n = len(images)
    cols = test_grid_cols
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
        text = f"{verdict}\n{label}\nScore: {score:.4f}\nτ: {threshold:.4f}"
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
    
    if logger:
        logger.info(f"Visualizations saved to {viz_path}")


def save_metrics_report(
    tp: int, fp: int, fn: int, tn: int,
    accuracy: float, precision: float, recall: float, f1: float,
    threshold: float,
    output_folder: Path,
    logger: logging.Logger = None,
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

Threshold: {threshold:.6f}
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
    
    if logger:
        logger.info(f"Metrics report saved to {report_path}")


def plot_model_2d_visualization(
    reducer: "FeatureReducer",
    scorer: "AnomalyScorer",
    threshold: float,
    normal_data: Tuple[np.ndarray, List[Path]],
    anomaly_data: List[Tuple[str, np.ndarray, List[Path]]],
    ft_reduction: str,
    reduction_method: str = "pca",
    figsize: Tuple[int, int] = (14, 10),
    logger: logging.Logger = None,
) -> plt.Figure:
    """
    Visualize model predictions in 2D space with decision boundaries.
    
    Args:
        reducer: FeatureReducer instance (PCAReducer or VAEReducer)
        scorer: AnomalyScorer instance
        threshold: Decision threshold
        normal_data: (features_array, paths_list) for normal samples
        anomaly_data: List of (folder_name, features_array, paths_list) for anomaly samples
        ft_reduction: "PCA" or "VAE" (for title information)
        reduction_method: '2d' reduction method ('pca' or 'umap'; default 'pca')
        figsize: Figure size (default (14, 10))
        logger: Optional logger instance
    
    Returns:
        matplotlib Figure object
    """
    try:
        import umap
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
    
    Xn, paths_n = normal_data
    
    # Reduce features and compute scores for all data
    Zn = reducer.transform(Xn)
    if hasattr(reducer, 'vae') and reducer.vae is not None:
        # VAE case: X_original may be needed for some scorers
        scores_n = scorer.score(Zn, Xn)
    else:
        # PCA case
        scores_n = scorer.score(Zn, Xn)
    
    auth_n = scores_n >= threshold
    
    # Process anomaly data
    an_data = []
    for folder_name, Xa, paths_a in anomaly_data:
        Za = reducer.transform(Xa)
        if hasattr(reducer, 'vae') and reducer.vae is not None:
            scores_a = scorer.score(Za, Xa)
        else:
            scores_a = scorer.score(Za, Xa)
        auth_a = scores_a >= threshold
        an_data.append((folder_name, Za, paths_a, auth_a, scores_a))
    
    # Stack all reduced features for 2D projection
    all_Z = np.vstack([Zn] + [Za for _, Za, _, _, _ in an_data]) if an_data else Zn
    all_scores = np.concatenate([scores_n] + [scores_a for _, _, _, _, scores_a in an_data]) if an_data else scores_n
    
    # Project to 2D
    if reduction_method.lower() == "pca":
        pca_2d = PCA(n_components=min(2, all_Z.shape[1]))
        all_Z_2d = pca_2d.fit_transform(all_Z)
        if len(pca_2d.explained_variance_ratio_) > 1:
            method_info = f"PCA ({pca_2d.explained_variance_ratio_[0]:.1%}, {pca_2d.explained_variance_ratio_[1]:.1%})"
        else:
            method_info = f"PCA ({pca_2d.explained_variance_ratio_[0]:.1%})"
    elif reduction_method.lower() == "umap":
        if not UMAP_AVAILABLE:
            if logger:
                logger.warning("UMAP not available; falling back to PCA")
            warnings.warn("UMAP not available; falling back to PCA.")
            pca_2d = PCA(n_components=2)
            all_Z_2d = pca_2d.fit_transform(all_Z)
            method_info = "PCA (fallback)"
        else:
            umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            all_Z_2d = umap_reducer.fit_transform(all_Z)
            method_info = "UMAP"
    else:
        raise ValueError(f"Unknown reduction_method: {reduction_method}")
    
    # Split back projected data
    idx = 0
    Zn_2d = all_Z_2d[:len(Zn)]
    idx = len(Zn)
    
    an_2d = []
    for folder_name, Za, paths_a, auth_a, scores_a in an_data:
        n_a = len(Za)
        Za_2d = all_Z_2d[idx:idx + n_a]
        idx += n_a
        an_2d.append((folder_name, Za_2d, paths_a, auth_a, scores_a))
    
    # Create plot with decision boundary
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate grid for decision boundary
    x_range = all_Z_2d[:, 0].max() - all_Z_2d[:, 0].min()
    y_range = all_Z_2d[:, 1].max() - all_Z_2d[:, 1].min()
    
    # Add proportional padding (10% on each side)
    x_pad = max(x_range * 0.1, 0.1)
    y_pad = max(y_range * 0.1, 0.1)
    
    x_min, x_max = all_Z_2d[:, 0].min() - x_pad, all_Z_2d[:, 0].max() + x_pad
    y_min, y_max = all_Z_2d[:, 1].min() - y_pad, all_Z_2d[:, 1].max() + y_pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    # Plot decision boundary using interpolation
    if len(Zn_2d) > 1:
        try:
            from scipy.interpolate import griddata
            Z_bg = griddata(all_Z_2d, all_scores, (xx, yy), method='linear', fill_value=np.nan)
            ax.contourf(xx, yy, Z_bg, levels=[Z_bg.min(), threshold], colors=['mistyrose'], alpha=0.3, zorder=0)
            ax.contour(xx, yy, Z_bg, levels=[threshold], colors='gray', linestyles='--', linewidths=1.5, alpha=0.7, zorder=1)
        except Exception as e:
            if logger:
                logger.warning(f"Could not plot decision boundary: {e}")
            else:
                warnings.warn(f"Could not plot decision boundary: {e}")
    
    # Plot normal data points
    mask_n = np.array(auth_n, dtype=bool)
    ax.scatter(Zn_2d[mask_n, 0], Zn_2d[mask_n, 1], s=20, alpha=0.7, color="green", label="normal (auth)", zorder=3)
    ax.scatter(Zn_2d[~mask_n, 0], Zn_2d[~mask_n, 1], s=20, alpha=0.7, color="red", label="normal (not auth)", zorder=3)
    
    # Plot anomaly data points with different markers
    markers = ["*", "X", "P", "^", "v", "s", "D", "p", "h", "8"]
    for i, (folder_name, Za_2d, paths_a, auth_a, scores_a) in enumerate(an_2d):
        key = Path(folder_name).name
        mask_a = np.array(auth_a, dtype=bool)
        ax.scatter(Za_2d[mask_a, 0], Za_2d[mask_a, 1],
                marker=markers[i % len(markers)], s=100, facecolors="none",
                edgecolors="green", linewidths=1.5, alpha=0.8, label=f"{key} (auth)", zorder=3)
        ax.scatter(Za_2d[~mask_a, 0], Za_2d[~mask_a, 1],
                marker=markers[i % len(markers)], s=100, facecolors="none",
                edgecolors="red", linewidths=1.5, alpha=0.8, label=f"{key} (not auth)", zorder=3)
    
    # Format plot
    title_suffix = f"({ft_reduction})"
    if ft_reduction == "VAE" and hasattr(reducer, 'vae') and reducer.vae is not None:
        latent_dim = reducer.vae.mu.out_features
        score_mode = getattr(scorer, 'config', None)
        if score_mode and hasattr(score_mode, 'score_mode'):
            title_suffix += f" [latent={latent_dim}, score_mode={score_mode.score_mode}]"
        else:
            title_suffix += f" [latent={latent_dim}]"
    
    ax.set_title(f"Model 2D Visualization {title_suffix} | {method_info}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    
    if logger:
        logger.info("2D visualization created successfully")
    
    return fig
