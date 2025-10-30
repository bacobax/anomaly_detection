import numpy as np
from pathlib import Path
from PIL import Image
import torch, open_clip
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from numpy.linalg import slogdet
from joblib import dump, load
import pickle
import json
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path
from ..models.VAE import VAEonCLIP, train_vae, vae_reduce_mu, vae_loss_per_sample

class Model:
    def __init__(
        self,
        vision_encoder,
        preprocess,
        device,
        ft_reduction: str = "PCA",  # or "VAE"
        pca_n_components: int = 20,
        pca_whiten: bool = True,
        pca_random_state: int = 42,
        # VAE hparams (optional; ignored if ft_reduction != 'VAE')
        vae_latent_dim: int | None = None,
        vae_hidden_dim: int = 512,
        vae_dropout: float = 0.2,
        vae_epochs: int = 30,
        vae_batch_size: int = 128,
        vae_lr: float = 1e-3,
        vae_weight_decay: float = 0.0,
        vae_beta: float = 2.0,
        vae_alpha: float = 0.5,
        vae_early_stopping: bool = True,
        vae_patience: int = 5,
        vae_seed: int = 42,
        vae_score_mode: str = "gmm_mu",  # 'gmm_mu' | 'recon_kl' | 'hybrid'
        vae_hybrid_w_gmm: float = 0.5,
        vae_hybrid_w_recon: float = 0.5,
        # VAE encoder finetuning
        vae_finetune_encoder_layers: int = 0,
        vae_encoder_lr: float = 3e-5,
        # optional run info for saving VAE training artifacts
        run_timestamp: str | None = None,
        runs_root: str = "training_runs",
        save_vae_training: bool = True,
        gmm_Ks = (1, 2, 3, 4, 5),
        gmm_covariance_type: str = "full",
        gmm_reg_covar: float = 1e-5,
        gmm_max_iter: int = 100,
        gmm_random_state: int = 42,
        train_val_split: float = 0.95,
        threshold_quantile: float = 0.05,

        file_extensions = ("*.png", "*.jpg", "*.jpeg"),
        test_grid_cols: int = 5,
    ):
        """
        Initialize the anomaly detection model.
        
        Args:
            vision_encoder: Pre-trained vision encoder for feature extraction
            preprocess: Preprocessing function to prepare images for the encoder
            device: Device to run computations on (cpu/cuda)
            
            pca_n_components (int): Number of principal components to keep.
                Increasing: Retains more variance, captures more details, but increases complexity and overfitting risk.
                Decreasing: Reduces dimensionality, focuses on major patterns, faster, but may lose information.
                
            pca_whiten (bool): Whether to whiten features (rescale to unit variance).
                True: Normalizes variance, helps GMM learn equally-weighted clusters, improves stability.
                False: Keeps original variance, may bias GMM toward high-variance directions.
                
            pca_random_state (int): Seed for reproducibility of PCA decomposition.
                Fixed value: Ensures consistent PCA transformation across runs.
                Different values: May yield different feature projections, affecting threshold and detection.
                
            gmm_Ks (tuple): Tuple of component counts to test for model selection (using BIC).
                Larger range: Tests more models, finds optimal complexity, slower but potentially better fit.
                Smaller range: Faster training, but may miss optimal components.
                
            gmm_covariance_type (str): Structure of covariance matrix ("full", "tied", "diag", "spherical").
                "full": Most flexible, captures feature correlations, more parameters to learn.
                "diag": Features conditionally independent, fewer parameters, faster, but misses correlations.
                "tied": All components share same covariance, regularized, good for small datasets but less flexible.
                "spherical": Most constrained, useful for limited data.
                
            gmm_reg_covar (float): Regularization strength for covariance matrix (prevents singularity).
                Higher: Stronger regularization, stable learning, but reduces GMM's ability to model sharp boundaries.
                Lower: Weaker regularization, allows sharper clusters, but risks numerical instability.
                
            gmm_max_iter (int): Maximum iterations for EM algorithm.
                Higher: More refinement of parameters, potentially better fit, but longer training and overfitting risk.
                Lower: Faster training, but may converge prematurely to suboptimal solution.
                
            gmm_random_state (int): Seed for reproducibility of GMM initialization.
                Fixed value: Ensures consistent EM results across runs.
                Different values: May lead to different local optima, affecting learned clusters.
                
            train_val_split (float): Fraction of data for training PCA/GMM (rest for threshold calibration).
                Higher (0.95): More training data, better model fit, but less validation data for threshold tuning.
                Lower (0.80): More validation data for better threshold, but poorer PCA/GMM fit.
                
            threshold_quantile (float): Quantile of validation log-likelihood as anomaly threshold.
                Lower (0.01): Conservative threshold, detects more anomalies, higher false positive rate.
                Higher (0.10): Lenient threshold, allows more anomalies, higher false negative rate.
                0.05: Typical - flags ~5% of normal data as anomalies.

                
            file_extensions (tuple): Image file patterns to search during embedding/testing.
                Broader patterns: Includes more types, more flexibility, may include unintended formats.
                Narrower patterns: More controlled, faster, but may skip relevant images.
                
            test_grid_cols (int): Number of columns in visualization grid for test results.
                Higher (8): More compact, see all results at once, but smaller images.
                Lower (3): Larger images, easier inspection, but requires scrolling.
        """
        self.vision_encoder = vision_encoder
        self.preprocess = preprocess
        self.device = device

        # Learned objects
        self.pca = None
        self.vae = None
        self.gmm = None
        self.tau = None
        
        # Hybrid scoring components
        self._hybrid_scaler: StandardScaler | None = None
        self._hybrid_clf: LogisticRegression | None = None
        
        # Mahalanobis distance components (for optional third stream)
        self._maha_mean: np.ndarray | None = None
        self._maha_inv_cov: np.ndarray | None = None
        
        # Reconstruction variance for recon_loglik
        self._recon_inv_sigma2: np.ndarray | None = None
        self._recon_const: float = 0.0

        # Store hyperparameters
        self.ft_reduction = (ft_reduction or "PCA").upper()
        self.pca_n_components = pca_n_components
        self.pca_whiten = pca_whiten
        self.pca_random_state = pca_random_state

        # VAE hyperparameters
        self.vae_latent_dim = vae_latent_dim  # if None, will default to pca_n_components
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_dropout = vae_dropout
        self.vae_epochs = vae_epochs
        self.vae_batch_size = vae_batch_size
        self.vae_lr = vae_lr
        self.vae_weight_decay = vae_weight_decay
        self.vae_beta = vae_beta
        self.vae_alpha = vae_alpha
        self.vae_early_stopping = vae_early_stopping
        self.vae_patience = vae_patience
        self.vae_seed = vae_seed
        self.vae_score_mode = (vae_score_mode or "gmm_mu").lower()
        self.vae_hybrid_w_gmm = float(vae_hybrid_w_gmm)
        self.vae_hybrid_w_recon = float(vae_hybrid_w_recon)
        self.vae_finetune_encoder_layers = int(vae_finetune_encoder_layers)
        self.vae_encoder_lr = float(vae_encoder_lr)

        # Run/session context
        self.run_timestamp = run_timestamp
        self.runs_root = runs_root
        self.save_vae_training = save_vae_training

        # Hybrid normalization params (computed on validation set)
        self._hybrid_norm = None  # dict with means/stds for 'gmm' and 'recon'

        self.gmm_Ks = tuple(gmm_Ks)
        self.gmm_covariance_type = gmm_covariance_type
        self.gmm_reg_covar = gmm_reg_covar
        self.gmm_max_iter = gmm_max_iter
        self.gmm_random_state = gmm_random_state

        self.train_val_split = float(train_val_split)
        self.threshold_quantile = float(threshold_quantile)

        self.file_extensions = tuple(file_extensions)
        self.test_grid_cols = int(test_grid_cols)


    def _load_images_as_tensors(self, folder_path):
        """Load image files as preprocessed tensors (not encoded)."""
        import torch
        images, paths = [], []
        for ext in self.file_extensions:
            paths.extend(sorted(Path(folder_path).glob(ext)))
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                img_tensor = self.preprocess(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"Error processing {p}: {e}")
        if not images:
            return torch.empty((0, 3, 224, 224))  # Placeholder shape
        return torch.stack(images)

    def _compute_recon_loglik(self, x_true_np, x_hat_np):
        """Compute per-sample reconstruction log-likelihood using variance-aware model."""
        if self._recon_inv_sigma2 is None or self._recon_const is None:
            # Fallback: use simple squared error
            return -np.mean((x_hat_np - x_true_np) ** 2, axis=1)
        resid = x_hat_np - x_true_np
        return -(0.5 * (resid * resid * self._recon_inv_sigma2).sum(axis=1) + self._recon_const)

    def _sample_from_latent(self, mu, logvar, K=5):
        """Draw K samples from q(z|x) = N(mu, exp(logvar))."""
        samples = []
        std = (0.5 * logvar).exp()
        for _ in range(K):
            z = mu + torch.randn_like(std) * std
            samples.append(z)
        return samples

    def _compute_scores_with_sampling(self, x, x_hat, mu, logvar, K=5):
        """
        Compute GMM and recon scores by averaging over K samples from q(z|x).
        
        Returns:
            gmm_s: averaged GMM scores (or None if no GMM)
            recon_s: averaged recon log-likelihood scores
        """
        gmm_s_list = []
        recon_s_list = []
        
        for k in range(K):
            # Sample from q(z|x)
            std = (0.5 * logvar).exp()
            z_sample = mu + torch.randn_like(std) * std
            
            # GMM score
            if self.gmm is not None:
                z_np = z_sample.cpu().numpy()
                gmm_s_sample = self.gmm.score_samples(z_np)
                gmm_s_list.append(gmm_s_sample)
            
            # Recon score (use decoder with sampled z)
            x_hat_sample = self.vae.decode(z_sample)
            x_hat_np = x_hat_sample.cpu().numpy()
            x_np = x.cpu().numpy()
            recon_s_sample = self._compute_recon_loglik(x_np, x_hat_np)
            recon_s_list.append(recon_s_sample)
        
        # Average over samples
        if gmm_s_list:
            gmm_s = np.mean(gmm_s_list, axis=0)
        else:
            gmm_s = None
        recon_s = np.mean(recon_s_list, axis=0)
        
        return gmm_s, recon_s

    def _best_tau_by_f1(self, scores, y):
        """Find threshold that maximizes F1 on validation set.
        
        Args:
            scores: anomaly scores (higher = more normal)
            y: labels (0=normal, 1=anomaly)
        
        Returns:
            float: optimal threshold
        """
        if len(np.unique(y)) < 2:
            # Only one class; use quantile method
            return float(np.quantile(scores, self.threshold_quantile))
        
        qs = np.linspace(0.01, 0.99, 99)
        grid = np.quantile(scores, qs)
        best_f1 = -1
        best_tau = float(np.quantile(scores, self.threshold_quantile))
        
        for t in grid:
            # score < t means anomaly (predicted positive)
            y_pred_anom = (scores < t).astype(int)
            tp = np.sum((y == 1) & (y_pred_anom == 1))
            fp = np.sum((y == 0) & (y_pred_anom == 1))
            fn = np.sum((y == 1) & (y_pred_anom == 0))
            
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            
            if f1 > best_f1:
                best_f1 = f1
                best_tau = t
        
        return float(best_tau)

    def fit(self, folder_path):
        # For VAE training with encoder finetuning, load raw images
        if self.ft_reduction == "VAE" and self.vae_finetune_encoder_layers > 0:
            # Load images as tensors for on-the-fly encoding during VAE training
            img_tensors = self._load_images_as_tensors(folder_path)
            n = img_tensors.shape[0]

            # Shuffle samples with seeded permutation for reproducibility
            perm = np.random.RandomState(self.pca_random_state).permutation(n)
            img_tensors = img_tensors[perm]

            split = self.train_val_split
            train_size = max(1, int(split * n))
            if train_size == n:  # ensure non-empty val
                train_size = n - 1

            img_train = img_tensors[:train_size]
            img_val = img_tensors[train_size:]

            # Train VAE with encoder on-the-fly encoding and finetuning
            d_lat = self.vae_latent_dim if self.vae_latent_dim is not None else self.pca_n_components

            self.vae, history, self.vision_encoder = train_vae(
                img_train,
                img_val if len(img_val) > 0 else None,
                d_in=None,  # Will be inferred from encoder
                d_lat=int(d_lat),
                d_hid=int(self.vae_hidden_dim),
                dropout=float(self.vae_dropout),
                epochs=int(self.vae_epochs),
                batch_size=int(self.vae_batch_size),
                lr=float(self.vae_lr),
                weight_decay=float(self.vae_weight_decay),
                beta=float(self.vae_beta),
                alpha=float(self.vae_alpha),
                early_stopping=bool(self.vae_early_stopping),
                patience=int(self.vae_patience),
                device=self.device,
                seed=int(self.vae_seed),
                verbose=True,
                vision_encoder=self.vision_encoder,
                preprocess=None,  # Images are already preprocessed
                finetune_encoder_layers=int(self.vae_finetune_encoder_layers),
                encoder_lr=float(self.vae_encoder_lr),
            )

            # Extract features using the trained/finetuned encoder for GMM training
            self.vision_encoder.eval()
            with torch.no_grad():
                feats_train = []
                for i in range(0, len(img_train), 64):
                    batch = img_train[i:i + 64].to(self.device)
                    feat = self.vision_encoder(batch).float()
                    feats_train.append(feat.cpu().numpy())
                X_train = np.vstack(feats_train)

            # Reduce using VAE encoder
            Z_train = vae_reduce_mu(self.vae, X_train)
            Z_train = Z_train.numpy()

            # Validation reduction if available
            if len(img_val) > 0:
                with torch.no_grad():
                    feats_val = []
                    for i in range(0, len(img_val), 64):
                        batch = img_val[i:i + 64].to(self.device)
                        feat = self.vision_encoder(batch).float()
                        feats_val.append(feat.cpu().numpy())
                    X_val = np.vstack(feats_val)
                Z_val = vae_reduce_mu(self.vae, X_val)
                Z_val = Z_val.numpy()
            else:
                Z_val = np.empty((0, self.vae_latent_dim or self.pca_n_components))

            reducted_train = Z_train
            reducted_val = Z_val

            # Compute reconstruction variance for variance-aware log-likelihood
            try:
                self.vae.eval()
                with torch.no_grad():
                    Xt = torch.from_numpy(X_train).to(self.device).float()
                    x_hat, _, _ = self.vae(Xt)
                resid = (x_hat.cpu().numpy() - X_train)
                sigma2 = resid.var(axis=0) + 1e-6
                self._recon_inv_sigma2 = 1.0 / sigma2
                self._recon_const = 0.5 * np.sum(np.log(2 * np.pi * sigma2))
            except Exception as e:
                warnings.warn(f"Failed to compute reconstruction variance: {e}")
                self._recon_inv_sigma2 = None
                self._recon_const = 0.0

            # Fit GMM if needed for scoring
            need_gmm = (self.vae_score_mode in ("gmm_mu", "hybrid"))
            gmm = None
            if need_gmm and len(reducted_train) > 0:
                gmm = self.fit_best_gmm(reducted_train, Ks=self.gmm_Ks)

            # Compute validation scores for threshold calibration
            if len(img_val) < 1:
                val_scores = np.array([])
            else:
                if self.vae_score_mode == "gmm_mu":
                    if gmm is None:
                        raise RuntimeError("GMM not available for VAE 'gmm_mu' scoring.")
                    val_scores = gmm.score_samples(reducted_val)

                elif self.vae_score_mode == "recon_kl":
                    # Use variance-aware reconstruction log-likelihood as score (higher is better)
                    self.vae.eval()
                    with torch.no_grad():
                        xv = torch.from_numpy(X_val).to(self.device).float()
                        x_hat, mu_v, logvar_v = self.vae(xv)
                    x_hat_np = x_hat.cpu().numpy()
                    val_scores = self._compute_recon_loglik(X_val, x_hat_np)

                elif self.vae_score_mode == "hybrid":
                    if gmm is None:
                        raise RuntimeError("GMM not available for VAE 'hybrid' scoring.")
                    self.vae.eval()
                    with torch.no_grad():
                        xv = torch.from_numpy(X_val).to(self.device).float()
                        x_hat, mu_v, logvar_v = self.vae(xv)
                        total_ps, parts = vae_loss_per_sample(
                            xv, x_hat, mu_v, logvar_v, beta=self.vae_beta, alpha=self.vae_alpha
                        )
                        recon_scores = (-total_ps).cpu().numpy()
                    gmm_scores = gmm.score_samples(reducted_val)

                    # z-normalize both based on validation set stats
                    eps = 1e-8
                    m_g, s_g = float(np.mean(gmm_scores)), float(np.std(gmm_scores) + eps)
                    m_r, s_r = float(np.mean(recon_scores)), float(np.std(recon_scores) + eps)
                    gmm_z = (gmm_scores - m_g) / s_g
                    recon_z = (recon_scores - m_r) / s_r

                    # Learned hybrid: train logistic regression on validation set
                    S_val = np.c_[gmm_z, recon_z]
                    y_val = np.zeros(len(S_val), dtype=int)
                    try:
                        scaler = StandardScaler().fit(S_val)
                        Sz = scaler.transform(S_val)
                        clf = LogisticRegression(
                            max_iter=1000, class_weight='balanced', random_state=self.vae_seed
                        )
                        clf.fit(Sz, y_val)
                        self._hybrid_scaler = scaler
                        self._hybrid_clf = clf
                        # Use classifier probability for normality score
                        proba = clf.predict_proba(Sz)
                        val_scores = 1.0 - proba[:, 1]  # higher = more normal
                    except Exception as e:
                        warnings.warn(f"Failed to train hybrid logistic classifier: {e}")
                        # Fallback to weighted sum
                        val_scores = (
                            self.vae_hybrid_w_gmm * gmm_z
                            + self.vae_hybrid_w_recon * recon_z
                        )

                    self._hybrid_norm = {
                        "gmm_mean": m_g, "gmm_std": s_g,
                        "recon_mean": m_r, "recon_std": s_r
                    }

                else:
                    raise ValueError(f"Unknown vae_score_mode: {self.vae_score_mode}")

            # Assign learned components for later inference
            self.gmm = gmm if need_gmm else None

            # Calibrate threshold (tau) from validation scores
            if len(img_val) < 1:
                warnings.warn("Very small validation set; using minimal score as τ.")
                self.tau = float(np.min(val_scores)) if val_scores.size > 0 else -1e9
            elif len(img_val) < 3:
                warnings.warn("Very small validation set; using min as τ.")
                self.tau = float(np.min(val_scores))
            else:
                self.tau = float(np.quantile(val_scores, self.threshold_quantile))

            # Early return: finetune path is complete
            return

        # Original path: extract features first (for PCA or non-finetuned VAE)
        X = self.embed_folder(folder_path)
        n = X.shape[0]

        # Shuffle samples with seeded permutation for reproducibility
        perm = np.random.RandomState(self.pca_random_state).permutation(n)
        X = X[perm]

        split = self.train_val_split
        train_size = max(1, int(split * n))
        if train_size == n:  # ensure non-empty val
            train_size = n - 1

        X_train = X[:train_size]
        X_val = X[train_size:]

        # Choose feature reduction method
        if self.ft_reduction == "PCA":
            reducted_train = self.fit_pca(X_train)
            reducted_val = self.pca.transform(X_val)

        elif self.ft_reduction == "VAE":
            # Train VAE on training features; use mu as reduced features
            d_in = int(X_train.shape[1]) if X_train.size else 0
            d_lat = self.vae_latent_dim if self.vae_latent_dim is not None else self.pca_n_components
            if d_in == 0:
                raise RuntimeError("No training features available to train VAE.")

            self.vae, history, _ = train_vae(
                X_train,
                X_val if len(X_val) > 0 else None,
                d_in=d_in,
                d_lat=int(d_lat),
                d_hid=int(self.vae_hidden_dim),
                dropout=float(self.vae_dropout),
                epochs=int(self.vae_epochs),
                batch_size=int(self.vae_batch_size),
                lr=float(self.vae_lr),
                weight_decay=float(self.vae_weight_decay),
                beta=float(self.vae_beta),
                alpha=float(self.vae_alpha),
                early_stopping=bool(self.vae_early_stopping),
                patience=int(self.vae_patience),
                device=self.device,
                seed=int(self.vae_seed),
                verbose=True,
                vision_encoder=None,
                preprocess=None,
                finetune_encoder_layers=0,
                encoder_lr=0.0,
            )

            # Reduce
            Z_train = vae_reduce_mu(self.vae, X_train)
            Z_val = vae_reduce_mu(self.vae, X_val) if len(X_val) > 0 else torch.empty((0, d_lat))
            reducted_train = Z_train.numpy()
            reducted_val = Z_val.numpy() if len(Z_val) > 0 else np.empty((0, d_lat), dtype=np.float32)

            # Compute reconstruction variance for variance-aware log-likelihood
            try:
                self.vae.eval()
                with torch.no_grad():
                    Xt = torch.from_numpy(X_train).to(self.device).float()
                    x_hat, _, _ = self.vae(Xt)
                resid = (x_hat.cpu().numpy() - X_train)
                sigma2 = resid.var(axis=0) + 1e-6
                self._recon_inv_sigma2 = 1.0 / sigma2
                self._recon_const = 0.5 * np.sum(np.log(2 * np.pi * sigma2))
            except Exception as e:
                warnings.warn(f"Failed to compute reconstruction variance: {e}")
                self._recon_inv_sigma2 = None
                self._recon_const = 0.0

            # Fit GMM if needed for scoring
            need_gmm = (self.vae_score_mode in ("gmm_mu", "hybrid"))
            gmm = None
            if need_gmm and len(reducted_train) > 0:
                gmm = self.fit_best_gmm(reducted_train, Ks=self.gmm_Ks)

            # Compute validation scores for threshold calibration
            if len(X_val) < 1:
                val_scores = np.array([])
            else:
                if self.vae_score_mode == "gmm_mu":
                    if gmm is None:
                        raise RuntimeError("GMM not available for VAE 'gmm_mu' scoring.")
                    val_scores = gmm.score_samples(reducted_val)

                elif self.vae_score_mode == "recon_kl":
                    # Use variance-aware reconstruction log-likelihood as score (higher is better)
                    self.vae.eval()
                    with torch.no_grad():
                        xv = torch.from_numpy(X_val).to(self.device).float()
                        x_hat, mu_v, logvar_v = self.vae(xv)
                    x_hat_np = x_hat.cpu().numpy()
                    val_scores = self._compute_recon_loglik(X_val, x_hat_np)

                elif self.vae_score_mode == "hybrid":
                    if gmm is None:
                        raise RuntimeError("GMM not available for VAE 'hybrid' scoring.")
                    self.vae.eval()
                    with torch.no_grad():
                        xv = torch.from_numpy(X_val).to(self.device).float()
                        x_hat, mu_v, logvar_v = self.vae(xv)
                        total_ps, parts = vae_loss_per_sample(
                            xv, x_hat, mu_v, logvar_v, beta=self.vae_beta, alpha=self.vae_alpha
                        )
                        recon_scores = (-total_ps).cpu().numpy()
                    gmm_scores = gmm.score_samples(reducted_val)

                    # z-normalize both based on validation set stats
                    eps = 1e-8
                    m_g, s_g = float(np.mean(gmm_scores)), float(np.std(gmm_scores) + eps)
                    m_r, s_r = float(np.mean(recon_scores)), float(np.std(recon_scores) + eps)
                    gmm_z = (gmm_scores - m_g) / s_g
                    recon_z = (recon_scores - m_r) / s_r

                    # Learned hybrid: train logistic regression on validation set
                    S_val = np.c_[gmm_z, recon_z]
                    # For now, assume we only have normal validation data (label=0)
                    y_val = np.zeros(len(S_val), dtype=int)
                    try:
                        scaler = StandardScaler().fit(S_val)
                        Sz = scaler.transform(S_val)
                        clf = LogisticRegression(
                            max_iter=1000, class_weight='balanced', random_state=self.vae_seed
                        )
                        clf.fit(Sz, y_val)
                        self._hybrid_scaler = scaler
                        self._hybrid_clf = clf
                        # Use classifier probability for normality score
                        proba = clf.predict_proba(Sz)
                        val_scores = 1.0 - proba[:, 1]  # higher = more normal
                    except Exception as e:
                        warnings.warn(f"Failed to train hybrid logistic classifier: {e}")
                        # Fallback to weighted sum
                        val_scores = (
                            self.vae_hybrid_w_gmm * gmm_z
                            + self.vae_hybrid_w_recon * recon_z
                        )

                    self._hybrid_norm = {
                        "gmm_mean": m_g, "gmm_std": s_g,
                        "recon_mean": m_r, "recon_std": s_r
                    }

                else:
                    raise ValueError(f"Unknown vae_score_mode: {self.vae_score_mode}")

                # Save VAE training artifacts if requested and timestamp is available
                if self.save_vae_training and self.run_timestamp:
                    out_dir = Path(self.runs_root) / self.run_timestamp / "VAE_training"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    # 1) Plot losses
                    try:
                        plt.figure(figsize=(7, 4))
                        plt.plot(range(1, len(history.get("loss", [])) + 1), history.get("loss", []), label="train")
                        if "val_loss" in history and len(history["val_loss"]) > 0:
                            plt.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="val")
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.title("VAE Training Loss")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(out_dir / "loss_curve.png", dpi=120)
                        plt.close()
                    except Exception as e:
                        warnings.warn(f"Failed to save VAE loss plot: {e}")
                    # 2) Save history CSV
                    try:
                        import csv
                        with open(out_dir / "history.csv", "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["epoch", "loss", "val_loss"])
                            max_e = max(len(history.get("loss", [])), len(history.get("val_loss", [])))
                            for i in range(max_e):
                                tr = history.get("loss", [])
                                va = history.get("val_loss", [])
                                writer.writerow([i + 1, tr[i] if i < len(tr) else "", va[i] if i < len(va) else ""])
                    except Exception as e:
                        warnings.warn(f"Failed to save VAE history CSV: {e}")
                    # 3) Save summary JSON
                    try:
                        summary = {
                            "ft_reduction": self.ft_reduction,
                            "d_in": d_in,
                            "d_lat": int(d_lat),
                            "d_hid": int(self.vae_hidden_dim),
                            "epochs_trained": len(history.get("loss", [])),
                            "final_train_loss": history.get("loss", [])[-1] if history.get("loss") else None,
                            "best_val_loss": min(history.get("val_loss", [])) if history.get("val_loss") else None,
                            "score_mode": self.vae_score_mode,
                            "hybrid_weights": {"gmm": self.vae_hybrid_w_gmm, "recon": self.vae_hybrid_w_recon},
                            "params": {
                                "vae_dropout": float(self.vae_dropout),
                                "vae_lr": float(self.vae_lr),
                                "vae_weight_decay": float(self.vae_weight_decay),
                                "vae_beta": float(self.vae_beta),
                                "vae_alpha": float(self.vae_alpha),
                                "vae_batch_size": int(self.vae_batch_size),
                                "vae_early_stopping": bool(self.vae_early_stopping),
                                "vae_patience": int(self.vae_patience),
                                "seed": int(self.vae_seed),
                            },
                        }
                        with open(out_dir / "summary.json", "w") as f:
                            json.dump(summary, f, indent=2)
                    except Exception as e:
                        warnings.warn(f"Failed to save VAE summary JSON: {e}")

            # Assign learned components for later inference
            if self.ft_reduction == "VAE":
                self.gmm = gmm if need_gmm else None

            # Calibrate threshold (tau) from validation scores
            if self.ft_reduction == "VAE":
                if len(X_val) < 1:
                    warnings.warn("Very small validation set; using minimal score as τ.")
                    self.tau = float(np.min(val_scores)) if val_scores.size > 0 else -1e9
                elif len(X_val) < 3:
                    warnings.warn("Very small validation set; using min as τ.")
                    self.tau = float(np.min(val_scores))
                else:
                    self.tau = float(np.quantile(val_scores, self.threshold_quantile))

        else:
            raise ValueError(f"Unknown ft_reduction: {self.ft_reduction}")

        # Handle PCA path if we're still in the non-finetuned case
        if self.ft_reduction == "PCA":
            # Existing PCA path: fit GMM and threshold on log-likelihood
            gmm = self.fit_best_gmm(reducted_train, Ks=self.gmm_Ks)
            val_ll = gmm.score_samples(reducted_val)
            if len(reducted_val) < 3:
                warnings.warn("Very small validation set; using min log-likelihood as τ.")
                self.tau = float(np.min(val_ll))
            else:
                self.tau = float(np.quantile(val_ll, self.threshold_quantile))
            self.gmm = gmm

    def full_test(self, image_path):
        print("EXECUTING FULL TEST..")
        is_auth, prob_is_auth = self.is_authorized(image_path)
        print(f"Is authorized: {is_auth} (score: {prob_is_auth}), tau={self.tau}")
        gmm = self.gmm
        if gmm is not None:
            print("Weights:", gmm.weights_)
            print("Means shape:", gmm.means_.shape)
            print("First mean vector:", gmm.means_[0][:10])     # first 10 dims
            print("Covariance type:", gmm.covariance_type)
            print("Covariances shape:", gmm.covariances_.shape)
            print("First covariance diagonal:", np.diag(gmm.covariances_[0])[:10])
            for i, cov in enumerate(self.gmm.covariances_):
                sign, logdet = slogdet(cov)
                print(f"Component {i}: log|Σ| = {logdet:.2f}")
        print("=========================")


    def is_authorized(self, pil_image):

        if type(pil_image) is str:
            pil_image = Image.open(pil_image)

        with torch.no_grad():
            f = self.vision_encoder(self.preprocess(pil_image).unsqueeze(0).to(self.device)).float()
        if self.ft_reduction == "PCA":
            reducted = self.pca.transform(f.cpu().numpy())
            score = self.gmm.score_samples(reducted)[0]
            return score >= self.tau, score
        elif self.ft_reduction == "VAE":
            if self.vae is None:
                raise RuntimeError("VAE not trained/loaded but ft_reduction is 'VAE'.")
            self.vae.eval()
            with torch.no_grad():
                x = f.to(next(self.vae.parameters()).device).float()
                x_hat, mu, logvar = self.vae(x)
            if self.vae_score_mode == "gmm_mu":
                reducted = mu.cpu().numpy()
                score = self.gmm.score_samples(reducted)[0]
            elif self.vae_score_mode == "recon_kl":
                x_hat_np = x_hat.cpu().numpy()
                x_np = x.cpu().numpy()
                score = float(self._compute_recon_loglik(x_np, x_hat_np)[0])
            elif self.vae_score_mode == "hybrid":
                reducted = mu.cpu().numpy()
                gmm_s = float(self.gmm.score_samples(reducted)[0])
                x_hat_np = x_hat.cpu().numpy()
                x_np = x.cpu().numpy()
                recon_s = float(self._compute_recon_loglik(x_np, x_hat_np)[0])
                # z-normalize using stored norms
                if not self._hybrid_norm:
                    # Fallback: no norms stored; treat as unnormalized weighted sum
                    score = self.vae_hybrid_w_gmm * gmm_s + self.vae_hybrid_w_recon * recon_s
                else:
                    eps = 1e-8
                    m_g, s_g = self._hybrid_norm.get("gmm_mean", 0.0), self._hybrid_norm.get("gmm_std", 1.0)
                    m_r, s_r = self._hybrid_norm.get("recon_mean", 0.0), self._hybrid_norm.get("recon_std", 1.0)
                    gmm_z = (gmm_s - m_g) / (s_g if s_g != 0 else 1.0)
                    recon_z = (recon_s - m_r) / (s_r if s_r != 0 else 1.0)
                    
                    # Use learned hybrid classifier if available
                    if self._hybrid_scaler and self._hybrid_clf:
                        S = np.array([[gmm_z, recon_z]])
                        Sz = self._hybrid_scaler.transform(S)
                        proba = self._hybrid_clf.predict_proba(Sz)
                        score = float(1.0 - proba[0, 1])  # higher = more normal
                    else:
                        # Fallback to weighted sum
                        score = self.vae_hybrid_w_gmm * gmm_z + self.vae_hybrid_w_recon * recon_z
            else:
                raise ValueError(f"Unknown vae_score_mode: {self.vae_score_mode}")
            return score >= self.tau, score
        else:
            raise ValueError(f"Unknown ft_reduction: {self.ft_reduction}")

    def test_folder(self, test_data_folder, output_folder="test_results"):
        """
        Test model on images in a folder with /normal and /anomalous subfolders.
        Generates test results visualization and confusion matrix report.
        
        Folder structure:
            test_data_folder/
                normal/     (label=0, should be authorized)
                anomalous/  (label=1, should be not authorized)
        
        Output files saved in output_folder:
            - test_results_images.png: Grid of test images with verdicts
            - metrics_report.png: Confusion matrix with precision, recall, F1 score
        
        Args:
            test_data_folder: Path to folder containing /normal and /anomalous subfolders
            output_folder: Name of output folder to save results (default: "test_results")
        
        Returns:
            Dictionary with results including all metrics
        """
        
        
        test_data_folder = Path(test_data_folder)
        output_folder = Path(output_folder)
        
        # Create output folder if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Check for required subfolders
        normal_folder = test_data_folder / "normal"
        anomalous_folder = test_data_folder / "anomalous"
        
        if not normal_folder.exists() or not anomalous_folder.exists():
            print(f"Error: Expected 'normal' and 'anomalous' subfolders in {test_data_folder}")
            return {}
        
        # Collect images with their labels
        image_data = []  # List of (img_path, label)
        
        # Collect normal images (label=0)
        for ext in self.file_extensions:
            for img_path in sorted(normal_folder.glob(ext)):
                image_data.append((img_path, 0))
        
        # Collect anomalous images (label=1)
        for ext in self.file_extensions:
            for img_path in sorted(anomalous_folder.glob(ext)):
                image_data.append((img_path, 1))
        
        if not image_data:
            print(f"No images found in {normal_folder} or {anomalous_folder}")
            return {}
        
        results = {}
        images = []
        verdicts = []
        scores = []
        labels = []
        correct_predictions = 0
        total_predictions = 0
        
        # Metrics for anomalous class (positive class, label=1)
        # is_auth=False means anomaly detected (predicted positive)
        tp = 0  # True Positives: anomalous detected as anomalous (label=1, is_auth=False)
        fp = 0  # False Positives: normal detected as anomalous (label=0, is_auth=False)
        fn = 0  # False Negatives: anomalous detected as normal (label=1, is_auth=True)
        tn = 0  # True Negatives: normal detected as normal (label=0, is_auth=True)
        
        true_labels = []
        pred_labels = []
        
        for img_path, label in image_data:
            try:
                is_auth, score = self.is_authorized(str(img_path))
                
                # Expected: is_auth should be True if label=0, False if label=1
                expected_is_auth = (label == 0)
                is_correct = (is_auth == expected_is_auth)
                
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                # For confusion matrix, convert to 0 and 1 (0=normal, 1=anomaly)
                # is_auth=False means anomaly detected (1), is_auth=True means normal (0)
                pred_label = 0 if is_auth else 1
                true_labels.append(label)
                pred_labels.append(pred_label)
                
                # Update confusion matrix for anomalous class (label=1 is positive)
                if label == 1:  # Anomalous (positive class)
                    if is_auth == False:  # Correctly detected as anomalous
                        tp += 1
                    else:  # Incorrectly detected as normal (false negative)
                        fn += 1
                else:  # Normal (negative class)
                    if is_auth == False:  # Incorrectly detected as anomalous (false positive)
                        fp += 1
                    else:  # Correctly detected as normal
                        tn += 1
                
                verdict = "✓ CORRECT" if is_correct else "✗ WRONG"
                label_str = "NORMAL" if label == 0 else "ANOMALOUS"
                
                results[str(img_path)] = {
                    "label": label,
                    "label_str": label_str,
                    "authorized": is_auth,
                    "expected_authorized": expected_is_auth,
                    "correct": is_correct,
                    "score": score,
                    "threshold": self.tau
                }
                
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                verdicts.append(verdict)
                scores.append(score)
                labels.append(label_str)
                                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Calculate metrics
        precision = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        # Precision (for anomalous class): TP / (TP + FP)
        precision_anomalous = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        
        # Recall (for anomalous class): TP / (TP + FN)
        recall_anomalous = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        
        # F1 Score: 2 * (precision * recall) / (precision + recall)
        if precision_anomalous + recall_anomalous > 0:
            f1_score = 2 * (precision_anomalous * recall_anomalous) / (precision_anomalous + recall_anomalous)
        else:
            f1_score = 0
        
        print(f"\n{'='*60}")
        print(f"Test Results Summary:")
        print(f"Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"Overall Accuracy: {precision:.2f}%")
        print(f"\nAnomaly Detection Metrics (Anomalous as Positive Class):")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"  Precision: {precision_anomalous:.2f}%")
        print(f"  Recall: {recall_anomalous:.2f}%")
        print(f"  F1 Score: {f1_score:.4f}")
        print(f"{'='*60}\n")
        
        # Create visualization grid with test results
        if images:
            n_images = len(images)
            cols = self.test_grid_cols
            rows = (n_images + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            if n_images == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for idx, (ax, img, verdict, score, label) in enumerate(zip(axes, images, verdicts, scores, labels)):
                ax.imshow(img)
                ax.axis("off")
                
                # Color code: green for correct, red for wrong
                color = "green" if "✓" in verdict else "red"
                
                # Add text with verdict, label, score, and threshold
                text_str = f"{verdict}\n{label}\nScore: {score:.4f}\nThreshold: {self.tau:.4f}"
                ax.text(0.5, -0.05, text_str, 
                       transform=ax.transAxes,
                       fontsize=10,
                       verticalalignment="top",
                       horizontalalignment="center",
                       bbox=dict(boxstyle="round", facecolor=color, alpha=0.3))
            
            # Hide unused subplots
            for idx in range(n_images, len(axes)):
                axes[idx].axis("off")
            
            plt.tight_layout()
            
            results_image_path = output_folder / "test_results_images.png"
            plt.savefig(results_image_path, dpi=150, bbox_inches="tight")
            print(f"Test results visualization saved to {results_image_path}")
            plt.close()
        
        # Create confusion matrix visualization with metrics report
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                   xticklabels=['Normal', 'Anomaly'], 
                   yticklabels=['Normal', 'Anomaly'],
                   cbar_kws={'label': 'Count'})
        axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
        axes[0].set_title('Confusion Matrix\n(Anomaly Detection)', fontsize=14, fontweight='bold')
        
        # Right plot: Metrics Report
        axes[1].axis('off')
        metrics_text = f"""
PERFORMANCE METRICS

Overall Results:
  Correct Predictions: {correct_predictions}/{total_predictions}
  Overall Accuracy: {precision:.2f}%

Confusion Matrix:
  True Positives (TP): {tp}
  False Positives (FP): {fp}
  False Negatives (FN): {fn}
  True Negatives (TN): {tn}

Anomaly Detection Metrics:
  Precision: {precision_anomalous:.2f}%
  Recall: {recall_anomalous:.2f}%
  F1 Score: {f1_score:.4f}

Threshold: {self.tau:.4f}
        """
        axes[1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        metrics_path = output_folder / "metrics_report.png"
        plt.savefig(metrics_path, dpi=150, bbox_inches="tight")
        print(f"Metrics report saved to {metrics_path}")
        plt.close()
        
        # Add summary to results
        results["_summary"] = {
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "overall_accuracy": precision,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            },
            "anomaly_metrics": {
                "precision": precision_anomalous,
                "recall": recall_anomalous,
                "f1_score": f1_score
            },
            "output_folder": str(output_folder),
            "results_image": str(output_folder / "test_results_images.png"),
            "metrics_report": str(output_folder / "metrics_report.png")
        }
        
        return results
    
    def fit_best_gmm(self, Z, Ks=(2,3,4)):
        best = None
        for k in Ks:
            g = GaussianMixture(
                n_components=k,
                covariance_type=self.gmm_covariance_type,
                reg_covar=self.gmm_reg_covar,
                max_iter=self.gmm_max_iter,
                random_state=self.gmm_random_state,
            ).fit(Z)
            if best is None or g.bic(Z) < best[0]:
                best = (g.bic(Z), g)
        return best[1]

    def fit_pca(self, X):
        n_samples, n_features = X.shape
        # Cap components by both n_samples-1 and n_features, and pca_n_components, min 1
        real_components = max(1, min(n_samples - 1, n_features, self.pca_n_components))
        print(f"Fitting PCA with {real_components} components")
        self.pca = PCA(
            n_components=real_components,
            whiten=self.pca_whiten,
            random_state=self.pca_random_state,
        )
        Z = self.pca.fit_transform(X)
        return Z

    def embed_folder(self, folder):
        X, paths = [], []
        for ext in self.file_extensions:
            paths.extend(sorted(Path(folder).glob(ext)))
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                x = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    f = self.vision_encoder(x).float()  # encoder handles normalization internally
                X.append(f.squeeze(0).cpu().numpy())
            except Exception as e:
                print(f"Error processing {p}: {e}")
        if not X:
            return np.empty((0, self.pca_n_components), dtype=np.float32)  # triggers empty check in fit()
        return np.vstack(X)
    def save_model(self, path):
        """Persist the exact sklearn objects using pickle."""
        model_data = {
            "pca": self.pca,
            "vae": None,
            "gmm": self.gmm,
            "tau": self.tau,
            "vision_encoder": None,
            "config": {
                "ft_reduction": self.ft_reduction,
                "pca_n_components": self.pca_n_components,
                "pca_whiten": self.pca_whiten,
                "pca_random_state": self.pca_random_state,
                # VAE config
                "vae_latent_dim": self.vae_latent_dim,
                "vae_hidden_dim": self.vae_hidden_dim,
                "vae_dropout": self.vae_dropout,
                "vae_epochs": self.vae_epochs,
                "vae_batch_size": self.vae_batch_size,
                "vae_lr": self.vae_lr,
                "vae_weight_decay": self.vae_weight_decay,
                "vae_beta": self.vae_beta,
                "vae_alpha": self.vae_alpha,
                "vae_early_stopping": self.vae_early_stopping,
                "vae_patience": self.vae_patience,
                "vae_seed": self.vae_seed,
                "vae_score_mode": self.vae_score_mode,
                "vae_hybrid_w_gmm": self.vae_hybrid_w_gmm,
                "vae_hybrid_w_recon": self.vae_hybrid_w_recon,
                "vae_finetune_encoder_layers": self.vae_finetune_encoder_layers,
                "vae_encoder_lr": self.vae_encoder_lr,
                "gmm_Ks": self.gmm_Ks,
                "gmm_covariance_type": self.gmm_covariance_type,
                "gmm_reg_covar": self.gmm_reg_covar,
                "gmm_max_iter": self.gmm_max_iter,
                "gmm_random_state": self.gmm_random_state,
                "train_val_split": self.train_val_split,
                "threshold_quantile": self.threshold_quantile,
                "file_extensions": self.file_extensions,
                "test_grid_cols": self.test_grid_cols,
            },
        }
        # Persist VAE state if present
        if self.vae is not None:
            first = self.vae.enc[0]
            d_in = first.in_features if hasattr(first, 'in_features') else None
            d_hid = first.out_features if hasattr(first, 'out_features') else None
            vae_info = {
                "state_dict": {k: v.cpu() for k, v in self.vae.state_dict().items()},
                "d_in": d_in,
                "d_lat": self.vae.mu.out_features,
                "d_hid": d_hid,
                "dropout": self.vae_dropout,
            }
            model_data["vae"] = vae_info
        # Persist finetuned vision encoder if present
        if self.vae_finetune_encoder_layers > 0 and self.vision_encoder is not None:
            try:
                encoder_info = {
                    "state_dict": {k: v.cpu() for k, v in self.vision_encoder.state_dict().items()},
                    "finetuned": True,
                }
                model_data["vision_encoder"] = encoder_info
            except Exception as e:
                warnings.warn(f"Failed to save finetuned vision encoder: {e}")
        # Persist hybrid normalization if any
        if self._hybrid_norm is not None:
            model_data["vae_hybrid_norm"] = self._hybrid_norm
        # Persist learned hybrid classifier if present
        if self._hybrid_scaler is not None and self._hybrid_clf is not None:
            model_data["vae_hybrid_scaler"] = self._hybrid_scaler
            model_data["vae_hybrid_clf"] = self._hybrid_clf
        # Persist Mahalanobis parameters if present
        if self._maha_mean is not None and self._maha_inv_cov is not None:
            model_data["maha_mean"] = self._maha_mean
            model_data["maha_inv_cov"] = self._maha_inv_cov
        # Persist reconstruction variance if present
        if self._recon_inv_sigma2 is not None:
            model_data["recon_inv_sigma2"] = self._recon_inv_sigma2
            model_data["recon_const"] = self._recon_const
        with open(path, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load_model(path, vision_encoder, preprocess, device):
        """Load model from file, handling both old joblib and new pickle formats."""
        blob = None
        
        # Try pickle first (new format)
        try:
            with open(path, 'rb') as f:
                blob = pickle.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load with pickle: {e}")
            
            # Fallback: try joblib with warnings suppressed
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    blob = load(path)
            except Exception as e2:
                raise RuntimeError(f"Failed to load model from {path}. Both pickle and joblib failed. "
                                 f"Please retrain the model. Errors: {e}, {e2}")
        
        if blob is None:
            raise RuntimeError(f"Failed to load model from {path}")
        
        model = Model(
            vision_encoder=vision_encoder,
            preprocess=preprocess,
            device=device,
        )
        model.pca = blob["pca"]
        model.gmm = blob["gmm"]
        model.tau = blob["tau"]
        if isinstance(blob, dict) and "config" in blob:
            cfg = blob["config"]
            # Use get with defaults to maintain backward compatibility
            model.ft_reduction = cfg.get("ft_reduction", getattr(model, "ft_reduction", "PCA")).upper()
            model.pca_n_components = cfg.get("pca_n_components", model.pca_n_components)
            model.pca_whiten = cfg.get("pca_whiten", model.pca_whiten)
            model.pca_random_state = cfg.get("pca_random_state", model.pca_random_state)

            # VAE
            model.vae_latent_dim = cfg.get("vae_latent_dim", model.vae_latent_dim)
            model.vae_hidden_dim = cfg.get("vae_hidden_dim", model.vae_hidden_dim)
            model.vae_dropout = cfg.get("vae_dropout", model.vae_dropout)
            model.vae_epochs = cfg.get("vae_epochs", model.vae_epochs)
            model.vae_batch_size = cfg.get("vae_batch_size", model.vae_batch_size)
            model.vae_lr = cfg.get("vae_lr", model.vae_lr)
            model.vae_weight_decay = cfg.get("vae_weight_decay", model.vae_weight_decay)
            model.vae_beta = cfg.get("vae_beta", model.vae_beta)
            model.vae_alpha = cfg.get("vae_alpha", model.vae_alpha)
            model.vae_early_stopping = cfg.get("vae_early_stopping", model.vae_early_stopping)
            model.vae_patience = cfg.get("vae_patience", model.vae_patience)
            model.vae_seed = cfg.get("vae_seed", model.vae_seed)
            model.vae_score_mode = cfg.get("vae_score_mode", model.vae_score_mode)
            model.vae_hybrid_w_gmm = cfg.get("vae_hybrid_w_gmm", model.vae_hybrid_w_gmm)
            model.vae_hybrid_w_recon = cfg.get("vae_hybrid_w_recon", model.vae_hybrid_w_recon)
            model.vae_finetune_encoder_layers = cfg.get("vae_finetune_encoder_layers", model.vae_finetune_encoder_layers)
            model.vae_encoder_lr = cfg.get("vae_encoder_lr", model.vae_encoder_lr)

            model.gmm_Ks = tuple(cfg.get("gmm_Ks", model.gmm_Ks))
            model.gmm_covariance_type = cfg.get("gmm_covariance_type", model.gmm_covariance_type)
            model.gmm_reg_covar = cfg.get("gmm_reg_covar", model.gmm_reg_covar)
            model.gmm_max_iter = cfg.get("gmm_max_iter", model.gmm_max_iter)
            model.gmm_random_state = cfg.get("gmm_random_state", model.gmm_random_state)

            model.train_val_split = cfg.get("train_val_split", model.train_val_split)
            model.threshold_quantile = cfg.get("threshold_quantile", model.threshold_quantile)

            model.file_extensions = tuple(cfg.get("file_extensions", model.file_extensions))
            model.test_grid_cols = cfg.get("test_grid_cols", model.test_grid_cols)

        # Rebuild VAE if saved and requested
        if isinstance(blob, dict) and blob.get("vae"):
            v = blob["vae"]
            try:
                d_in = v.get("d_in") or 512
                d_lat = v.get("d_lat") or (model.vae_latent_dim or model.pca_n_components)
                d_hid = v.get("d_hid") or model.vae_hidden_dim
                model.vae = VAEonCLIP(d_in=d_in, d_lat=d_lat, d_hid=d_hid, dropout=model.vae_dropout).to(device)
                model.vae.load_state_dict({k: (t if isinstance(t, torch.Tensor) else torch.tensor(t)) for k, t in v["state_dict"].items()})
                model.vae.eval()
            except Exception as e:
                warnings.warn(f"Failed to rebuild VAE from checkpoint: {e}")
        # Load finetuned vision encoder if saved
        if isinstance(blob, dict) and blob.get("vision_encoder"):
            ve = blob["vision_encoder"]
            try:
                if ve.get("finetuned", False):
                    # Restore finetuned encoder state
                    model.vision_encoder.load_state_dict({k: (t if isinstance(t, torch.Tensor) else torch.tensor(t)) for k, t in ve["state_dict"].items()})
                    model.vision_encoder.to(device).eval()
            except Exception as e:
                warnings.warn(f"Failed to reload finetuned vision encoder: {e}")
        # Load hybrid normalization if exists
        if isinstance(blob, dict) and blob.get("vae_hybrid_norm"):
            model._hybrid_norm = blob["vae_hybrid_norm"]
        # Load learned hybrid classifier if exists
        if isinstance(blob, dict) and blob.get("vae_hybrid_scaler"):
            model._hybrid_scaler = blob["vae_hybrid_scaler"]
        if isinstance(blob, dict) and blob.get("vae_hybrid_clf"):
            model._hybrid_clf = blob["vae_hybrid_clf"]
        # Load Mahalanobis parameters if exist
        if isinstance(blob, dict) and blob.get("maha_mean") is not None:
            model._maha_mean = blob["maha_mean"]
            model._maha_inv_cov = blob.get("maha_inv_cov")
        # Load reconstruction variance if exists
        if isinstance(blob, dict) and blob.get("recon_inv_sigma2") is not None:
            model._recon_inv_sigma2 = blob["recon_inv_sigma2"]
            model._recon_const = blob.get("recon_const", 0.0)
        return model
    
    def plot_with_model_2d(
        self, normal_paths, anomaly_folders, batch_size=64, reduction_method="pca", figsize=(14, 10)
    ):
        """
        Visualize model predictions in 2D space with decision boundaries.
        
        Works with both PCA and VAE feature reduction:
        - If ft_reduction="PCA": uses fitted PCA to project CLIP features directly
        - If ft_reduction="VAE": uses VAE to reduce to latent space, then projects to 2D
        
        Args:
            normal_paths: path(s) to normal images
            anomaly_folders: list of paths to anomaly folders
            batch_size: batch size for encoding
            reduction_method: '2d' reduction method ('pca', 'umap', etc.; default 'pca')
            figsize: figure size (default (14, 10))
        """
        try:
            import umap
            UMAP_AVAILABLE = True
        except ImportError:
            UMAP_AVAILABLE = False
       
        def collect(paths_or_root):
            if isinstance(paths_or_root, (list, tuple)):
                return [Path(p) for p in paths_or_root]
            root = Path(paths_or_root)
            exts = ("*.jpg","*.jpeg","*.png")
            return [p for ext in exts for p in root.rglob(ext)]

        def encode(paths):
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
                    feats.append(f.cpu()); keep += stash
            X = np.concatenate([t.numpy() for t in feats], axis=0) if feats else np.empty((0, 512))
            return X, keep

        def reduce_and_score(X):
            """Reduce features using PCA or VAE, compute scores, and return reduced features + scores."""
            if self.ft_reduction == "PCA":
                Z = self.pca.transform(X)
                scores = self.gmm.score_samples(Z)
            elif self.ft_reduction == "VAE":
                if self.vae is None:
                    raise RuntimeError("VAE not available for reduction.")
                self.vae.eval()
                with torch.no_grad():
                    Xt = torch.from_numpy(X).float().to(self.device)
                    x_hat, mu, logvar = self.vae(Xt)
                Z_lat = mu.cpu().numpy()
                
                # Compute scores based on selected mode
                if self.vae_score_mode == "gmm_mu":
                    if self.gmm is None:
                        raise RuntimeError("GMM not available for VAE 'gmm_mu' scoring.")
                    scores = self.gmm.score_samples(Z_lat)
                elif self.vae_score_mode == "recon_kl":
                    x_hat_np = x_hat.cpu().numpy()
                    x_np = Xt.cpu().numpy()
                    scores = self._compute_recon_loglik(x_np, x_hat_np)
                elif self.vae_score_mode == "hybrid":
                    if self.gmm is None:
                        raise RuntimeError("GMM not available for VAE 'hybrid' scoring.")
                    gmm_s = self.gmm.score_samples(Z_lat)
                    x_hat_np = x_hat.cpu().numpy()
                    x_np = Xt.cpu().numpy()
                    recon_s = self._compute_recon_loglik(x_np, x_hat_np)
                    if self._hybrid_norm:
                        eps = 1e-8
                        m_g, s_g = self._hybrid_norm.get("gmm_mean", 0.0), self._hybrid_norm.get("gmm_std", 1.0)
                        m_r, s_r = self._hybrid_norm.get("recon_mean", 0.0), self._hybrid_norm.get("recon_std", 1.0)
                        gmm_z = (gmm_s - m_g) / (s_g if s_g != 0 else 1.0)
                        recon_z = (recon_s - m_r) / (s_r if s_r != 0 else 1.0)
                        scores = self.vae_hybrid_w_gmm * gmm_z + self.vae_hybrid_w_recon * recon_z
                    else:
                        scores = self.vae_hybrid_w_gmm * gmm_s + self.vae_hybrid_w_recon * recon_s
                else:
                    raise ValueError(f"Unknown vae_score_mode: {self.vae_score_mode}")
                Z = Z_lat
            else:
                raise ValueError(f"Unknown ft_reduction: {self.ft_reduction}")
            
            auth_flags = scores >= self.tau
            return Z, auth_flags, scores

        # 1) Encode all image sets
        Xn, paths_n = encode(collect(normal_paths))
        an_sets = []
        for folder in (anomaly_folders or []):
            Xa, pa = encode(collect(folder))
            an_sets.append((folder, Xa, pa))

        # 2) Reduce and score all sets
        Zn, auth_n, scores_n = reduce_and_score(Xn)
        an_reduced = []
        for folder, Xa, pa in an_sets:
            Za, auth_a, scores_a = reduce_and_score(Xa)
            an_reduced.append((folder, Za, pa, auth_a, scores_a))

        # 3) Project reduced features to 2D
        all_Z = np.vstack([Zn] + [Za for _, Za, _, _, _ in an_reduced]) if an_reduced else Zn
        all_scores = np.concatenate([scores_n] + [scores_a for _, _, _, _, scores_a in an_reduced]) if an_reduced else scores_n
        
        if reduction_method.lower() == "pca":
            reducer = PCA(n_components=min(2, all_Z.shape[1]))
            all_Z_2d = reducer.fit_transform(all_Z)
            method_info = f"PCA ({reducer.explained_variance_ratio_[0]:.1%}, {reducer.explained_variance_ratio_[1] if len(reducer.explained_variance_ratio_) > 1 else 0:.1%})"
        elif reduction_method.lower() == "umap":
            if not UMAP_AVAILABLE:
                warnings.warn("UMAP not available; falling back to PCA.")
                reducer = PCA(n_components=2)
                all_Z_2d = reducer.fit_transform(all_Z)
                method_info = "PCA (fallback)"
            else:
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                all_Z_2d = reducer.fit_transform(all_Z)
                method_info = "UMAP"
        else:
            raise ValueError(f"Unknown reduction_method: {reduction_method}")
        
        # Split back
        idx = 0
        Zn_2d = all_Z_2d[:len(Zn)]
        idx = len(Zn)
        
        an_2d = []
        for folder, Za, pa, auth_a, scores_a in an_reduced:
            n_a = len(Za)
            Za_2d = all_Z_2d[idx:idx+n_a]
            idx += n_a
            an_2d.append((folder, Za_2d, pa, auth_a, scores_a))

        # 4) Plot with decision boundary
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot decision boundary (score = tau)
        x_range = all_Z_2d[:, 0].max() - all_Z_2d[:, 0].min()
        y_range = all_Z_2d[:, 1].max() - all_Z_2d[:, 1].min()
        
        # Add proportional padding (10% on each side)
        x_pad = max(x_range * 0.1, 0.1)  # min padding of 0.1 for single-point case
        y_pad = max(y_range * 0.1, 0.1)
        
        x_min, x_max = all_Z_2d[:, 0].min() - x_pad, all_Z_2d[:, 0].max() + x_pad
        y_min, y_max = all_Z_2d[:, 1].min() - y_pad, all_Z_2d[:, 1].max() + y_pad
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        
        # Approximate boundary by coloring background based on scores from train data
        if len(Zn_2d) > 1:
            try:
                from scipy.interpolate import griddata
                Z_bg = griddata(all_Z_2d, all_scores, (xx, yy), method='linear', fill_value=np.nan)
                levels = [self.tau]
                ax.contourf(xx, yy, Z_bg, levels=[Z_bg.min(), self.tau], colors=['mistyrose'], alpha=0.3, zorder=0)
                ax.contour(xx, yy, Z_bg, levels=levels, colors='gray', linestyles='--', linewidths=1.5, alpha=0.7, zorder=1)
            except Exception as e:
                warnings.warn(f"Could not plot decision boundary: {e}")
        
        # Plot points
        mask_n = np.array(auth_n, dtype=bool)
        ax.scatter(Zn_2d[mask_n, 0], Zn_2d[mask_n, 1], s=20, alpha=0.7, color="green", label="normal (auth)", zorder=3)
        ax.scatter(Zn_2d[~mask_n, 0], Zn_2d[~mask_n, 1], s=20, alpha=0.7, color="red", label="normal (not auth)", zorder=3)
        
        markers = ["*", "X", "P", "^", "v", "s", "D", "p", "h", "8"]
        for i, (folder, Za_2d, pa, auth_a, scores_a) in enumerate(an_2d):
            key = Path(folder).name
            mask_a = np.array(auth_a, dtype=bool)
            ax.scatter(Za_2d[mask_a, 0], Za_2d[mask_a, 1],
                    marker=markers[i % len(markers)], s=100, facecolors="none",
                    edgecolors="green", linewidths=1.5, alpha=0.8, label=f"{key} (auth)", zorder=3)
            ax.scatter(Za_2d[~mask_a, 0], Za_2d[~mask_a, 1],
                    marker=markers[i % len(markers)], s=100, facecolors="none",
                    edgecolors="red", linewidths=1.5, alpha=0.8, label=f"{key} (not auth)", zorder=3)
        
        title_suffix = f"({self.ft_reduction})"
        if self.ft_reduction == "VAE":
            title_suffix += f" [latent={self.vae.mu.out_features}, score_mode={self.vae_score_mode}]"
        ax.set_title(f"Model 2D Visualization {title_suffix} | {method_info}")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc='best')
        plt.tight_layout()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.tight_layout()
        plt.show()
    
    # Keep backward compatibility alias
    def plot_with_model_pca_2d(self, normal_paths, anomaly_folders, batch_size=64):
        """Backward compatibility wrapper. Use plot_with_model_2d() instead."""
        return self.plot_with_model_2d(normal_paths, anomaly_folders, batch_size=batch_size, reduction_method="pca")