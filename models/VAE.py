
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Tuple, Union
from tqdm import tqdm

# Small VAE over 512-d CLIP features
class VAEonCLIP(nn.Module):
    def __init__(self, d_in=512, d_lat=32, d_hid=512, dropout=0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Dropout(dropout),
        )
        self.mu = nn.Linear(d_hid, d_lat)
        self.logvar = nn.Linear(d_hid, d_lat)
        self.dec = nn.Sequential(
            nn.Linear(d_lat, d_hid), nn.ReLU(),
            nn.Linear(d_hid, d_in),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.dec(z)  # we'll handle norm in loss

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

def vae_loss(x, x_hat, mu, logvar, beta=4.0, alpha=0.5, eps=1e-8):
    # x, x_hat: (B, 512) CLIP embeddings (assume unit-norm)
    # combine MSE + cosine; and KL to N(0,I)
    mse = F.mse_loss(x_hat, x, reduction='mean')
    x_hat_norm = F.normalize(x_hat, dim=-1)
    x_norm = F.normalize(x, dim=-1)
    cos = 1.0 - (x_hat_norm * x_norm).sum(-1).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    recon = alpha*mse + (1-alpha)*cos
    return recon + beta*kl, {'mse':mse.item(), 'cos':cos.item(), 'kl':kl.item()}


def vae_loss_per_sample(x, x_hat, mu, logvar, beta=4.0, alpha=0.5):
        """Compute per-sample VAE loss components on features.

        Returns:
            total: (B,) tensor of total loss per sample (alpha*mse + (1-alpha)*cos + beta*kl)
            parts: dict with (B,) tensors for 'mse', 'cos', 'kl'
        Notes:
            - mse is mean over feature dims per sample
            - cos is 1 - cosine similarity per sample
            - kl is mean over latent dims per sample
        """
        # Per-sample MSE (average over feature dim)
        mse_ps = (x_hat - x).pow(2).mean(dim=-1)
        # Per-sample cosine dissimilarity
        xh_n = F.normalize(x_hat, dim=-1)
        x_n = F.normalize(x, dim=-1)
        cos_ps = 1.0 - (xh_n * x_n).sum(dim=-1)
        # Per-sample KL (mean over latent dims)
        kl_sum = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        kl_ps = kl_sum / mu.shape[-1]
        total = alpha*mse_ps + (1-alpha)*cos_ps + beta*kl_ps
        return total, {"mse": mse_ps, "cos": cos_ps, "kl": kl_ps}


@torch.no_grad()
def vae_reduce_mu(model: VAEonCLIP, X: Union[torch.Tensor, 'np.ndarray']) -> torch.Tensor:
    """
    Deterministic feature reduction using the encoder mean (mu).
    Accepts a (N, d_in) tensor/ndarray and returns (N, d_lat) tensor on CPU.
    """
    if not torch.is_tensor(X):
        import numpy as np
        X = torch.from_numpy(np.asarray(X)).float()
    model.eval()
    device = next(model.parameters()).device
    mus = []
    for i in range(0, X.shape[0], 4096):
        xb = X[i:i+4096].to(device)
        mu, logvar = model.encode(xb)
        mus.append(mu.cpu())
    return torch.cat(mus, dim=0) if mus else torch.empty((0, model.mu.out_features))


def train_vae(
    X_train: Union[torch.Tensor, 'np.ndarray'],
    X_val: Optional[Union[torch.Tensor, 'np.ndarray']] = None,
    *,
    d_in: Optional[int] = None,
    d_lat: int = 32,
    d_hid: int = 512,
    dropout: float = 0.2,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    beta: float = 4.0,
    alpha: float = 0.5,
    early_stopping: bool = True,
    patience: int = 5,
    device: Union[str, torch.device] = 'cpu',
    seed: int = 42,
    verbose: bool = True,
    vision_encoder: Optional[torch.nn.Module] = None,
    preprocess: Optional[callable] = None,
    finetune_encoder_layers: int = 0,
    encoder_lr: float = 1e-4,
) -> Tuple[VAEonCLIP, Dict[str, list], Optional[torch.nn.Module]]:
    """
    Train a small VAE on pre-extracted features or on-the-fly from images.

    Inputs:
      - X_train: (N, d_in) numpy array/tensor OR (N, C, H, W) image tensors if encoder provided
      - X_val: optional validation set for early stopping
      - d_in: feature dimension (inferred if not given)
      - vision_encoder: if provided, encode images on-the-fly during training
      - preprocess: preprocess function (required if vision_encoder is provided)
      - finetune_encoder_layers: number of last layers to unfreeze for finetuning (0=frozen)
      - encoder_lr: learning rate for encoder parameters (if finetuning)

    Returns: (trained_model, history, vision_encoder_updated)
    """
    import numpy as np

    if not torch.is_tensor(X_train):
        X_train = torch.from_numpy(np.asarray(X_train)).float()
    if X_val is not None and not torch.is_tensor(X_val):
        X_val = torch.from_numpy(np.asarray(X_val)).float()

    if isinstance(device, str):
        device = torch.device(device)

    # Determine if we're encoding on-the-fly or using pre-computed features
    use_encoder = vision_encoder is not None
    
    # Infer d_in from first sample if not provided
    if d_in is None:
        if use_encoder:
            # For images: get feature dim by running one batch through encoder
            with torch.no_grad():
                sample = X_train[:1].to(device)
                feats = vision_encoder(sample).float()
                d_in = int(feats.shape[-1])
        else:
            d_in = int(X_train.shape[1])

    torch.manual_seed(seed)

    model = VAEonCLIP(d_in=d_in, d_lat=d_lat, d_hid=d_hid, dropout=dropout).to(device)
    
    # Setup optimizer(s)
    vae_params = model.parameters()
    if use_encoder and finetune_encoder_layers > 0:
        # Freeze all encoder layers except the last k
        vision_encoder = vision_encoder.to(device)
        encoder_params_to_finetune = []
        
        # Get all named parameters in vision_encoder
        encoder_named_params = list(vision_encoder.named_parameters())
        
        # Unfreeze only the last finetune_encoder_layers
        for i, (name, param) in enumerate(encoder_named_params):
            if i < len(encoder_named_params) - finetune_encoder_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
                encoder_params_to_finetune.append(param)
        
        # Create separate param groups for encoder and VAE with different learning rates
        opt = torch.optim.Adam([
            {'params': vae_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': encoder_params_to_finetune, 'lr': encoder_lr, 'weight_decay': weight_decay}
        ])
        if verbose:
            print(f"[VAE] Finetuning last {finetune_encoder_layers} encoder layers (encoder_lr={encoder_lr})")
    elif use_encoder:
        # Encoder frozen
        vision_encoder = vision_encoder.to(device).eval()
        opt = torch.optim.Adam(vae_params, lr=lr, weight_decay=weight_decay)
        if verbose:
            print("[VAE] Vision encoder frozen (no finetuning)")
    else:
        # No encoder
        opt = torch.optim.Adam(vae_params, lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(X_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = None
    if X_val is not None:
        val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False, drop_last=False)

    history = {"loss": [], "val_loss": []}
    best_val = float('inf')
    best_state = None
    best_encoder_state = None
    no_improve = 0

    pbar = tqdm(range(1, epochs+1), desc="[VAE Training]", disable=not verbose)
    
    for ep in pbar:
        model.train()
        if use_encoder:
            if finetune_encoder_layers > 0:
                vision_encoder.train()
            else:
                vision_encoder.eval()
        
        run_loss = 0.0
        n_batches = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            
            # Encode on-the-fly if encoder provided
            # Note: VisualEncoder handles normalization, so we don't normalize here
            if use_encoder:
                with torch.no_grad() if finetune_encoder_layers == 0 else torch.enable_grad():
                    feats = vision_encoder(xb).float()
            else:
                feats = xb
            
            opt.zero_grad(set_to_none=True)
            x_hat, mu, logvar = model(feats)
            loss, parts = vae_loss(feats, x_hat, mu, logvar, beta=beta, alpha=alpha)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            n_batches += 1
        train_epoch_loss = run_loss / max(1, n_batches)
        history["loss"].append(train_epoch_loss)

        val_epoch_loss = None
        if val_loader is not None:
            model.eval()
            if use_encoder:
                vision_encoder.eval()
            vloss = 0.0
            vb = 0
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(device)
                    
                    # Encode validation data
                    # Note: VisualEncoder handles normalization, so we don't normalize here
                    if use_encoder:
                        feats = vision_encoder(xb).float()
                    else:
                        feats = xb
                    
                    x_hat, mu, logvar = model(feats)
                    loss, parts = vae_loss(feats, x_hat, mu, logvar, beta=beta, alpha=alpha)
                    vloss += loss.item()
                    vb += 1
            val_epoch_loss = vloss / max(1, vb)
            history["val_loss"].append(val_epoch_loss)

            # Early stopping
            if val_epoch_loss < best_val - 1e-6:
                best_val = val_epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if use_encoder and finetune_encoder_layers > 0:
                    best_encoder_state = {k: v.detach().cpu().clone() for k, v in vision_encoder.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if early_stopping and no_improve >= patience:
                    pbar.close()
                    if verbose:
                        print(f"Early stopping at epoch {ep}: best val loss {best_val:.6f}")
                    break

        # Update progress bar with loss values
        if val_epoch_loss is not None:
            pbar.set_postfix({"loss": f"{train_epoch_loss:.6f}", "val_loss": f"{val_epoch_loss:.6f}"})
        else:
            pbar.set_postfix({"loss": f"{train_epoch_loss:.6f}"})

    # Load best state if we tracked validation
    if best_state is not None:
        model.load_state_dict(best_state)
    if use_encoder and best_encoder_state is not None:
        vision_encoder.load_state_dict(best_encoder_state)

    return model, history, (vision_encoder if use_encoder else None)