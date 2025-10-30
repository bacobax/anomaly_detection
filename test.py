import numpy as np
import torch, open_clip
from models.model import Model
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train import VisualEncoder
from PIL import Image

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Enable deterministic algorithms
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# For MPS (Mac), set environment variable for determinism
import os
os.environ["PYTHONHASHSEED"] = str(SEED)

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")



if __name__ == "__main__":
    clip_model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    enc = VisualEncoder(clip_model.visual, normalize=False).eval()

    

    # Summary prints to show results and help debugging
    my_model = Model.load(
        vision_encoder=enc,
        path="weights/model_VAE_recon_kl_score.pth",
        preprocess=preprocess,
        device=device,
    )
    for i in range(3):
        print(f"--- Demo run {i+1} ---")
        my_model.test_folder("trento_house_data", output_folder="custom_test")



    # Demo: run is_authorized() on one sample from training_data_faces if available
