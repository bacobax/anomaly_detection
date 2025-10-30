
# dataset_clip_features.py
import os
from pathlib import Path
from typing import Sequence, Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ClipFeatureDataset(Dataset):
    """
    Dataset that returns raw image tensors (preprocessed) without encoding.
    Encoding will be performed by the training loop, allowing for encoder finetuning.
    Each __getitem__ returns a preprocessed image tensor and optionally the image path.

    Args:
        root: folder with training images
        preprocess: CLIP preprocess callable (e.g., from open_clip.create_model_and_transforms)
        exts: image filename patterns to include
        return_paths: if True, also return the image path alongside the image tensor
    """
    def __init__(
        self,
        root: str,
        preprocess,
        exts: Sequence[str] = (".jpg", ".jpeg", ".png"),
        return_paths: bool = False,
    ):
        self.root = Path(root)
        self.preprocess = preprocess
        self.return_paths = return_paths

        # Collect image paths (sorted for determinism; DataLoader handles shuffling)
        paths: List[Path] = []
        for p in self.root.rglob("*"):
            if p.suffix.lower() in exts:
                paths.append(p)
        self.paths = sorted(paths)

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {self.root} with exts {exts}")


    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        # Return preprocessed image tensor (without encoding)
        img_tensor = self.preprocess(img)
        
        return (img_tensor, str(img_path)) if self.return_paths else img_tensor


def make_loader(
    root: str,
    preprocess,
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    return_paths: bool = False,
) -> Tuple[DataLoader]:
    """
    Convenience wrapper that returns a DataLoader of preprocessed image tensors.
    Encoding will be done during training with optional encoder finetuning.
    """
    ds = ClipFeatureDataset(
        root=root,
        preprocess=preprocess,
        return_paths=return_paths,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,          # <-- shuffles order each epoch
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader