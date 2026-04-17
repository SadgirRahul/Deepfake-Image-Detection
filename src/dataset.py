"""
Custom PyTorch Dataset & DataLoaders for Deepfake Detection
============================================================

Provides a Dataset class that loads real/fake images from the directory
structure, optionally computes ELA features, and returns tensors ready
for training or inference.

Expected directory layout::

    <root_dir>/
    ├── train/
    │   ├── real/   (label 0)
    │   └── fake/   (label 1)
    ├── valid/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
"""

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from src.features import compute_ela
from src.transforms import (
    get_train_transforms,
    get_valid_transforms,
    get_ela_transforms,
)

# Allowed image file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# =============================================================================
# Dataset
# =============================================================================

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for binary deepfake detection (Real vs Fake).

    Each sample is returned as a tuple ``(tensor, label)`` where *tensor*
    is either a 3-channel RGB tensor or a 6-channel RGB+ELA tensor,
    depending on ``use_ela``.

    Args:
        root_dir (str): Path to the dataset root (e.g. ``"F:/Dataset"``).
        split (str): One of ``"train"``, ``"valid"``, or ``"test"``.
        transform: Albumentations pipeline for the RGB image.
        ela_transform: Albumentations pipeline for the ELA image.
        use_ela (bool): If True, compute ELA and concatenate as extra
            channels (resulting in a 6-channel tensor).
    """

    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        ela_transform=None,
        use_ela=True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.ela_transform = ela_transform
        self.use_ela = use_ela

        # Collect (image_path, label) pairs
        self.samples = []
        split_dir = os.path.join(root_dir, split)

        # Real images → label 0
        real_dir = os.path.join(split_dir, "real")
        if os.path.isdir(real_dir):
            for fname in sorted(os.listdir(real_dir)):
                if os.path.splitext(fname)[1].lower() in ALLOWED_EXTENSIONS:
                    self.samples.append((os.path.join(real_dir, fname), 0))

        # Fake images → label 1
        fake_dir = os.path.join(split_dir, "fake")
        if os.path.isdir(fake_dir):
            for fname in sorted(os.listdir(fake_dir)):
                if os.path.splitext(fname)[1].lower() in ALLOWED_EXTENSIONS:
                    self.samples.append((os.path.join(fake_dir, fname), 1))

        if len(self.samples) == 0:
            print(f"⚠ No images found in {split_dir}. Check directory structure.")

        print(
            f"📂 [{split.upper():5s}] Loaded {len(self.samples):,} images  "
            f"(Real: {sum(1 for _, l in self.samples if l == 0):,}  |  "
            f"Fake: {sum(1 for _, l in self.samples if l == 1):,})"
        )

    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # -----------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Load one sample and return ``(tensor, label)``.

        If the image at *idx* cannot be read, the method silently moves
        to the next valid index (wrapping around if necessary) so that
        a single corrupt file does not crash the entire training run.
        """
        # Try up to len(self) times to find a readable image
        for attempt in range(len(self.samples)):
            actual_idx = (idx + attempt) % len(self.samples)
            img_path, label = self.samples[actual_idx]

            try:
                # ------ Load RGB image ------
                bgr = cv2.imread(img_path)
                if bgr is None:
                    raise IOError(f"cv2.imread returned None for {img_path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # ------ Compute ELA (optional) ------
                if self.use_ela:
                    ela_bgr = compute_ela(bgr)  # pass array directly
                    ela_rgb = cv2.cvtColor(ela_bgr, cv2.COLOR_BGR2RGB)

                # ------ Apply transforms ------
                if self.transform:
                    augmented = self.transform(image=rgb)
                    rgb_tensor = augmented["image"]  # (3, H, W) float32
                else:
                    rgb_tensor = torch.from_numpy(
                        rgb.transpose(2, 0, 1).astype("float32") / 255.0
                    )

                if self.use_ela:
                    if self.ela_transform:
                        ela_aug = self.ela_transform(image=ela_rgb)
                        ela_tensor = ela_aug["image"]  # (3, H, W)
                    else:
                        ela_tensor = torch.from_numpy(
                            ela_rgb.transpose(2, 0, 1).astype("float32") / 255.0
                        )

                    # Concatenate RGB + ELA → 6-channel tensor
                    tensor = torch.cat([rgb_tensor, ela_tensor], dim=0)
                else:
                    tensor = rgb_tensor

                return tensor, label

            except Exception as e:
                if attempt == 0:
                    print(f"⚠ Skipping corrupt image [{actual_idx}]: {img_path} — {e}")
                continue

        # Absolute fallback: return a black tensor
        channels = 6 if self.use_ela else 3
        print(f"❌ Could not load any image near index {idx}, returning zeros.")
        return torch.zeros(channels, 224, 224), 0


# =============================================================================
# DataLoaders factory
# =============================================================================

def get_dataloaders(
    root_dir="F:/Dataset",
    batch_size=32,
    image_size=224,
    use_ela=True,
    num_workers=2,
):
    """
    Create train / valid / test DataLoaders with appropriate transforms.

    Args:
        root_dir (str): Dataset root directory.
        batch_size (int): Batch size for all loaders.
        image_size (int): Target image size (height = width).
        use_ela (bool): Whether to include ELA channels.
        num_workers (int): Number of parallel data-loading workers.

    Returns:
        dict: ``{"train": DataLoader, "valid": DataLoader, "test": DataLoader}``
    """
    # Build transform pipelines
    train_tf = get_train_transforms(image_size)
    valid_tf = get_valid_transforms(image_size)
    ela_tf = get_ela_transforms(image_size) if use_ela else None

    # Create datasets
    train_ds = DeepfakeDataset(
        root_dir, split="train",
        transform=train_tf, ela_transform=ela_tf, use_ela=use_ela,
    )
    valid_ds = DeepfakeDataset(
        root_dir, split="valid",
        transform=valid_tf, ela_transform=ela_tf, use_ela=use_ela,
    )
    test_ds = DeepfakeDataset(
        root_dir, split="test",
        transform=valid_tf, ela_transform=ela_tf, use_ela=use_ela,
    )

    # Build DataLoaders
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    print(
        f"\n📦 DataLoaders ready  —  "
        f"Train: {len(train_loader)} batches  |  "
        f"Valid: {len(valid_loader)} batches  |  "
        f"Test: {len(test_loader)} batches  "
        f"(batch_size={batch_size})\n"
    )

    return {"train": train_loader, "valid": valid_loader, "test": test_loader}


# =============================================================================
# Quick sanity check
# =============================================================================

if __name__ == "__main__":
    loaders = get_dataloaders(root_dir="F:/Dataset", batch_size=4, use_ela=True)

    # Grab one batch from training loader
    images, labels = next(iter(loaders["train"]))
    print(f"✅ Batch tensor shape : {images.shape}")   # expect (4, 6, 224, 224)
    print(f"✅ Labels             : {labels.tolist()}")
    print(f"✅ Tensor dtype       : {images.dtype}")
    print(f"✅ Value range        : [{images.min():.3f}, {images.max():.3f}]")
