"""
Deepfake Detection Model — EfficientNet-B0 with Custom Head
=============================================================

Uses a pretrained EfficientNet-B0 backbone as a feature extractor,
with a modified first convolution layer (6 input channels when ELA
is enabled) and a custom binary classification head.

Architecture::

    Input (3 or 6 channels)
        │
        ▼
    EfficientNet-B0 Backbone (pretrained, partially frozen)
        │  1280-dim feature vector
        ▼
    AdaptiveAvgPool2d(1)
        │
        ▼
    Dropout(0.3) → Linear(1280, 512) → BatchNorm → ReLU
        │
        ▼
    Dropout(0.3) → Linear(512, 1)   →  raw logit
        │
        ▼
    BCEWithLogitsLoss (during training)
    Sigmoid          (during inference)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class DeepfakeDetector(nn.Module):
    """
    Binary classifier for deepfake detection built on EfficientNet-B0.

    Args:
        num_classes (int): Number of output units.  1 for binary
            classification with BCEWithLogitsLoss.
        use_ela (bool): If True, the first conv layer accepts 6 channels
            (RGB + ELA).  If False, standard 3-channel RGB input.
        pretrained (bool): Whether to load ImageNet-pretrained weights
            for the backbone.
    """

    def __init__(self, num_classes=1, use_ela=True, pretrained=True):
        super().__init__()
        self.use_ela = use_ela

        # ------------------------------------------------------------------
        # Load pretrained EfficientNet-B0
        # ------------------------------------------------------------------
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)

        # ------------------------------------------------------------------
        # Modify first conv layer for 6-channel input (RGB + ELA)
        # ------------------------------------------------------------------
        if use_ela:
            original_conv = base_model.features[0][0]  # Conv2d(3, 32, ...)
            new_conv = nn.Conv2d(
                in_channels=6,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None,
            )

            with torch.no_grad():
                # Copy pretrained RGB weights to the first 3 channels
                new_conv.weight[:, :3, :, :] = original_conv.weight.clone()
                # Initialize ELA channels by copying RGB weights
                # (gives ELA channels a reasonable starting point)
                new_conv.weight[:, 3:, :, :] = original_conv.weight.clone()
                if original_conv.bias is not None:
                    new_conv.bias = nn.Parameter(original_conv.bias.clone())

            base_model.features[0][0] = new_conv

        # ------------------------------------------------------------------
        # Extract backbone (feature extractor)
        # ------------------------------------------------------------------
        self.backbone = base_model.features  # all convolutional blocks
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ------------------------------------------------------------------
        # Custom classifier head
        # ------------------------------------------------------------------
        # EfficientNet-B0 outputs 1280-dimensional feature vectors
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    # -----------------------------------------------------------------
    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape ``(B, C, H, W)`` where
                C is 3 (RGB) or 6 (RGB + ELA).

        Returns:
            Tensor: Raw logits of shape ``(B, 1)``.  Apply ``torch.sigmoid``
                for probability output.
        """
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # (B, 1280)
        x = self.classifier(x)
        return x

    # -----------------------------------------------------------------
    def freeze_backbone(self):
        """Freeze all backbone parameters (for Phase-1 head-only training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("❄️  Backbone frozen — only classifier head is trainable.")

    # -----------------------------------------------------------------
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters (for Phase-2 fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔥 Backbone unfrozen — full network is trainable.")


# =============================================================================
# Helper factory
# =============================================================================

def get_model(use_ela=True, pretrained=True, device="cpu"):
    """
    Create a DeepfakeDetector and move it to the specified device.

    Args:
        use_ela (bool): Enable 6-channel input (RGB + ELA).
        pretrained (bool): Load ImageNet weights.
        device (str): Target device (``"cpu"`` or ``"cuda"``).

    Returns:
        DeepfakeDetector: The model on the requested device.
    """
    model = DeepfakeDetector(
        num_classes=1,
        use_ela=use_ela,
        pretrained=pretrained,
    )
    model = model.to(device)

    # Print parameter summary
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"🧠 Model created  —  "
        f"Total params: {total:,}  |  Trainable: {trainable:,}  |  "
        f"Device: {device}  |  ELA: {use_ela}"
    )

    return model


# =============================================================================
# Quick check
# =============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test with ELA (6-channel)
    model = get_model(use_ela=True, device=device)
    dummy = torch.randn(2, 6, 224, 224).to(device)
    out = model(dummy)
    print(f"✅ ELA mode  — Input: {dummy.shape} → Output: {out.shape}")

    # Test without ELA (3-channel)
    model_rgb = get_model(use_ela=False, device=device)
    dummy_rgb = torch.randn(2, 3, 224, 224).to(device)
    out_rgb = model_rgb(dummy_rgb)
    print(f"✅ RGB mode  — Input: {dummy_rgb.shape} → Output: {out_rgb.shape}")
