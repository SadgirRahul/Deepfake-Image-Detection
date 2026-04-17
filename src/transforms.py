"""
Image Preprocessing & Augmentation Pipelines
=============================================

Defines Albumentations-based transform pipelines for training, validation,
and ELA (Error Level Analysis) images used in the deepfake detection model.

Pipelines:
    - Training:   augmentations + ImageNet normalization
    - Validation:  resize + ImageNet normalization (no augmentation)
    - ELA:         resize + simple 0.5-centered normalization

All pipelines output PyTorch tensors via ToTensorV2.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet statistics (used for pretrained EfficientNet backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Simple centered normalization for ELA channels
ELA_MEAN = [0.5, 0.5, 0.5]
ELA_STD = [0.5, 0.5, 0.5]


def get_train_transforms(image_size=224):
    """
    Build the training augmentation + preprocessing pipeline.

    Augmentations are applied to make the model robust to variations in
    orientation, lighting, and image quality — but are intentionally kept
    mild so they don't destroy forensic artifacts the model needs to learn.

    Args:
        image_size (int): Target height and width. Default is 224
            (EfficientNet-B0 input size).

    Returns:
        albumentations.Compose: The composed transform pipeline.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.3,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_valid_transforms(image_size=224):
    """
    Build the validation / test preprocessing pipeline.

    No augmentations — only deterministic resize and normalisation so that
    evaluation results are reproducible.

    Args:
        image_size (int): Target height and width. Default is 224.

    Returns:
        albumentations.Compose: The composed transform pipeline.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_ela_transforms(image_size=224):
    """
    Build the preprocessing pipeline for ELA (Error Level Analysis) images.

    Uses a simple 0.5-centered normalization instead of ImageNet stats
    because ELA images have a fundamentally different intensity distribution
    (mostly dark with bright spots at manipulated regions).

    Args:
        image_size (int): Target height and width. Default is 224.

    Returns:
        albumentations.Compose: The composed transform pipeline.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=ELA_MEAN, std=ELA_STD),
        ToTensorV2(),
    ])
