"""
Single-Image Prediction Pipeline for Deepfake Detection
=========================================================

Loads the trained model and provides a simple API for predicting
whether a single image is real or fake. Supports both file paths
and numpy arrays (for Gradio / webcam integration).

Usage::

    # From Python
    from src.predict import DeepfakePredictor
    predictor = DeepfakePredictor()
    result = predictor.predict("path/to/image.jpg")

    # From CLI
    python -m src.predict --image path/to/image.jpg
"""

import os
import argparse
import time

import cv2
import numpy as np
import torch

from src.model import DeepfakeDetector
from src.features import compute_ela, compute_fft_spectrum, compute_edge_map
from src.transforms import get_valid_transforms, get_ela_transforms


class DeepfakePredictor:
    """
    End-to-end predictor for deepfake detection.

    Handles image loading, preprocessing, ELA generation, model inference,
    and feature visualization — all in one class.

    Args:
        model_path (str): Path to the saved model weights (``.pth``).
        device (str or None): Target device.  Auto-detects if None.
        use_ela (bool): Whether the model expects 6-channel (RGB+ELA) input.
        image_size (int): Expected input size for the model.
    """

    def __init__(
        self,
        model_path="f:/CV_CP/models/best_model.pth",
        device=None,
        use_ela=True,
        image_size=224,
    ):
        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_ela = use_ela
        self.image_size = image_size

        # Load model
        self.model = DeepfakeDetector(num_classes=1, use_ela=use_ela, pretrained=False)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Model loaded from: {model_path}")
        else:
            print(f"⚠ Model file not found: {model_path} — using random weights!")

        self.model.to(self.device)
        self.model.eval()

        # Transforms (validation — no augmentation)
        self.rgb_transform = get_valid_transforms(image_size)
        self.ela_transform = get_ela_transforms(image_size) if use_ela else None

    # -----------------------------------------------------------------
    def _load_rgb(self, image_input):
        """
        Convert input to an RGB numpy array.

        Args:
            image_input: File path (str) or numpy array (BGR or RGB).

        Returns:
            tuple: ``(rgb_array, bgr_array)`` both as uint8 numpy arrays.
        """
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image_input}")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                # Grayscale → BGR
                bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                # RGBA → BGR
                bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            else:
                # Assume RGB from Gradio → convert to BGR for OpenCV
                bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, bgr

    # -----------------------------------------------------------------
    def predict(self, image_input):
        """
        Predict whether an image is real or fake.

        Args:
            image_input: File path (str) or numpy array (RGB from Gradio,
                or BGR from cv2).

        Returns:
            dict: Prediction results::

                {
                    "label": "FAKE" or "REAL",
                    "confidence": 94.2,       # percentage
                    "probability": 0.942,     # raw sigmoid
                    "is_fake": True,
                    "inference_time_ms": 45.3,
                }
        """
        start = time.time()

        rgb, bgr = self._load_rgb(image_input)

        # --- Apply RGB transforms ---
        augmented = self.rgb_transform(image=rgb)
        rgb_tensor = augmented["image"]  # (3, 224, 224)

        # --- ELA (optional) ---
        if self.use_ela:
            ela_bgr = compute_ela(bgr)
            ela_rgb = cv2.cvtColor(ela_bgr, cv2.COLOR_BGR2RGB)
            ela_aug = self.ela_transform(image=ela_rgb)
            ela_tensor = ela_aug["image"]  # (3, 224, 224)
            input_tensor = torch.cat([rgb_tensor, ela_tensor], dim=0)  # (6, 224, 224)
        else:
            input_tensor = rgb_tensor  # (3, 224, 224)

        # --- Inference ---
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(input_tensor)
            prob = torch.sigmoid(logit).item()

        elapsed_ms = (time.time() - start) * 1000

        is_fake = prob > 0.5
        confidence = prob * 100 if is_fake else (1 - prob) * 100

        return {
            "label": "FAKE" if is_fake else "REAL",
            "confidence": round(confidence, 2),
            "probability": round(prob, 4),
            "is_fake": is_fake,
            "inference_time_ms": round(elapsed_ms, 1),
        }

    # -----------------------------------------------------------------
    def get_feature_visualizations(self, image_input):
        """
        Generate CV feature maps for display in the Gradio app.

        Args:
            image_input: File path or numpy array.

        Returns:
            dict: Feature images (all resized to image_size × image_size)::

                {
                    "ela": RGB numpy array (uint8),
                    "fft": RGB numpy array (uint8),
                    "edges": RGB numpy array (uint8),
                }
        """
        _, bgr = self._load_rgb(image_input)
        size = (self.image_size, self.image_size)

        # ELA
        ela = compute_ela(bgr)
        ela = cv2.resize(ela, size)
        ela_rgb = cv2.cvtColor(ela, cv2.COLOR_BGR2RGB)

        # FFT
        fft = compute_fft_spectrum(bgr)
        fft = cv2.resize(fft, size)
        fft_rgb = cv2.cvtColor(fft, cv2.COLOR_GRAY2RGB)

        # Edges
        edges = compute_edge_map(bgr)
        edges = cv2.resize(edges, size)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return {
            "ela": ela_rgb,
            "fft": fft_rgb,
            "edges": edges_rgb,
        }

    # -----------------------------------------------------------------
    def get_gradcam(self, image_input):
        """
        Generate a Grad-CAM overlay for the given image.

        Args:
            image_input: File path or numpy array.

        Returns:
            np.ndarray: RGB image with Grad-CAM heatmap overlay (uint8),
                or the original image if pytorch-grad-cam is not installed.
        """
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
        except ImportError:
            print("⚠ pytorch-grad-cam not installed. Returning original image.")
            rgb, _ = self._load_rgb(image_input)
            return cv2.resize(rgb, (self.image_size, self.image_size))

        rgb, bgr = self._load_rgb(image_input)

        # Build input tensor (same as predict)
        augmented = self.rgb_transform(image=rgb)
        rgb_tensor = augmented["image"]

        if self.use_ela:
            ela_bgr = compute_ela(bgr)
            ela_rgb = cv2.cvtColor(ela_bgr, cv2.COLOR_BGR2RGB)
            ela_aug = self.ela_transform(image=ela_rgb)
            ela_tensor = ela_aug["image"]
            input_tensor = torch.cat([rgb_tensor, ela_tensor], dim=0)
        else:
            input_tensor = rgb_tensor

        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # Grad-CAM
        target_layer = self.model.backbone[-1]
        cam = GradCAM(model=self.model, target_layers=[target_layer])
        targets = [BinaryClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        # Overlay on de-normalized RGB image
        rgb_resized = cv2.resize(rgb, (self.image_size, self.image_size))
        rgb_float = rgb_resized.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

        return cam_image


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict: Real or Fake?")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--model_path", default="f:/CV_CP/models/best_model.pth")
    parser.add_argument("--no_ela", action="store_true")
    args = parser.parse_args()

    predictor = DeepfakePredictor(
        model_path=args.model_path,
        use_ela=not args.no_ela,
    )

    result = predictor.predict(args.image)

    print("\n" + "=" * 40)
    print("🔍 PREDICTION RESULT")
    print("=" * 40)
    emoji = "❌" if result["is_fake"] else "✅"
    print(f"  {emoji}  Label      : {result['label']}")
    print(f"      Confidence : {result['confidence']}%")
    print(f"      P(Fake)    : {result['probability']}")
    print(f"      Time       : {result['inference_time_ms']} ms")
    print("=" * 40)
