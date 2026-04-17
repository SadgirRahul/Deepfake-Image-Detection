"""
Computer Vision Feature Extraction for Deepfake Detection
==========================================================

This module implements forensic feature extraction techniques used to
distinguish real images from deepfake/manipulated images.

Techniques implemented:
    - Error Level Analysis (ELA)
    - Frequency Domain Analysis (2D FFT)
    - Edge Detection (Canny)
    - Local Binary Patterns (LBP) for texture analysis

All functions accept either a file path (str) or a NumPy array (BGR format)
as input, making them compatible with both disk-based and in-memory workflows.
"""

import cv2
import numpy as np

try:
    from skimage.feature import local_binary_pattern as skimage_lbp
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("⚠ scikit-image not found. LBP will use a basic fallback implementation.")


# =============================================================================
# Helper: Flexible image loading
# =============================================================================

def _load_image(image_input):
    """
    Load an image from a file path or accept a NumPy array directly.

    Args:
        image_input (str or np.ndarray): File path to an image, or a BGR
            NumPy array (as returned by cv2.imread).

    Returns:
        np.ndarray: The image in BGR format (uint8).

    Raises:
        ValueError: If the input is neither a valid path nor a NumPy array.
        FileNotFoundError: If the path does not point to a readable image.
    """
    if isinstance(image_input, np.ndarray):
        return image_input.copy()
    elif isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_input}")
        return img
    else:
        raise ValueError(
            f"Expected a file path (str) or NumPy array, got {type(image_input)}"
        )


# =============================================================================
# 1. Error Level Analysis (ELA)
# =============================================================================

def compute_ela(image_input, quality=90):
    """
    Perform Error Level Analysis on an image.

    ELA works by re-saving the image at a known JPEG quality level and
    computing the pixel-wise difference between the original and the
    re-compressed version. Manipulated regions typically show higher
    error levels because they were compressed at a different quality.

    Args:
        image_input (str or np.ndarray): Path to the image file, or a BGR
            NumPy array.
        quality (int): JPEG re-compression quality (1-100). Default is 90.

    Returns:
        np.ndarray: The ELA difference image (BGR, uint8, same size as input).
            Pixel differences are scaled by 20x for visibility.

    Raises:
        FileNotFoundError: If the image path is invalid.
        ValueError: If the image cannot be encoded to JPEG.
    """
    try:
        original = _load_image(image_input)

        # Re-compress in memory as JPEG at the specified quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, encoded = cv2.imencode(".jpg", original, encode_params)
        if not success:
            raise ValueError("Failed to encode image to JPEG buffer.")

        # Decode the re-compressed image
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Compute absolute pixel-wise difference
        diff = cv2.absdiff(original, recompressed)

        # Scale by 20x for visibility and clip to [0, 255]
        ela_image = np.clip(diff.astype(np.float32) * 20.0, 0, 255).astype(np.uint8)

        return ela_image

    except Exception as e:
        print(f"❌ ELA computation failed: {e}")
        raise


# =============================================================================
# 2. Frequency Domain Analysis (2D FFT)
# =============================================================================

def compute_fft_spectrum(image_input):
    """
    Compute the 2D FFT magnitude spectrum of an image.

    GAN-generated images often exhibit distinct periodic patterns in the
    frequency domain that are absent in real camera captures. The magnitude
    spectrum can reveal these artifacts as unusual peaks or grid-like
    structures.

    Args:
        image_input (str or np.ndarray): Path to the image file, or a BGR
            NumPy array.

    Returns:
        np.ndarray: The log-magnitude FFT spectrum image (grayscale, uint8),
            with zero frequency centered.

    Raises:
        FileNotFoundError: If the image path is invalid.
    """
    try:
        img = _load_image(image_input)

        # Convert to grayscale for frequency analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute 2D FFT
        f_transform = np.fft.fft2(gray.astype(np.float32))

        # Shift zero-frequency component to the center
        f_shift = np.fft.fftshift(f_transform)

        # Compute log-magnitude spectrum
        magnitude = np.abs(f_shift)
        spectrum = 20.0 * np.log(1.0 + magnitude)

        # Normalize to 0-255 for display
        spectrum_min = spectrum.min()
        spectrum_max = spectrum.max()
        if spectrum_max - spectrum_min > 0:
            spectrum_normalized = (
                (spectrum - spectrum_min) / (spectrum_max - spectrum_min) * 255.0
            )
        else:
            spectrum_normalized = np.zeros_like(spectrum)

        return spectrum_normalized.astype(np.uint8)

    except Exception as e:
        print(f"❌ FFT spectrum computation failed: {e}")
        raise


# =============================================================================
# 3. Edge Detection (Canny)
# =============================================================================

def compute_edge_map(image_input):
    """
    Compute a Canny edge map of an image.

    Deepfakes often exhibit unnatural edge transitions, especially around
    face boundaries where the generated face meets the original background.
    Edge maps can highlight these blending artifacts.

    Args:
        image_input (str or np.ndarray): Path to the image file, or a BGR
            NumPy array.

    Returns:
        np.ndarray: Binary edge map (grayscale, uint8, values 0 or 255).

    Raises:
        FileNotFoundError: If the image path is invalid.
    """
    try:
        img = _load_image(image_input)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        return edges

    except Exception as e:
        print(f"❌ Edge map computation failed: {e}")
        raise


# =============================================================================
# 4. Local Binary Patterns (LBP) — Texture Analysis
# =============================================================================

def compute_lbp(image_input, radius=3, n_points=24):
    """
    Compute Local Binary Pattern (LBP) texture descriptor of an image.

    LBP encodes local texture information by comparing each pixel with its
    circular neighbors. Deepfake faces often have unnaturally smooth skin
    textures or inconsistent micro-patterns that LBP can capture.

    Uses scikit-image's implementation if available, otherwise falls back
    to a basic manual implementation.

    Args:
        image_input (str or np.ndarray): Path to the image file, or a BGR
            NumPy array.
        radius (int): Radius of the circular LBP neighborhood. Default is 3.
        n_points (int): Number of neighbor points to sample. Default is 24.

    Returns:
        np.ndarray: LBP image (grayscale, uint8, normalized to 0-255).

    Raises:
        FileNotFoundError: If the image path is invalid.
    """
    try:
        img = _load_image(image_input)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if HAS_SKIMAGE:
            # Use scikit-image's optimized LBP implementation
            lbp = skimage_lbp(gray, n_points, radius, method="uniform")
        else:
            # Fallback: basic manual LBP implementation
            lbp = _manual_lbp(gray, radius, n_points)

        # Normalize to 0-255 for display
        lbp_min = lbp.min()
        lbp_max = lbp.max()
        if lbp_max - lbp_min > 0:
            lbp_normalized = (lbp - lbp_min) / (lbp_max - lbp_min) * 255.0
        else:
            lbp_normalized = np.zeros_like(lbp, dtype=np.float64)

        return lbp_normalized.astype(np.uint8)

    except Exception as e:
        print(f"❌ LBP computation failed: {e}")
        raise


def _manual_lbp(gray, radius, n_points):
    """
    Basic manual LBP implementation as a fallback when scikit-image
    is not available.

    For each pixel, samples `n_points` neighbors on a circle of the given
    `radius` and encodes a binary pattern based on intensity comparisons
    with the center pixel.

    Args:
        gray (np.ndarray): Grayscale image (uint8).
        radius (int): Radius of the circular neighborhood.
        n_points (int): Number of sampling points on the circle.

    Returns:
        np.ndarray: LBP result (float64, same shape as input).
    """
    rows, cols = gray.shape
    lbp = np.zeros((rows, cols), dtype=np.float64)

    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            center = gray[i, j]
            binary_code = 0

            for k in range(n_points):
                # Compute neighbor coordinates on the circle
                angle = 2.0 * np.pi * k / n_points
                ni = int(round(i - radius * np.sin(angle)))
                nj = int(round(j + radius * np.cos(angle)))

                # Clamp to image boundaries
                ni = max(0, min(ni, rows - 1))
                nj = max(0, min(nj, cols - 1))

                # Compare: set bit if neighbor >= center
                if gray[ni, nj] >= center:
                    binary_code |= (1 << k)

            lbp[i, j] = binary_code

    return lbp


# =============================================================================
# 5. Combined Feature Extraction
# =============================================================================

def extract_all_features(image_input):
    """
    Extract all CV-based forensic features from a single image.

    Runs ELA, FFT spectrum, edge detection, and LBP analysis, returning
    all results in a dictionary for downstream consumption.

    Args:
        image_input (str or np.ndarray): Path to the image file, or a BGR
            NumPy array.

    Returns:
        dict: A dictionary with keys:
            - "ela"   : ELA difference image (BGR, uint8)
            - "fft"   : FFT magnitude spectrum (grayscale, uint8)
            - "edges" : Canny edge map (grayscale, uint8)
            - "lbp"   : LBP texture image (grayscale, uint8)

    Raises:
        FileNotFoundError: If the image path is invalid.
    """
    features = {}

    try:
        features["ela"] = compute_ela(image_input)
    except Exception as e:
        print(f"⚠ Skipping ELA: {e}")
        features["ela"] = None

    try:
        features["fft"] = compute_fft_spectrum(image_input)
    except Exception as e:
        print(f"⚠ Skipping FFT: {e}")
        features["fft"] = None

    try:
        features["edges"] = compute_edge_map(image_input)
    except Exception as e:
        print(f"⚠ Skipping Edge Map: {e}")
        features["edges"] = None

    try:
        features["lbp"] = compute_lbp(image_input)
    except Exception as e:
        print(f"⚠ Skipping LBP: {e}")
        features["lbp"] = None

    return features


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        # Default: try a sample from the dataset
        test_path = "F:/Dataset/train/real"
        if os.path.isdir(test_path):
            files = [f for f in os.listdir(test_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                test_image = os.path.join(test_path, files[0])
            else:
                print("No images found in default test path.")
                sys.exit(1)
        else:
            print(f"Default path not found: {test_path}")
            print("Usage: python features.py <image_path>")
            sys.exit(1)
    else:
        test_image = sys.argv[1]

    print(f"🔍 Extracting features from: {test_image}")
    results = extract_all_features(test_image)

    for name, img in results.items():
        if img is not None:
            print(f"  ✅ {name:6s} → shape: {img.shape}, dtype: {img.dtype}")
        else:
            print(f"  ❌ {name:6s} → failed")

    print("\n🎉 Feature extraction complete!")
