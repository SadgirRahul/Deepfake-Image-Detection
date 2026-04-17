"""
Deepfake Image Detector — Gradio Web Application
===================================================

Provides an interactive web interface where users can upload images
or use their webcam to detect deepfakes in real-time. Displays the
prediction along with CV feature visualizations (ELA, FFT, Edge Map)
and a Grad-CAM attention heatmap.

Usage::

    python scripts/app.py
    # Opens at http://localhost:7861 (default)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from src.predict import DeepfakePredictor


# =============================================================================
# Initialize predictor (loaded once at startup)
# =============================================================================

predictor = DeepfakePredictor(
    model_path="f:/CV_CP/models/best_model.pth",
    use_ela=True,
)


# =============================================================================
# Prediction handler
# =============================================================================

def analyze_image(image):
    """Main callback for the Gradio interface."""
    if image is None:
        return (
            {"Real": 0.5, "Fake": 0.5},
            "⚠️ No image provided",
            None, None, None, None,
        )

    result = predictor.predict(image)

    prob_fake = result["probability"]
    label_dict = {
        "Fake": float(prob_fake),
        "Real": float(1 - prob_fake),
    }

    emoji = "❌" if result["is_fake"] else "✅"
    confidence_text = (
        f"{emoji}  {result['label']}  —  "
        f"Confidence: {result['confidence']}%  |  "
        f"Inference: {result['inference_time_ms']} ms"
    )

    features = predictor.get_feature_visualizations(image)
    ela_img = features["ela"]
    fft_img = features["fft"]
    edge_img = features["edges"]

    gradcam_img = predictor.get_gradcam(image)

    return label_dict, confidence_text, ela_img, fft_img, edge_img, gradcam_img


# =============================================================================
# Gradio UI
# =============================================================================

DESCRIPTION = """
# 🔍 DeepFake Image Detector

Upload an image or use your webcam to detect whether an image is **real** or **AI-generated (fake)**
using deep learning and computer vision forensic analysis.

**How it works:** The model uses an EfficientNet-B0 backbone trained on 10,000 images, combined with
Error Level Analysis (ELA) to detect manipulation artifacts invisible to the human eye.
"""

FEATURE_INFO = """
### 🧪 Feature Analysis Guide

| Feature | What It Reveals |
|---------|----------------|
| **🔬 ELA** | Compression inconsistencies — manipulated regions glow brighter |
| **📊 FFT** | Frequency artifacts — GAN-generated images show unnatural spectral patterns |
| **🔲 Edges** | Boundary artifacts — deepfakes often have blurred or unnatural face edges |
| **🔥 Grad-CAM** | Model attention — highlights where the AI focused to make its decision |
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    title="DeepFake Image Detector",
    css="""
        .gradio-container { max-width: 1200px !important; }
        .gr-button-primary { font-size: 1.1em !important; }
    """,
) as app:
    gr.Markdown(DESCRIPTION)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📷 Upload Image or Use Webcam",
                type="numpy",
                sources=["upload", "webcam"],
                height=350,
            )
            analyze_btn = gr.Button(
                "🔍 Analyze Image",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=1):
            prediction_label = gr.Label(
                label="📊 Prediction",
                num_top_classes=2,
            )
            confidence_text = gr.Textbox(
                label="Result",
                interactive=False,
                lines=1,
            )

    gr.Markdown("---")
    gr.Markdown("### 🧪 Computer Vision Feature Analysis")

    with gr.Row(equal_height=True):
        ela_output = gr.Image(label="🔬 Error Level Analysis (ELA)", height=250)
        fft_output = gr.Image(label="📊 FFT Frequency Spectrum", height=250)
        edge_output = gr.Image(label="🔲 Edge Detection Map", height=250)
        gradcam_output = gr.Image(label="🔥 Grad-CAM Attention", height=250)

    gr.Markdown(FEATURE_INFO)

    analyze_btn.click(
        fn=analyze_image,
        inputs=[input_image],
        outputs=[
            prediction_label,
            confidence_text,
            ela_output,
            fft_output,
            edge_output,
            gradcam_output,
        ],
    )

    input_image.change(
        fn=analyze_image,
        inputs=[input_image],
        outputs=[
            prediction_label,
            confidence_text,
            ela_output,
            fft_output,
            edge_output,
            gradcam_output,
        ],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--server_port", type=int, default=7861)
    args = parser.parse_args()

    print(f"\n🚀 Starting Deepfake Detector Web App on port {args.server_port}...")
    print(f"   Open http://localhost:{args.server_port} in your browser\n")
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=args.server_port,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
            .gradio-container { max-width: 1200px !important; }
            .gr-button-primary { font-size: 1.1em !important; }
        """,
    )
