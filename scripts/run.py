"""Deepfake Detection — Unified Entry Point
==========================================

A single CLI script to train, evaluate, predict, or launch the web app.

Usage::

    python scripts/run.py --mode train                          # Train the model
    python scripts/run.py --mode train --epochs 30 --no_ela     # Train without ELA
    python scripts/run.py --mode evaluate                       # Evaluate on test set
    python scripts/run.py --mode predict --image photo.jpg      # Single prediction
    python scripts/run.py --mode app                            # Launch Gradio web app
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def print_banner():
    print()
    print("=" * 58)
    print("  🔍  DEEPFAKE IMAGE DETECTION SYSTEM")
    print("  📚  Computer Vision Course Project")
    print("  🧠  Model: EfficientNet-B0 + ELA (6-channel)")
    print("=" * 58)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Image Detection — End-to-End System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "evaluate", "predict", "app"],
        help=(
            "Operation mode:\n"
            "  train    — Train the model on the dataset\n"
            "  evaluate — Evaluate on the test set + generate plots\n"
            "  predict  — Predict a single image (requires --image)\n"
            "  app      — Launch the Gradio web interface"
        ),
    )
    parser.add_argument("--image", default=None, help="Image path (for predict mode)")
    parser.add_argument("--data_dir", default="F:/Dataset", help="Dataset root directory")
    parser.add_argument("--model_path", default="f:/CV_CP/models/best_model.pth", help="Model weights path")
    parser.add_argument("--save_dir", default="f:/CV_CP/models", help="Model save directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--epochs", type=int, default=25, help="Total training epochs")
    parser.add_argument("--phase1_epochs", type=int, default=8, help="Phase 1 (head-only) epochs")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate for classifier head")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for backbone")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--no_ela", action="store_true", help="Disable ELA channels (use RGB only)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 'auto', 'cpu', or 'cuda'",
    )
    parser.add_argument("--port", type=int, default=7861, help="Port for the web app")

    args = parser.parse_args()

    print_banner()

    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    use_ela = not args.no_ela

    if args.mode == "train":
        from src.train import train_model, plot_training_curves

        config = {
            "data_dir": args.data_dir,
            "save_dir": args.save_dir,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "total_epochs": args.epochs,
            "phase1_epochs": args.phase1_epochs,
            "lr_head": args.lr_head,
            "lr_backbone": args.lr_backbone,
            "weight_decay": 1e-4,
            "patience": args.patience,
            "use_ela": use_ela,
            "device": device,
            "num_workers": args.num_workers,
        }

        model, history = train_model(config)
        plot_training_curves(history)

    elif args.mode == "evaluate":
        from src.model import get_model
        from src.dataset import get_dataloaders
        from src.evaluate import (
            evaluate_model,
            plot_confusion_matrix,
            plot_roc_curve,
            plot_precision_recall_curve,
            show_misclassified,
            visualize_gradcam_samples,
        )
        import torch

        model = get_model(use_ela=use_ela, pretrained=False, device=device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print(f"✅ Model loaded from: {args.model_path}")

        loaders = get_dataloaders(
            root_dir=args.data_dir,
            batch_size=args.batch_size,
            use_ela=use_ela,
            num_workers=args.num_workers,
        )

        results = evaluate_model(model, loaders["test"], device)

        plot_confusion_matrix(results["y_true"], results["y_pred"])
        plot_roc_curve(results["y_true"], results["y_prob"])
        plot_precision_recall_curve(results["y_true"], results["y_prob"])
        show_misclassified(model, loaders["test"], device)
        visualize_gradcam_samples(model, loaders["test"], device)

        print("\n🎉 All evaluation artifacts saved in f:/CV_CP/results/")

    elif args.mode == "predict":
        if args.image is None:
            print("❌ Error: --image is required for predict mode.")
            print("   Example: python scripts/run.py --mode predict --image photo.jpg")
            sys.exit(1)

        from src.predict import DeepfakePredictor

        predictor = DeepfakePredictor(
            model_path=args.model_path,
            use_ela=use_ela,
            device=device,
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

    elif args.mode == "app":
        import subprocess

        app_path = Path(__file__).with_name("app.py")
        print(f"🚀 Launching Gradio app on http://localhost:{args.port}")
        subprocess.run(
            [sys.executable, str(app_path), f"--server_port={args.port}"],
            cwd=str(ROOT),
            check=False,
        )


if __name__ == "__main__":
    main()
